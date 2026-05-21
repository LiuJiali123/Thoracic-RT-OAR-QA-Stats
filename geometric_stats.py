import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests
import numpy as np

# ==========================================
# 1. 加载与清洗
# ==========================================
# file_path = '0_All_Cases_Consolidated_Metrics_no08bp.xlsx'
file_path = '0_All_Cases_Consolidated_Metrics2.xlsx'

df = pd.read_excel(file_path, sheet_name='All_Data_Raw (Plotting)')

# 预处理
df['Unique_ID'] = df['Case'].astype(str) + "_" + df['Hospital_Number'].astype(str)
valid_phases = ['Real-World', 'Manual (Post-Education)', 'Education+AI-Assisted']
df = df[df['Phase'].isin(valid_phases)]
df = df[df['Hospital_Number'] != 'DL']

# 器官清洗
df = df[~df['Organ'].isin([ 'BrachialPlexus'])]#'Cord_Full', 'Cord_Mis',
df.loc[df['Organ'] == 'Cord_Com', 'Organ'] = 'SpinalCord'
df.loc[df['Organ'] == 'BrachialPlexus_Cancer_Lateral', 'Organ'] = 'BrachialPlexus (Lat)'

# ==========================================
# 2. 统计检验核心函数 (含正态性检测 + FDR)
# ==========================================
def calculate_pairwise_stats(data_series1, data_series2):
    """
    输入: 两个配对的数据序列 (Series)
    输出: P-value (原始)
    逻辑:
    1. 计算差值 (diff)
    2. Shapiro-Wilk 检验差值正态性
    3. 如果正态 -> Paired T-test
    4. 如果非正态 -> Wilcoxon Signed-Rank Test
    """
    if len(data_series1) < 3: return np.nan
    
    diff = data_series1 - data_series2
    
    # Shapiro-Wilk test for normality
    # 如果 N < 3，shapiro 可能会报错或不准，前面已拦截
    try:
        _, p_shapiro = shapiro(diff)
        is_normal = p_shapiro > 0.05
    except:
        is_normal = False # Fallback to non-parametric

    if is_normal:
        try:
            _, p_val = ttest_rel(data_series1, data_series2)
        except:
            p_val = np.nan
    else:
        try:
            _, p_val = wilcoxon(data_series1, data_series2)
        except:
            # wilcoxon fails if all diffs are zero
            if np.all(diff == 0):
                p_val = 1.0
            else:
                p_val = np.nan
            
    return p_val

# def get_stats_str(data_series):
#     if len(data_series) == 0: return "-"
#     return f"{data_series.mean():.2f} ± {data_series.std():.2f} (N={len(data_series)})"

# 修改前：
# def get_stats_str(data_series):
#     if len(data_series) == 0: return "-"
#     return f"{data_series.mean():.2f} ± {data_series.std():.2f} (N={len(data_series)})"

# 修改后：
# def get_stats_str(data_series, metric_name):
#     if len(data_series) == 0: return "-"
#     # DSC保留2位，其他(HD95/ASD)保留1位
#     precision = 2 if metric_name == 'DSC' else 1
#     return f"{data_series.mean():.{precision}f} ± {data_series.std():.{precision}f} (N={len(data_series)})"

# ==========================================
# 新增：非参数 Bootstrap 方法计算 95% CI
# ==========================================
def calculate_bootstrap_ci(data_series, n_iterations=1000):
    """
    使用 non-parametric bootstrap 方法计算 95% CI
    每次允许重复地随机抽取与原数据集同等大小的样本，循环 1000 次，
    然后从相对频率分布中取 2.5% 和 97.5% 分位数作为 95% CI。
    """
    if len(data_series) == 0: 
        return np.nan, np.nan
        
    values = data_series.dropna().values
    n_size = len(values)
    
    if n_size == 0:
        return np.nan, np.nan

    # 使用向量化操作一次性生成 1000 次重复抽样，大幅提高运算速度
    samples = np.random.choice(values, size=(n_iterations, n_size), replace=True)
    bootstrapped_means = np.mean(samples, axis=1)
    
    # 提取 95% CI 的上下限 (即分布的 2.5th 和 97.5th percentiles)
    lower_ci = np.percentile(bootstrapped_means, 2.5)
    upper_ci = np.percentile(bootstrapped_means, 97.5)
    
    return lower_ci, upper_ci

# ==========================================
# 修改后：get_stats_str (整合 mean, SD, range 和 95% CI)
# ==========================================
def get_stats_str(data_series, metric_name):
    if len(data_series) == 0: return "-"
    
    # DSC保留2位，其他(HD95/ASD)保留1位
    precision = 2 if metric_name == 'DSC' else 1
    
    # 基础统计量
    mean_val = data_series.mean()
    sd_val = data_series.std()
    min_val = data_series.min()
    max_val = data_series.max()
    
    # 获取 Bootstrap 95% CI
    ci_lower, ci_upper = calculate_bootstrap_ci(data_series)
    
    # 格式拼接：Mean ± SD [Range] (95% CI: lower - upper) (N=X)
    return (f"{mean_val:.{precision}f} ± {sd_val:.{precision}f} "
            f"[{min_val:.{precision}f} - {max_val:.{precision}f}] "
            f"(95% CI: {ci_lower:.{precision}f} - {ci_upper:.{precision}f}) "
            f"(N={len(data_series)})")

# ==========================================
# 3. 收集所有 P 值用于 FDR 校正
# ==========================================
results_buffer = [] # 暂存计算结果，不含 FDR P值
p_values_list = []  # 收集所有的 P 值，用于一次性 FDR
p_values_index_map = [] # 记录 P 值对应的 (行索引, 对比名称)

# metrics = ['DSC', 'HD95', 'ASD',"Volume [cm³]", "Max Dose [Gy]","Mean Dose [Gy]","Vol > 1 Gy (%)","Vol > 5 Gy (%)","Vol > 20 Gy (%)", "Vol > 30 Gy (%)", "Vol > 40 Gy (%)"]
metrics = ['DSC', 'HD95', 'ASD']
organ_list = sorted(df['Organ'].unique())
case_list = sorted(df['Case'].unique()) # 🌟 新增：获取所有独立的 Case 列表
comparisons = [
    ('Real-World', 'Manual (Post-Education)', 'RW vs Manual'),
    ('Manual (Post-Education)', 'Education+AI-Assisted', 'Manual vs AI'),
    ('Real-World', 'Education+AI-Assisted', 'RW vs AI')
]

for organ in organ_list:
    for case_id in case_list:       # 🌟 新增：加一层对 Case 的循环                                                   
        for metric in metrics:
#             row = {'Organ': organ, 'Metric': metric}
            row = {'Organ': organ, 'Case': case_id, 'Metric': metric}  # 🌟 这里把 case_id 存进字典！
            for p1, p2, label in comparisons:
                # 提取配对数据
#                 d1_all = df[(df['Organ'] == organ) & (df['Phase'] == p1)]
#                 d2_all = df[(df['Organ'] == organ) & (df['Phase'] == p2)]
# 🌟 修改：提取配对数据时，加入 & (df['Case'] == case_id) 筛选条件
                d1_all = df[(df['Organ'] == organ) & (df['Case'] == case_id) & (df['Phase'] == p1)]
                d2_all = df[(df['Organ'] == organ) & (df['Case'] == case_id) & (df['Phase'] == p2)]
                merged = pd.merge(d1_all[['Unique_ID', metric]], d2_all[['Unique_ID', metric]], on='Unique_ID', suffixes=('_1', '_2'))

                # 计算原始 P 值
                raw_p = np.nan
                if len(merged) >= 3:
                    raw_p = calculate_pairwise_stats(merged[f'{metric}_1'], merged[f'{metric}_2'])

                # 记录结果
                prefix = f"[{label}] "
    #             row[f"{prefix}{p1} (All)"] = get_stats_str(d1_all[metric])
    #             row[f"{prefix}{p2} (All)"] = get_stats_str(d2_all[metric])
    #             row[f"{prefix}{p1} (Paired)"] = get_stats_str(merged[f'{metric}_1'])
    #             row[f"{prefix}{p2} (Paired)"] = get_stats_str(merged[f'{metric}_2'])
                # 修改前：
                # row[f"{prefix}{p1} (All)"] = get_stats_str(d1_all[metric])
                # row[f"{prefix}{p2} (All)"] = get_stats_str(d2_all[metric])
                # row[f"{prefix}{p1} (Paired)"] = get_stats_str(merged[f'{metric}_1'])
                # row[f"{prefix}{p2} (Paired)"] = get_stats_str(merged[f'{metric}_2'])

                # 修改后：
                row[f"{prefix}{p1} (All)"] = get_stats_str(d1_all[metric], metric)
                row[f"{prefix}{p2} (All)"] = get_stats_str(d2_all[metric], metric)
                row[f"{prefix}{p1} (Paired)"] = get_stats_str(merged[f'{metric}_1'], metric)
                row[f"{prefix}{p2} (Paired)"] = get_stats_str(merged[f'{metric}_2'], metric)



                row[f"{prefix}Raw_P"] = raw_p # 暂存 Raw P

                # 收集有效 P 值用于 FDR
                if not np.isnan(raw_p):
                    p_values_list.append(raw_p)
                    # 记录这个 P 值属于哪一行 (len(results_buffer)) 和哪一列 (label)
                    p_values_index_map.append((len(results_buffer), label))

            results_buffer.append(row)

# ==========================================
# 4. 执行 Benjamini-Hochberg FDR 校正
# ==========================================
# 对收集到的所有有效 P 值进行校正
reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')

# 将校正后的 P 值填回表格
# 先初始化所有 P-value 列为 "N<3" 或 "Err"
for row in results_buffer:
    for _, _, label in comparisons:
        row[f"[{label}] P-value"] = "N<3" if pd.isna(row[f"[{label}] Raw_P"]) else "Err"

# 填入 FDR P 值
for idx, (row_idx, label) in enumerate(p_values_index_map):
    corrected_p = pvals_corrected[idx]
    
    # 格式化
    sig = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else ""
    p_str = f"{corrected_p:.3f}{sig}"
    
    results_buffer[row_idx][f"[{label}] P-value"] = p_str

# ==========================================
# 5. 生成最终 DataFrame 并保存
# ==========================================
df_final = pd.DataFrame(results_buffer)

# 清理掉临时的 Raw_P 列 (如果不想展示)
cols_to_drop = [c for c in df_final.columns if 'Raw_P' in c]
# df_final.drop(columns=cols_to_drop, inplace=True)

print("\n📊 严格统计流程 (Shapiro -> T/Wilcoxon -> FDR) 完成。")
print(df_final[['Organ', 'Metric', '[Manual vs AI] P-value']].head().to_string())

output_file = "Table_Geometric_Strict_Stats_FDR2.csv"
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 最终表格已保存: {output_file}")
                                                   
                                                   
## ==========================================
# 6. 新增功能：计算特定 Case (1106, 3502) 指定核心 OAR 的平均指标
# ==========================================
print("\n" + "="*50)
print("🌟 开始计算特定 Case 的核心 OAR 宏观平均指标")
print("="*50)

# 定义需要重点分析的 Case 和指定的 5 个核心 OAR
target_cases = ['1106', '3502']
target_oars = ['BrachialPlexus (Lat)', 'Esophagus', 'Heart', 'Lungs_GTV', 'SpinalCord']

df['Case_str'] = df['Case'].astype(str) 

# 🌟 核心修改：同时过滤 Case 和指定的 Organ
target_df = df[(df['Case_str'].isin(target_cases)) & (df['Organ'].isin(target_oars))]

if not target_df.empty:
    # 第一步：计算每个 Case, Phase, 核心 Organ 的均值
    mean_per_organ = target_df.groupby(['Case_str', 'Phase', 'Organ'])[metrics].mean().reset_index()
    
    # 第二步：计算每个 Case, Phase 下这 5 个 Organ 的平均值 (sum / N)
    mean_of_oars = mean_per_organ.groupby(['Case_str', 'Phase'])[metrics].mean().reset_index()
    
    # 额外统计一下实际包含的 OAR 数量 (用来核对是否每个 Phase 都成功匹配到了这 5 个器官)
    count_oars = mean_per_organ.groupby(['Case_str', 'Phase'])['Organ'].count().reset_index()
    count_oars.rename(columns={'Organ': 'Included_OAR_Count'}, inplace=True)
    
    # 合并指标和器官数量
    summary_df = pd.merge(mean_of_oars, count_oars, on=['Case_str', 'Phase'])
    summary_df.rename(columns={'Case_str': 'Case'}, inplace=True)
    
    # 按照 Phase 顺序排序，方便阅读对比
    summary_df['Phase'] = pd.Categorical(summary_df['Phase'], categories=valid_phases, ordered=True)
    summary_df = summary_df.sort_values(['Case', 'Phase']).reset_index(drop=True)
    
    # 格式化输出：DSC保留2位小数，ASD和HD95保留1位小数
    if 'DSC' in summary_df.columns:
        summary_df['DSC'] = summary_df['DSC'].round(2)
    if 'ASD' in summary_df.columns:
        summary_df['ASD'] = summary_df['ASD'].round(1)
    if 'HD95' in summary_df.columns:
        summary_df['HD95'] = summary_df['HD95'].round(1)
    
    print("\n📊 特定 Case (1106, 3502) 在 5 个核心 OAR 上的整体均值：")
    print(summary_df.to_string())
    
    # 保存结果到单独的 CSV 文件
    summary_output_file = "Table_Case_1106_3502_Core5_OAR_Averages.csv"
    summary_df.to_csv(summary_output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 核心 OAR 整体分析已保存至: {summary_output_file}")
else:
    print(f"\n⚠️ 未在数据中找到 Case {target_cases} 或指定的 5 个 OAR，请检查数据源。")

# 清理临时列
df.drop(columns=['Case_str'], inplace=True, errors='ignore')
                                                   
            
                                                   
# ==========================================
# 7. 新增功能：针对特定 Case 的核心 5 个 OAR 进行整体配对统计检验 (含 FDR)
# ==========================================
print("\n" + "="*50)
print("🌟 开始计算特定 Case 核心 5 个 OAR 的配对统计检验")
print("="*50)

target_cases = ['1106', '3502']
target_oars = ['BrachialPlexus (Lat)', 'Esophagus', 'Heart', 'Lungs_GTV', 'SpinalCord']
df['Case_str'] = df['Case'].astype(str)

case_stats_buffer = []
case_p_values_list = []
case_p_values_index_map = []

for case_id in target_cases:
    for metric in metrics:
        row = {'Case': case_id, 'Metric': metric, 'Target_OAR_Count': len(target_oars)}
        
        for p1, p2, label in comparisons:
            # 提取特定 Case 下，指定的 5 个 OAR 在两个 Phase 的数据
            d1 = df[(df['Case_str'] == case_id) & (df['Organ'].isin(target_oars)) & (df['Phase'] == p1)]
            d2 = df[(df['Case_str'] == case_id) & (df['Organ'].isin(target_oars)) & (df['Phase'] == p2)]
            
            # 因为是在同一个 Case 内比较整体水平，所以以 Organ 为基准进行配对 (Pairing by Organ)
            merged = pd.merge(d1[['Organ', metric]], d2[['Organ', metric]], on='Organ', suffixes=('_1', '_2'))
            
            raw_p = np.nan
            # 只有配对成功的器官数量 >= 3 才计算 P 值
            if len(merged) >= 3:
                raw_p = calculate_pairwise_stats(merged[f'{metric}_1'], merged[f'{metric}_2'])
            
            prefix = f"[{label}] "
            # 记录均值差异（方便在表格里直观看到数值变化）
            row[f"{prefix}{p1} Mean"] = f"{merged[f'{metric}_1'].mean():.2f}" if not merged.empty else "-"
            row[f"{prefix}{p2} Mean"] = f"{merged[f'{metric}_2'].mean():.2f}" if not merged.empty else "-"
            row[f"{prefix}Raw_P"] = raw_p
            
            # 收集有效 P 值用于 FDR
            if not np.isnan(raw_p):
                case_p_values_list.append(raw_p)
                case_p_values_index_map.append((len(case_stats_buffer), label))
                
        case_stats_buffer.append(row)

# 执行本组独立计算的 FDR 校正
if case_p_values_list:
    reject_case, pvals_corrected_case, _, _ = multipletests(case_p_values_list, alpha=0.05, method='fdr_bh')

    # 填回 FDR 校正后的 P 值
    for row in case_stats_buffer:
        for _, _, label in comparisons:
            row[f"[{label}] P-value"] = "N<3" if pd.isna(row[f"[{label}] Raw_P"]) else "Err"

    for idx, (row_idx, label) in enumerate(case_p_values_index_map):
        corrected_p = pvals_corrected_case[idx]
        sig = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else ""
        case_stats_buffer[row_idx][f"[{label}] P-value"] = f"{corrected_p:.3f}{sig}"
else:
    print("⚠️ 警告：没有产生有效的 P 值，可能是由于数据缺失导致器官匹配数量不足。")

# 生成最终 DataFrame 并保存
df_case_stats = pd.DataFrame(case_stats_buffer)

# 移除 Raw_P 列以保持最终表格整洁
cols_to_drop_case = [c for c in df_case_stats.columns if 'Raw_P' in c]
df_case_stats.drop(columns=cols_to_drop_case, inplace=True, errors='ignore')

print("\n📊 特定 Case (1106, 3502) 核心 5 器官宏观统计检验结果：")
print(df_case_stats[['Case', 'Metric', '[Manual vs AI] P-value']].head().to_string())

case_stats_output = "Table_Case_1106_3502_Core5_Stats_FDR.csv"
df_case_stats.to_csv(case_stats_output, index=False, encoding='utf-8-sig')
print(f"\n✅ 宏观统计检验结果已保存至: {case_stats_output}")

# 清理临时列
df.drop(columns=['Case_str'], inplace=True, errors='ignore')                               
                                    

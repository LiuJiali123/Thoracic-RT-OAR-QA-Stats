import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. Load Data
# ==========================================
file_path_stats = '../6_Cleaned_Paired_DSC_and_Mean_Dose_AI_Comperison_with_Time_1106_and_3502.csv'
file_path_all = '../6_Cleaned_Time_all_nopaired.csv'

df_stats = pd.read_csv(file_path_stats)
df_all = pd.read_csv(file_path_all)

# 清洗数据：强制转换为数字，并将秒转换为分钟
df_all['Time_Manual_Min'] = pd.to_numeric(df_all['Time_WithoutAI'], errors='coerce') / 60
df_all['Time_AI_Min'] = pd.to_numeric(df_all['Time_WithAI'], errors='coerce') / 60

df_stats['Time_Manual_Min'] = pd.to_numeric(df_stats['Time_WithoutAI'], errors='coerce') / 60
df_stats['Time_AI_Min'] = pd.to_numeric(df_stats['Time_WithAI'], errors='coerce') / 60


# ==========================================
# 新增：Bootstrap 计算与格式化辅助函数
# ==========================================
def calculate_bootstrap_ci(data_series, n_iterations=1000):
    values = data_series.dropna().values
    n_size = len(values)
    if n_size == 0:
        return np.nan, np.nan
    
    samples = np.random.choice(values, size=(n_iterations, n_size), replace=True)
    bootstrapped_means = np.mean(samples, axis=1)
    return np.percentile(bootstrapped_means, 2.5), np.percentile(bootstrapped_means, 97.5)

def get_time_stats_str(series):
    series = series.dropna()
    if len(series) == 0: return "-"
    
    m, sd = series.mean(), series.std()
    mi, ma = series.min(), series.max()
    ci_l, ci_u = calculate_bootstrap_ci(series)
    
    return f"{m:.1f} ± {sd:.1f} [{mi:.1f} - {ma:.1f}] (95% CI: {ci_l:.1f} - {ci_u:.1f}) min"

# ==========================================
# 2. Case Level Analysis
# ==========================================
# 分组求和
df_case_stats = df_stats.groupby(['Case', 'Hospital_Number'])[['Time_Manual_Min', 'Time_AI_Min']].sum().reset_index()
df_case_all = df_all.groupby(['Case', 'Hospital_Number'])[['Time_Manual_Min', 'Time_AI_Min']].sum(min_count=1).reset_index()

case_results = []
cases = [1106, 3502]

for c in cases:
    # --- 描述性统计 (使用全部数据) ---
    subset_all = df_case_all[df_case_all['Case'] == c]
    m_man = subset_all['Time_Manual_Min'].mean()
    sd_man = subset_all['Time_Manual_Min'].std()
    m_ai = subset_all['Time_AI_Min'].mean()
    sd_ai = subset_all['Time_AI_Min'].std()
    derived_saving = m_man - m_ai
    derived_pct = (derived_saving / m_man * 100) if m_man != 0 else 0
    
    # --- 统计检验 (使用配对数据) ---
    subset_stats = df_case_stats[df_case_stats['Case'] == c]
    # 手动处理 NaN，确保配对数据完整
    valid_data = subset_stats[['Time_Manual_Min', 'Time_AI_Min']].dropna()
    t_man_p = valid_data['Time_Manual_Min']
    t_ai_p = valid_data['Time_AI_Min']
    diff = t_man_p - t_ai_p
    
    if len(diff) < 3: 
        p = np.nan
    elif shapiro(diff)[1] > 0.05: 
        _, p = ttest_rel(t_man_p, t_ai_p) # 移除 nan_policy
    else: 
        _, p = wilcoxon(t_man_p, t_ai_p) # 移除 nan_policy
        
#     case_results.append({
#         'Analysis Level': 'Case Level',
#         'Subgroup': f'Case {c} ({"Adaptive" if c==1106 else "SBRT"})',
#         'Manual Time (Mean ± SD)': f"{m_man:.1f} ± {sd_man:.1f} min",
#         'AI-Assisted Time (Mean ± SD)': f"{m_ai:.1f} ± {sd_ai:.1f} min",
#         'Time Saving (Mean)': f"{derived_saving:.1f} min",
#         'Time Saving (%)': f"{derived_pct:.1f}%",
#         'Raw P': p
#     })
# 替换原有的 case_results.append 如下：
    case_results.append({
        'Analysis Level': 'Case Level',
        'Subgroup': f'Case {c} ({"Adaptive" if c==1106 else "SBRT"})',
        'Manual Time': get_time_stats_str(subset_all['Time_Manual_Min']),
        'AI-Assisted Time': get_time_stats_str(subset_all['Time_AI_Min']),
        'Time Saving (Mean)': f"{derived_saving:.1f} min",
        'Time Saving (%)': f"{derived_pct:.1f}%",
        'Raw P': p
    })
# --- Combined Total ---
m_man_comb = df_case_all['Time_Manual_Min'].mean()
sd_man_comb = df_case_all['Time_Manual_Min'].std()
m_ai_comb = df_case_all['Time_AI_Min'].mean()
sd_ai_comb = df_case_all['Time_AI_Min'].std()
derived_saving_comb = m_man_comb - m_ai_comb
derived_pct_comb = (derived_saving_comb / m_man_comb * 100)

# 合并数据的配对检验
# valid_comb = df_case_stats[['Time_Manual_Min', 'Time_AI_Min']].dropna()
# _, p_comb = wilcoxon(valid_comb['Time_Manual_Min'], valid_comb['Time_AI_Min'])
# 合并数据的配对检验 (补齐 Shapiro 自适应正态检验)
valid_comb = df_case_stats[['Time_Manual_Min', 'Time_AI_Min']].dropna()
t_man_comb = valid_comb['Time_Manual_Min']
t_ai_comb = valid_comb['Time_AI_Min']
diff_comb = t_man_comb - t_ai_comb

if len(diff_comb) < 3: 
    p_comb = np.nan
else:
    try:
        _, p_shapiro = shapiro(diff_comb)
        is_normal = p_shapiro > 0.05
    except:
        is_normal = False
        
    if is_normal: 
        _, p_comb = ttest_rel(t_man_comb, t_ai_comb)
    else: 
        _, p_comb = wilcoxon(t_man_comb, t_ai_comb)
# case_results.append({
#     'Analysis Level': 'Case Level',
#     'Subgroup': 'Combined Total',
#     'Manual Time (Mean ± SD)': f"{m_man_comb:.1f} ± {sd_man_comb:.1f} min",
#     'AI-Assisted Time (Mean ± SD)': f"{m_ai_comb:.1f} ± {sd_ai_comb:.1f} min",
#     'Time Saving (Mean)': f"{derived_saving_comb:.1f} min",
#     'Time Saving (%)': f"{derived_pct_comb:.1f}%",
#     'Raw P': p_comb
# })
# 替换 Combined Total 的 case_results.append 如下：
case_results.append({
    'Analysis Level': 'Case Level',
    'Subgroup': 'Combined Total',
    'Manual Time': get_time_stats_str(df_case_all['Time_Manual_Min']),
    'AI-Assisted Time': get_time_stats_str(df_case_all['Time_AI_Min']),
    'Time Saving (Mean)': f"{derived_saving_comb:.1f} min",
    'Time Saving (%)': f"{derived_pct_comb:.1f}%",
    'Raw P': p_comb
})
# ==========================================
# 3. Organ Level Analysis
# ==========================================
organs = ['Lungs_GTV', 'Brachial Plexus', 'SpinalCord', 'Heart', 'Esophagus']
organ_map = {'Lungs_GTV': 'Lungs-GTV', 'SpinalCord': 'Spinal Cord'}
organ_results = []

for org in organs:
    subset_all = df_all[df_all['Organ'] == org]
    m_man_min = subset_all['Time_Manual_Min'].mean()
    sd_man_min = subset_all['Time_Manual_Min'].std()
    m_ai_min = subset_all['Time_AI_Min'].mean()
    sd_ai_min = subset_all['Time_AI_Min'].std()
    derived_saving = m_man_min - m_ai_min
    derived_pct = (derived_saving / m_man_min * 100) if m_man_min != 0 else 0
    
    subset_stats = df_stats[df_stats['Organ'] == org]
    valid_data = subset_stats[['Time_Manual_Min', 'Time_AI_Min']].dropna()
    t_man_p = valid_data['Time_Manual_Min']
    t_ai_p = valid_data['Time_AI_Min']
    diff = t_man_p - t_ai_p
    
    if len(diff) < 3: 
        p = np.nan
    elif shapiro(diff)[1] > 0.05: 
        _, p = ttest_rel(t_man_p, t_ai_p)
    else: 
        _, p = wilcoxon(t_man_p, t_ai_p)
    
#     organ_results.append({
#         'Analysis Level': 'Organ Level',
#         'Subgroup': organ_map.get(org, org),
#         'Manual Time (Mean ± SD)': f"{m_man_min:.1f} ± {sd_man_min:.1f} min",
#         'AI-Assisted Time (Mean ± SD)': f"{m_ai_min:.1f} ± {sd_ai_min:.1f} min",
#         'Time Saving (Mean)': f"{derived_saving:.1f} min",
#         'Time Saving (%)': f"{derived_pct:.1f}%",
#         'Raw P': p
#     })
# 原有的 derived_saving 计算保持不变...
    
    # 替换原有的 organ_results.append 如下：
    organ_results.append({
        'Analysis Level': 'Organ Level',
        'Subgroup': organ_map.get(org, org),
        'Manual Time': get_time_stats_str(subset_all['Time_Manual_Min']),
        'AI-Assisted Time': get_time_stats_str(subset_all['Time_AI_Min']),
        'Time Saving (Mean)': f"{derived_saving:.1f} min",
        'Time Saving (%)': f"{derived_pct:.1f}%",
        'Raw P': p
    })
# ==========================================
# 4. FDR Correction (分层校正)
# ==========================================
def apply_fdr(results_list):
    p_vals = [res['Raw P'] for res in results_list]
    # 提取非空索引
    valid_idx = [i for i, p in enumerate(p_vals) if not np.isnan(p)]
    valid_p = [p_vals[i] for i in valid_idx]
    
    if len(valid_p) > 0:
        _, p_fdr, _, _ = multipletests(valid_p, method='fdr_bh')
        p_final = [np.nan] * len(p_vals)
        for idx, val in zip(valid_idx, p_fdr):
            p_final[idx] = val
        return p_final
    return p_vals

# 分别计算 Case Level 和 Organ Level 的 FDR
case_fdr_vals = apply_fdr(case_results)
organ_fdr_vals = apply_fdr(organ_results)

# 格式化函数
def fmt_p(p):
    if np.isnan(p): return "N/A"
    if p < 0.001: return "< 0.001"
    return f"{p:.3f}"

# 整合结果
final_rows = []
for i, res in enumerate(case_results):
    res['P-value (FDR)'] = fmt_p(case_fdr_vals[i])
    del res['Raw P']
    final_rows.append(res)

for i, res in enumerate(organ_results):
    res['P-value (FDR)'] = fmt_p(organ_fdr_vals[i])
    del res['Raw P']
    final_rows.append(res)

df_final = pd.DataFrame(final_rows)

# ==========================================
# 5. Output
# ==========================================
print(df_final.to_string(index=False))
df_final.to_csv('Efficiency_Analysis_FDR_Corrected.csv', index=False, encoding='utf-8-sig')

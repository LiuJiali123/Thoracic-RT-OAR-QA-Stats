import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. Configuration
# ==========================================
# file_path = '0_All_Cases_Consolidated_Metrics (1).xlsx - All_Data_Raw (Plotting).csv'
# df = pd.read_csv(file_path)
# file_path = '0_All_Cases_Consolidated_Metrics_no08bp.xlsx'
file_path = '0_All_Cases_Consolidated_Metrics2.xlsx'
df = pd.read_excel(file_path, sheet_name='All_Data_Raw (Plotting)')

metrics_map = {
    'Lungs_GTV': {
        'dose': ['Mean Dose [Gy]'], 
        'vol': ['Vol > 1 Gy (%)', 'Vol > 5 Gy (%)', 'Vol > 20 Gy (%)']
    },
    'Heart': {
        'dose': ['Mean Dose [Gy]', 'Max Dose [Gy]'], 
        'vol': ['Vol > 30 Gy (%)', 'Vol > 40 Gy (%)']
    },
    'Esophagus': {
        'dose': ['Mean Dose [Gy]', 'Max Dose [Gy]'], 
        'vol': []
    },
    'Cord_Com': {
        'dose': ['Max Dose [Gy]'], 
        'vol': []
    },
    'Cord_Full': {
        'dose': ['Max Dose [Gy]'], 
        'vol': []
    },
    'Cord_Mis': {
        'dose': ['Max Dose [Gy]'], 
        'vol': []
    },
    'BrachialPlexus_Cancer_Lateral': {
        'dose': ['Max Dose [Gy]'], 
        'vol': []
    }
}

phases = ['Real-World', 'Manual (Post-Education)', 'Education+AI-Assisted']
comparisons = [
    ('Real-World', 'Manual (Post-Education)', 'RW vs Manual'),
    ('Manual (Post-Education)', 'Education+AI-Assisted', 'Manual vs AI'),
    ('Real-World', 'Education+AI-Assisted', 'RW vs AI')
]

# ==========================================
# 2. Data Loading & Processing & Raw Diff Export
# ==========================================
available_organs = df['Organ'].unique()
actual_organ_map = {}
for key in metrics_map:
    match = [o for o in available_organs if key in o]
    if match:
        actual_organ_map[match[0]] = metrics_map[key]

# Structure: Organ -> Case -> Metric -> Deviation Data
processed_data = []
raw_diff_export = [] # List for raw diff export

for organ, config in actual_organ_map.items():
    df_organ = df[df['Organ'] == organ]
    
    for case_id in df_organ['Case'].unique():
        df_case = df_organ[df_organ['Case'] == case_id]
        gt_row = df_case[df_case['Phase'] == 'Ground Truth'].iloc[0]
        
        all_metrics = config['dose'] + config['vol']
        participants = df_case['Hospital_Number'].unique()
        
        for pid in participants:
            # === CRITICAL: EXCLUDE REF/GT FROM EVERYTHING (BASELINE) ===
            if pid in ['Ref', 'GT']: continue

            # === CRITICAL: DL LOGIC ===
            # DL (Deep Learning) is included in Raw Export but Excluded from Stats
            is_stat_participant = (pid != 'DL')
            
            p_data = df_case[df_case['Hospital_Number'] == pid]
            
            # For Stats Processing
            row_dict = {'Organ': organ, 'Case': case_id, 'PID': f"{case_id}_{pid}"}
            
            # For Raw Export
            export_row = {'Organ': organ, 'Case': case_id, 'Hospital_Number': pid}
            
            for m in all_metrics:
                gt_val = gt_row[m]
                
                for phase in phases:
                    phase_row = p_data[p_data['Phase'] == phase]
                    
                    if not phase_row.empty:
                        val = phase_row.iloc[0][m]
                        
                        # Deviation Calculation
                        if m in config['dose']:
                            # Relative (%)
                            dev = (val - gt_val) / gt_val * 100 if gt_val != 0 else 0
                        else:
                            # Absolute (percentage points)
                            dev = val - gt_val
                        
                        row_dict[f'{m}_{phase}_Signed'] = dev
                        row_dict[f'{m}_{phase}_Abs'] = abs(dev)
                        
                        # Add to export row
                        # Column naming convention: "Metric_Phase_Diff"
                        export_row[f'{m} ({phase}) Diff'] = dev
                        
                    else:
                        row_dict[f'{m}_{phase}_Signed'] = np.nan
                        row_dict[f'{m}_{phase}_Abs'] = np.nan
                        # Leave empty in export row or set NaN?
                        # Pandas handles missing keys as NaN
            
            # === SEPARATION LOGIC ===
            # 1. Always add to Raw Export (Includes DL)
            raw_diff_export.append(export_row)
            
            # 2. Only add to Stats if Human
            if is_stat_participant:
                processed_data.append(row_dict)

df_proc = pd.DataFrame(processed_data)
df_diff_export = pd.DataFrame(raw_diff_export)

# Sort raw export columns
base_cols = ['Organ', 'Case', 'Hospital_Number']
metric_cols = sorted([c for c in df_diff_export.columns if c not in base_cols])
df_diff_export = df_diff_export[base_cols + metric_cols]

# ==========================================
# 3. Statistical Analysis Function (Shapiro -> T/Wilcoxon)
# ==========================================
def calculate_p_value(data_series1, data_series2):
    # Remove NaNs and ensure pairing
    mask = (~np.isnan(data_series1)) & (~np.isnan(data_series2))
    s1 = data_series1[mask]
    s2 = data_series2[mask]
    
    if len(s1) < 3: return np.nan
    
    diff = s1 - s2
    
    # Shapiro-Wilk test for normality of difference
    try:
        _, p_shapiro = stats.shapiro(diff)
        is_normal = p_shapiro > 0.05
    except:
        is_normal = False # Fallback

    if is_normal:
        try:
            _, p_val = stats.ttest_rel(s1, s2)
        except: p_val = np.nan
    else:
        try:
            _, p_val = stats.wilcoxon(s1, s2)
        except: p_val = np.nan
            
    return p_val

# def get_stats_str(series):
#     series = series.dropna()
#     if len(series) == 0: return "-"
#     return f"{series.mean():.2f} ± {series.std():.2f} (N={len(series)})"

# --- 修改前 ---
# def get_stats_str(series):
#     series = series.dropna()
#     if len(series) == 0: return "-"
#     return f"{series.mean():.2f} ± {series.std():.2f} (N={len(series)})"

# --- 修改后 ---
# def get_stats_str(series):
#     series = series.dropna()
#     if len(series) == 0: return "-"
#     # 剂量指标保留1位小数
#     return f"{series.mean():.1f} ± {series.std():.1f} (N={len(series)})"
# ==========================================
# 新增：非参数 Bootstrap 方法计算 95% CI
# ==========================================
def calculate_bootstrap_ci(data_series, n_iterations=1000):
    """
    使用 non-parametric bootstrap 方法计算 95% CI
    随机放回抽样 1000 次，每次样本量与原数据相同。
    """
    values = data_series.dropna().values
    n_size = len(values)
    
    if n_size == 0:
        return np.nan, np.nan

    # 向量化生成 1000 次重复抽样以加速运算
    samples = np.random.choice(values, size=(n_iterations, n_size), replace=True)
    bootstrapped_means = np.mean(samples, axis=1)
    
    # 获取 2.5% 和 97.5% 分位数作为 CI 的上下限
    lower_ci = np.percentile(bootstrapped_means, 2.5)
    upper_ci = np.percentile(bootstrapped_means, 97.5)
    
    return lower_ci, upper_ci

# --- 修改后 (包含 Range 和 95% CI) ---
def get_stats_str(series):
    series = series.dropna()
    if len(series) == 0: return "-"
    
    # 基础统计量
    mean_val = series.mean()
    sd_val = series.std()
    min_val = series.min()
    max_val = series.max()
    
    # 计算 Bootstrap 95% CI
    ci_lower, ci_upper = calculate_bootstrap_ci(series)
    
    # 格式拼接，剂量指标保留1位小数
    return (f"{mean_val:.1f} ± {sd_val:.1f} "
            f"[{min_val:.1f} - {max_val:.1f}] "
            f"(95% CI: {ci_lower:.1f} - {ci_upper:.1f}) "
            f"(N={len(series)})")
# ==========================================
# 4. Generate Stats Table per Case per Organ
# ==========================================
results_buffer = []
p_values_collection = [] # Store all raw P-values for FDR
p_values_index_map = []  # Map back to result row index and comparison type

for organ in actual_organ_map.keys():
    # Identify relevant metrics
    conf_key = None
    for k in metrics_map:
        if k in organ: 
            conf_key = k
            break
    metrics = metrics_map[conf_key]['dose'] + metrics_map[conf_key]['vol']
    
    for case_id in df_proc[df_proc['Organ'] == organ]['Case'].unique():
        subset = df_proc[(df_proc['Organ'] == organ) & (df_proc['Case'] == case_id)]
        
        for m in metrics:
            row_res = {'Organ': organ, 'Case': case_id, 'Metric': m}
            
            # 4.1 All Data Stats (Mean ± STD) using Signed Deviation
            for phase in phases:
                col_signed = f'{m}_{phase}_Signed'
                stats_str = get_stats_str(subset[col_signed])
                row_res[f'{phase} (All)'] = stats_str
            
            # 4.2 Paired Comparisons
            for p1, p2, label in comparisons:
                col1_abs = f'{m}_{p1}_Abs'
                col2_abs = f'{m}_{p2}_Abs'
                col1_signed = f'{m}_{p1}_Signed'
                col2_signed = f'{m}_{p2}_Signed'
                
                # Extract paired subset for stats display
                mask = subset[col1_abs].notna() & subset[col2_abs].notna()
                paired_subset = subset[mask]
                
                # Store Paired Mean ± STD
                row_res[f'{p1} (Paired vs {p2})'] = get_stats_str(paired_subset[col1_signed])
                row_res[f'{p2} (Paired vs {p1})'] = get_stats_str(paired_subset[col2_signed])
                
                # Calculate Raw P-value (comparing magnitude of error: Abs Deviation)
                raw_p = calculate_p_value(subset[col1_abs], subset[col2_abs])
                row_res[f'{label} Raw_P'] = raw_p
                
                # Collect for FDR
                if not np.isnan(raw_p):
                    p_values_collection.append(raw_p)
                    p_values_index_map.append((len(results_buffer), label))
            
            results_buffer.append(row_res)

# ==========================================
# 5. FDR Correction (Benjamini-Hochberg)
# ==========================================
if p_values_collection:
    reject, pvals_corrected, _, _ = multipletests(p_values_collection, alpha=0.05, method='fdr_bh')
else:
    pvals_corrected = []

# Fill back corrected P-values
for row in results_buffer:
    for _, _, label in comparisons:
        row[f'{label} P-value'] = "N<3" # Initialize

for idx, (row_idx, label) in enumerate(p_values_index_map):
    corrected_p = pvals_corrected[idx]
    
    # Format Sig
    sig = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else ""
    p_str = f"{corrected_p:.3f}{sig}"
    
    results_buffer[row_idx][f'{label} P-value'] = p_str

# ==========================================
# 6. Save Output (Both Stats and Raw Diff)
# ==========================================
df_final_dose = pd.DataFrame(results_buffer)

# Reorder columns for clarity
cols_order = ['Organ', 'Case', 'Metric']
for phase in phases:
    cols_order.append(f'{phase} (All)')

for p1, p2, label in comparisons:
    cols_order.append(f'{label} P-value')
    cols_order.append(f'{p1} (Paired vs {p2})')
    cols_order.append(f'{p2} (Paired vs {p1})')

final_cols = [c for c in cols_order if c in df_final_dose.columns]
df_final_dose = df_final_dose[final_cols]

# output_filename_stats = '2_Table_Dosimetric_Diff_Strict_Stats_FDR_By_Case_no08bp.csv'
output_filename_stats = '2_Table_Dosimetric_Diff_Strict_Stats_FDR_By_Case2.csv'
df_final_dose.to_csv(output_filename_stats, index=False, encoding='utf-8-sig')

# output_filename_raw = '0_All_Cases_Dosimetric_Differences_Raw_no08bp.csv'
output_filename_raw = '0_All_Cases_Dosimetric_Differences_Raw2.csv'
df_diff_export.to_csv(output_filename_raw, index=False, encoding='utf-8-sig')

print(f"✅ 剂量学严格统计表 (By Case, FDR Corrected) 已生成: {output_filename_stats}")
print(f"✅ 剂量学原始差值表 (Raw Differences) 已生成: {output_filename_raw}")
print("\nStats Table Preview (No DL):")
print(df_final_dose[['Organ', 'Case', 'Metric', 'RW vs Manual P-value', 'Manual vs AI P-value']].head().to_string())
print("\nRaw Diff Table Preview (Check for DL):")
dl_check = df_diff_export[df_diff_export['Hospital_Number'] == 'DL']
if not dl_check.empty:
    print(dl_check.head().iloc[:, :5].to_string())
else:
    print("DL not found in Raw Diff (Check logic?)")


# CODE 2 

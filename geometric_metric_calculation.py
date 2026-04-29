import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import re

# ==========================================
# 1. 核心计算函数
# ==========================================

def dice_score(gt_arr, seg_arr):
    intersection = np.sum((gt_arr > 0) & (seg_arr > 0))
    size_sum = np.sum(gt_arr > 0) + np.sum(seg_arr > 0)
    return 2.0 * intersection / size_sum if size_sum > 0 else 1.0

def calculate_surface_metrics(gt_img, pred_img):
    try:
        gt = sitk.Cast(gt_img != 0, sitk.sitkUInt8)
        pred = sitk.Cast(pred_img != 0, sitk.sitkUInt8)
        if sitk.GetArrayViewFromImage(gt).sum() == 0 or sitk.GetArrayViewFromImage(pred).sum() == 0:
            return np.nan, np.nan
        dist_map_gt = sitk.Abs(sitk.SignedMaurerDistanceMap(gt, squaredDistance=False, useImageSpacing=True))
        dist_map_pred = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True))
        cf = sitk.BinaryContourImageFilter()
        dist_p_to_g = sitk.GetArrayFromImage(dist_map_gt)[sitk.GetArrayFromImage(cf.Execute(pred)) == 1]
        dist_g_to_p = sitk.GetArrayFromImage(dist_map_pred)[sitk.GetArrayFromImage(cf.Execute(gt)) == 1]
        all_dist = np.concatenate([dist_p_to_g, dist_g_to_p])
        return np.percentile(all_dist, 95), np.mean(all_dist)
    except:
        return np.nan, np.nan

# ==========================================
# 2. 改进的文件匹配逻辑
# ==========================================

def parse_file_info(file_name, organ_list):
    fname = file_name.replace(".nii.gz", "")
    
    # 按长度降序排列，确保优先匹配长字符串（如 Cord_Full 优先于 Cord）
    sorted_organs = sorted(organ_list, key=len, reverse=True)
    
    matched_organ = None
    for o in sorted_organs:
        if o in fname:
            matched_organ = o
            break
            
    if not matched_organ:
        return None, None

    # 提取 ID: 针对 RW01_... 或 AI_Heart_01...
    # 先找前缀数字 (如 RW01)
    prefix_match = re.search(r'^[A-Z]{2}(\d+)', fname)
    if prefix_match:
        observer_id = prefix_match.group(1)
    else:
        # 再找末尾数字 (如 _01)
        ids = re.findall(r'\d+', fname)
        observer_id = ids[-1] if ids else "DL"
    
    return matched_organ, observer_id

# ==========================================
# 3. 执行处理
# ==========================================

def process_case_comprehensive(case_id, base_dir, organ_list, target_folders):
    path_gt = os.path.join(base_dir, "GT")
    results = []

    # 自动识别并缓存 GT
    gt_cache = {}
    print(f"--- 正在加载 {case_id} 的 GT 文件 ---")
    for organ in organ_list:
        # 兼容多种命名格式: GT_Cord_Full.nii.gz 或 GT_Lungs_GTV.nii.gz
        possible_names = [f"GT_{organ}.nii.gz", f"GT_{organ.replace('Lungs_GTV','GTV')}.nii.gz"]
        for p_name in possible_names:
            p_path = os.path.join(path_gt, p_name)
            if os.path.exists(p_path):
                img = sitk.ReadImage(p_path)
                gt_cache[organ] = (img, sitk.GetArrayFromImage(img))
                print(f"  [Found GT] {organ}")
                break

    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path): continue
        
        print(f"\n处理目录: {folder}")
        files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
        
        for f_name in files:
            matched_organ, obs_id = parse_file_info(f_name, organ_list)
            
            if matched_organ and matched_organ in gt_cache:
                gt_img, gt_arr = gt_cache[matched_organ]
                try:
                    seg_img = sitk.ReadImage(os.path.join(folder_path, f_name))
                    seg_arr = sitk.GetArrayFromImage(seg_img)
                    
                    dsc = dice_score(gt_arr, seg_arr)
                    hd95, asd = calculate_surface_metrics(gt_img, seg_img)
                    
                    results.append({
                        "Case": case_id,
                        "Folder": folder,
                        "Organ": matched_organ,
                        "ID": obs_id,
                        "DSC": dsc, "HD95": hd95, "ASD": asd,
                        "FileName": f_name
                    })
                except:
                    pass
    return results
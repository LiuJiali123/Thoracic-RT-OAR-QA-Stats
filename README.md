# Statistical Analysis Pipeline for Thoracic RT Protocol Deviations

This repository contains the custom Python scripts used for the statistical analysis in our manuscript.

## Scripts Included

### 1. Metric Extraction
- **`geometric_metric_calculation.py`**: Extracts geometric indices (**DSC, HD95, and ASD**) from raw NIfTI (.nii.gz) files. It implements symmetric surface distance calculations and is fully voxel-spacing aware using SimpleITK.

### 2. Inferential Statistics
- **`geometric_fdr_p_value.py`**: Performs pairwise geometric comparisons across multi-center cohorts using pooled analysis and FDR correction.
- **`dosimetric_stats.py`**: Performs independent dosimetric evaluation (Mean/Max dose, Volume metrics) for each clinical scenario.
- **`efficiency_stats.py`**: Evaluates active contouring duration using a hybrid approach (independent case-level and pooled organ-level assessments).

## Nomenclature Mapping (Data Dictionary)
Please note that the internal folder names and variables in the scripts correspond to the finalized clinical phases in the manuscript as follows:
- **`RW` (or 'Real-World')** in code $\rightarrow$ **`Real-World Manual`** in manuscript.
- **`TR` (or 'Manual Post-Education')** in code $\rightarrow$ **`Post-Education Manual`** in manuscript.
- **`AI` (or 'Education+AI-Assisted')** in code $\rightarrow$ **`Post-Education AI-Assisted`** in manuscript.

*This separation between internal data-handling variables and final publication terminology was strictly maintained to ensure original data integrity and prevent file-path corruption.*

## Statistical Methodology

The pipeline ensures high reproducibility by executing:
1. **Normality Testing**: Shapiro-Wilk test on paired differences to guide test selection.
2. **Adaptive Testing**: Automatically selects Paired t-test (parametric) or Wilcoxon Signed-Rank test (non-parametric) based on distribution.
3. **Multiple Testing Correction**: Benjamini-Hochberg False Discovery Rate (FDR) adjustment applied to all inferential P values.

## Environment Requirements
The analyses were conducted in the following environment:
- Python == 3.7.4
- pandas == 0.25.1
- scipy == 1.3.1
- statsmodels == 0.10.1
- numpy == 1.19.5

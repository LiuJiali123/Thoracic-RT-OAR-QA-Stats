# Statistical Analysis Pipeline for Thoracic RT Protocol Deviations

This repository contains the custom Python scripts used for the statistical analysis in our manuscript.

## Scripts Included
- `geometric_fdr_p_value.py`: Performs pairwise geometric comparisons (DSC, HD95, ASD) across multi-center cohorts.
- **`dosimetric_stats.py`**: Performs independent dosimetric evaluation (Mean/Max dose, Volume metrics) for each clinical scenario.
- **`efficiency_stats.py`**: Evaluates active contouring duration using a hybrid approach (independent case-level and pooled organ-level assessments).

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

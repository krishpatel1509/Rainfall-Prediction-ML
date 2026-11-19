# Exploratory Data Analysis — cleaned weatherAUS

- Rows and columns: **142,193 × 23**
- Observation period: **2007-11-01 to 2017-06-25**
- RainTomorrow=Yes: **31,877 (22.42%)**
- Predictors: **16 numeric**, **5 categorical** (Date excluded from model predictors).

## Findings

- The label is imbalanced: the positive class is 22.42%; accuracy should be interpreted together with recall, F1, ROC-AUC, and average precision.
- Missing feature values are intentionally retained for train-only imputation. Highest missingness: Sunshine (47.7%), Evaporation (42.8%), Cloud3pm (40.2%), Cloud9am (37.7%), Pressure9am (9.9%).
- Tukey 1.5×IQR outlier flags are descriptive; no observations were removed or clipped based on the fences. Highest rates: Rainfall (17.9%), Evaporation (2.4%), WindGustSpeed (2.3%), WindSpeed3pm (1.8%), WindSpeed9am (1.2%).
- Rainfall is strongly right-skewed and has the largest Tukey outlier rate (17.9% of its nonmissing values); these flagged values are retained as potentially genuine extremes.
- The strongest pairwise numeric association is MaxTemp–Temp3pm (r=0.985); Pressure9am–Pressure3pm (r=0.961); MinTemp–Temp9am (r=0.902). Correlation is descriptive, not causal.
- Monthly RainTomorrow rates range from 19.3% (month 1) to 26.9% (month 7); this pooled seasonal pattern may reflect location and year composition.
- Among rows with `RainToday=No`, the next-day positive rate is 15.2%; after `RainToday=Yes`, it is 46.4%. This is an association, not a causal estimate.

## Figures and tables

See the adjacent PNG and CSV files: target distribution, missingness, feature distributions, numeric correlation heatmap, Tukey outlier rates, monthly positive-label rate, and RainToday/RainTomorrow cross-tab.

## Interpretation cautions

Outlier fences are univariate and may flag valid meteorological extremes. Pairwise Pearson correlations capture only linear associations and may not reveal nonlinear or conditional relationships.

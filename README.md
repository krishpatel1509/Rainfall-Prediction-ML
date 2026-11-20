# Next-Day Rainfall Classification (Australia)

A reproducible study of the course-supplied `weatherAUS.csv` dataset. The workflow keeps the original CSV unchanged, creates a cleaned derivative, performs EDA, tunes seven classifiers with five chronological CV folds, and reports results on a later-period holdout.

## Project layout

```text
rainfall-ml-paper/
├── raw_data/weatherAUS.csv                  # supplied raw file; do not edit
├── cleaned_data/weatherAUS_clean.csv         # output of the independent cleaner
├── cleaned_data/cleaning_log.json
├── src/clean_data.py                         # cleaning only
├── src/eda.py                                # EDA on cleaned data
├── src/train.py                              # 7-model tuning and evaluation
├── src/ablation.py                           # one-component-removed ablation of the selected CatBoost pipeline
├── outputs/eda/                              # EDA plots, CSV summaries, observations
├── outputs/modeling/                         # CV trials, comparison, final evaluation
├── outputs/ablation/                         # ablation table and figure
├── paper/manuscript.md                       # full paper source
├── paper/Rainfall_Prediction_Research_Paper.docx
├── paper/Rainfall_Prediction_Research_Paper.pdf
├── Rainfall_Prediction_Presentation.pptx
└── docs/                                     # workflow diagram, dataset notes, submission guidance
```

## Reproduce

Use Python 3.10+ (tested with Python 3.12.3). From the project root:

```bash
python -m pip install -r requirements.txt
python src/clean_data.py --input raw_data/weatherAUS.csv --output-dir cleaned_data
python src/eda.py --input cleaned_data/weatherAUS_clean.csv --output-dir outputs/eda
python src/train.py --data cleaned_data/weatherAUS_clean.csv --output-dir outputs/modeling --test-fraction 0.20 --cv-rows 8000 --knn-fit-rows 20000
python src/ablation.py --data cleaned_data/weatherAUS_clean.csv --output-dir outputs/ablation
```

Cleaning trims text whitespace, parses dates, removes records with missing/invalid dates or target labels, deduplicates exact remaining rows, and sorts observations. Feature missingness and statistical extremes are retained; missing-value imputation is fitted inside each training fold/model pipeline to avoid leakage. The cleaner records SHA-256 hashes to demonstrate that the original raw file was not modified.

EDA writes label counts, missingness tables, numeric distributions, Pearson correlations, Tukey 1.5×IQR outlier analysis, monthly rain-label rates, and the `RainToday` × `RainTomorrow` cross-tab. Outlier flags are descriptive only; no observations are clipped or removed based on them.

The model script compares KNN, CatBoost, Random Forest, Decision Tree, Linear SVM, Logistic Regression, and Histogram Gradient Boosting. Each model has five predeclared hyperparameter configurations. `GridSearchCV` uses five expanding-window chronological folds, grouped by complete calendar dates, on a deterministic subset (8,000 rows by default) of the training era, optimizing average precision. The chosen configuration per family is refit on all training-era rows; KNN uses a date-spanning sample capped at 20,000 rows because exact neighbor prediction cost grows rapidly with training size. The per-model refit row count is recorded in `model_comparison.csv`. A later chronological 20% of distinct dates is held out until final evaluation. The best family is selected using training-period CV average precision, not the holdout.

## Main results and artifacts

- Selected model: **CatBoost** (`depth=5`, 220 iterations, learning rate 0.07, L2 7) with training-era CV average precision 0.6837.
- Later-period holdout (32,891 rows, 21.92% positive): accuracy 0.8041, precision 0.5376, recall 0.7622, F1 0.6305, ROC-AUC 0.8735, average precision 0.7068.
- Metric leaders differ: Random Forest has the highest accuracy (0.8417) and F1 (0.6327); K-Nearest Neighbors has the highest precision (0.7586, with the lowest recall, 0.3224); Decision Tree has the highest recall (0.7679).
- Ablation (`src/ablation.py`, `outputs/ablation/`): removing 3 pm humidity lowers holdout AP by 0.054; removing class weighting leaves AP essentially unchanged (+0.0019) but moves precision/recall to 0.7552/0.4847.
- `outputs/modeling/hyperparameter_search_results.csv` — all 35 configurations, fold-mean average precision, variation, and timing.
- `outputs/modeling/model_comparison.csv` — best configuration and holdout metrics per model.
- `outputs/modeling/final_classification_report.csv` and `final_confusion_matrix.csv` — selected-model report and counts.
- `outputs/modeling/final_roc_curve.csv` and `final_evaluation.png` — ROC data and plot.
- `outputs/modeling/best_model.joblib` — fitted CatBoost pipeline for the selected configuration.
- `outputs/eda/EDA_findings.md` — concise findings; the full manuscript integrates the results and plots.

Results depend on the input snapshot, package versions, and compute settings; consult the recorded run metadata. This project is an empirical course study, not an operational forecast service. The dataset package did not include explicit redistribution terms, so confirm permission before public hosting. The manuscript follows the attached two-column IEEE-style reference and lists Krish Patel, Dharak Patel, Het Viradiya and Nisharg Trivedi with the supplied Karnavati University affiliation; only the email line remains blank.

## Paper and presentation

The paper (four authors, nine pages, two-column IEEE-style) is in `paper/` as Markdown (`manuscript.md`), editable Word, and PDF, with the Markdown figures in `paper/media/`. It documents cleaning, EDA, the seven-model benchmark, tuning experiments, model comparison, selected-model evaluation, limitations, and an appendix listing all 35 tested configurations. The eight-slide editable PowerPoint is at the project root. See `docs/SUBMISSION_CHECKLIST.md` for citation, email, and reproducibility checks before submission.
"# Rainfall-Prediction-ML" 

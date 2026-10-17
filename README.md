# Predicting Diabetes Risk from Health-Survey Data: Traditional Machine Learning vs. Deep Learning

Can a short health questionnaire identify who is likely to have diabetes or prediabetes, and do deep neural networks beat gradient-boosted trees on this kind of tabular data? This project answers both questions with **14 controlled experiments** on the CDC BRFSS 2015 survey (229,474 respondents after cleaning), comparing four traditional models with four Keras architectures.



## Key findings

* **Traditional ML and deep learning tie.** The best traditional model (shallow XGBoost, Exp 6) reaches test ROC-AUC **0.8217**; the best deep model (residual network with L2, Exp 14) reaches **0.8223**. A paired bootstrap gives a difference of -0.0006 with a 95% CI of [-0.0020, +0.0005], which includes zero.
* **All 14 models lie within 0.01 ROC-AUC** (0.8125 to 0.8223). Performance is limited by the information in the survey features, not by model capacity.
* **Gradient boosting is the pragmatic choice:** the same accuracy in about 1 second of training versus 136 to 172 seconds for the Keras models (CPU).
* **Class weights change the operating point, not the ranking.** Recall at threshold 0.5 rose from 0.152 to 0.764, but ROC-AUC was unchanged.
* **Feature engineering helps linear models but not trees:** +0.0108 validation PR-AUC for logistic regression, -0.0018 for XGBoost.
* **Hyper-parameter search and a residual connection gave no measurable gain.**
* **Error patterns:** at the F1-optimal threshold the best models miss about 40% of cases. Missed cases look healthier on the strongest predictors. Error rates differ strongly by age, income and sex, mostly because prevalence differs.

## Dataset

| Property | Value |
|:--|:--|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (ID 891) |
| Raw size | 253,680 rows, 21 features + binary target `Diabetes_binary` |
| Feature types | 14 binary, 6 ordinal/count, 1 continuous (BMI) |
| Class balance | About 13.9% positive (6.18 : 1) in the raw data |
| After cleaning | 229,474 rows (24,206 exact duplicates removed), 15.3% positive |
| Split | Stratified 70 / 15 / 15: train 160,631, validation 34,421, test 34,422 |

## Method overview

**Preprocessing and leakage control.** Exact duplicates are removed before splitting. BMI is clipped to the training 0.5th/99.5th percentiles. The scaler, clipping bounds and class weight are learned from the training set only. All tuning, early stopping, threshold selection and model selection use the **validation** set; the test set is used once for reporting.

**Feature engineering.** Nine features (BMI tier, metabolic risk, BP x cholesterol interaction, cardiovascular burden, lifestyle score, physical limitation, age x BMI, healthcare access, socioeconomic status), taking 21 features to 30. Their effect is measured by ablation, not assumed.

**Class imbalance.** Cost-sensitive weighting (positive weight 5.538); no resampling.

**Evaluation.** ROC-AUC and PR-AUC (threshold-free); precision, recall, F1 and confusion matrices at the threshold that maximises F1 on validation; 300-resample bootstrap confidence intervals; learning curves; bias-variance diagnostics; permutation importance; subgroup error analysis.

### The 14 experiments

Each experiment changes one thing relative to a named parent, so differences can be attributed to a single decision.

| Exp | Model | Change vs. parent |
|:--:|:--|:--|
| 1 | Logistic regression | Baseline, no class weights |
| 2 | Logistic regression | + class weights |
| 3 | Logistic regression | + engineered features |
| 4 | Random forest (depth 8) | Non-linear ensemble (bagging) |
| 5 | Random forest (depth 14, leaf >= 5) | More capacity |
| 6 | XGBoost (shallow) | Boosting, raw features |
| 7 | XGBoost (shallow) | + engineered features (ablation) |
| 8 | XGBoost | Randomized search (3-fold CV, scoring PR-AUC) |
| 9 | LightGBM | Randomized search |
| 10 | XGBoost | Top-15 features only (feature selection) |
| 11 | Keras Sequential (128-64-32) | Deep-learning baseline |
| 12 | Keras Sequential | + batch normalization |
| 13 | Keras Functional | Residual skip connection |
| 14 | Keras Functional | + L2 weight decay, lower learning rate |

The Keras models use a `tf.data` pipeline (shuffle, batch, prefetch), Adam, weighted binary cross-entropy, early stopping on validation loss and learning-rate reduction on plateau.

## Results

Test-set results (precision, recall and F1 at the validation-tuned threshold). The full 14-row table, with hyper-parameters and per-run observations, is in the notebook and the report.

| Exp | Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | Logistic regression (baseline) | 0.8125 | 0.4200 | 0.4689 | 0.622 | 0.376 |
| 3 | Logistic regression + weights + FE | 0.8154 | 0.4332 | 0.4689 | 0.648 | 0.367 |
| 5 | Random forest (depth 14) | 0.8167 | 0.4376 | 0.4723 | 0.646 | 0.372 |
| 6 | **XGBoost shallow (best traditional)** | 0.8217 | 0.4535 | 0.4809 | 0.599 | 0.402 |
| 8 | XGBoost tuned | 0.8220 | 0.4535 | 0.4776 | 0.601 | 0.396 |
| 9 | LightGBM tuned | 0.8214 | 0.4522 | 0.4742 | 0.566 | 0.408 |
| 11 | Keras Sequential | 0.8209 | 0.4503 | 0.4777 | 0.600 | 0.397 |
| 14 | **Keras residual + L2 (best deep)** | 0.8223 | 0.4546 | 0.4768 | 0.563 | 0.414 |

Best models were selected on **validation** PR-AUC. Differences in the third decimal are within bootstrap noise (a single model's 95% CI is roughly +/-0.005).

## Repository structure

```
.
├── diabetes_risk_ml_dl_colab.ipynb   # Full pipeline: data, preprocessing, 14 experiments, analysis
├── requirements.txt                  # Python dependencies
├── README.md
├── LICENSE
├── .gitignore                        # Excludes raw data, models, logs
├── docs/
│   └── figures/                      # Figures used in this README and the report
├── data/                             # Created at runtime (NOT committed)
├── models/                           # Created at runtime (best ML + best DL weights)
├── results/                          # Created at runtime (experiment table, figures)
└── logs/tensorboard/                 # Created at runtime (one run per experiment)
```

## Getting started

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top.
2. Run **Runtime > Run all**. The notebook installs missing packages, downloads the dataset and runs everything. The Keras models run fine on CPU; a GPU is optional.
3. To copy the dataset zip, model weights and results to Google Drive, set `SAVE_TO_DRIVE = True` in the setup cell before running.

### Option 2: Run locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

jupyter lab diabetes_risk_ml_dl_colab.ipynb
```

Then run all cells top to bottom (roughly 20 to 40 minutes on a laptop CPU).

### Inspecting the experiment logs

Every experiment has a TensorBoard record of its hyper-parameters, rationale and final metrics (Keras runs also log per-epoch curves):

```bash
tensorboard --logdir logs/tensorboard
```

## Data

The dataset is **not committed** to this repository, as required by the assignment. The notebook downloads it with the `ucimlrepo` package (`fetch_ucirepo(id=891)`), saves it to `data/`, and writes a zip file for the Google Drive folder. Check the [UCI dataset page](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) for the licence terms.

## Reproducibility

* One seed (`SEED = 42`) controls Python, NumPy and TensorFlow randomness.
* Library versions are printed at the top of the notebook.
* Preprocessing statistics come from the training set only; the test set is used once.
* `QUICK_RUN = True` in the setup cell runs a tiny smoke test; keep it `False` for reportable results. GPU runs can differ slightly in the last decimals.

## Limitations

* Self-reported telephone survey from a single year in the United States; results may not transfer to other populations or clinical data.
* The label merges prediabetes and diabetes.
* Coarse features (13 age bands; no laboratory values, family history or waist circumference) cap achievable accuracy.
* Exact duplicates were removed to prevent leakage, which may also have removed some genuine respondents.
* Two Keras models used the full 40 epochs and may be slightly under-trained; a single seed and a single split were used.

## References

1. International Diabetes Federation, *IDF Diabetes Atlas*, 11th ed., 2025. https://diabetesatlas.org
2. Z. Xie, O. Nikolayeva, J. Luo, and D. Li, "Building risk prediction models for type 2 diabetes using machine learning techniques," *Prev. Chronic Dis.*, vol. 16, p. E130, 2019.
3. L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Proc. NeurIPS*, 2022.
4. R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84-90, 2022.
5. CDC, "CDC Diabetes Health Indicators," UCI Machine Learning Repository, dataset ID 891.

The full reference list is in the written report.



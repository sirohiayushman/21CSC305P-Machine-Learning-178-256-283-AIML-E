# CardioScope — local BRFSS 2022 research screener

This is a **working localhost FastAPI + responsive HTML/CSS/JavaScript frontend** created from the uploaded Jupyter notebook `ml_ann_merged_pipeline_gpu_optuna (2)(2).ipynb`. No cloud deployment, credentials, database, or third-party inference API is required.

## Important: understand the label

`HadHeartAttack` in the 2022 BRFSS survey means **a respondent reports having experienced a heart attack in the past**. The model does **not** predict imminent/future heart attacks, does not diagnose coronary artery disease, and cannot declare anyone safe. It is an *educational research prototype*, not a medical device. Real-world clinical use would require independent external validation, representative data, prospective outcome labels, calibration checks and review by qualified clinicians.

## 1. Install dependencies (Windows PowerShell)

Python 3.10+ recommended (3.11 or 3.12 preferred). Open a terminal inside the extracted `heart_screening_local` folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Or on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**The uploaded notebook contains no raw CSV and no trained model artifact**; a notebook `.ipynb` with printed outputs is not by itself a deployable trained model. Prepare the model using **either Option A (best if you still have the notebook's saved output files)** or **Option B (retrain and evaluate with MI/Pearson/VIF)**.

## 2A. Import the exact notebook CatBoost (no retraining)

Use this if the notebook's `out_dir` folder (as generated in the notebook) is still accessible:

```powershell
python import_notebook.py --folder "C:\Users\Ayushman\Downloads\out_dir"
```

The folder must contain:

- `model_catboost.joblib`
- `survey_preprocessor.joblib`
- `train.csv` (the **preprocessed training split from that exact notebook execution**)
- `probability_calibrators.joblib`
- `validation_selected_thresholds.json`

**Why `train.csv` is essential:** notebook advanced-model cell uses `StandardScaler().fit_transform(X_train)` but does **not save the scaler**. This importer rebuilds the same scaler using the original preprocessed `train.csv`, thereby matching the fitted model's feature order and scaling. DO NOT fit the scaler on a new patient, test set or a new training split.

This path uses the original CatBoost model with **all engineered features**. The notebook applied MI → Pearson → VIF only to custom MoM models, **not** CatBoost; we do not pretend otherwise. This import path has F1 and cost-weighted operating modes. To run MI/Pearson/VIF on the deployable CatBoost, use Option B instead.

> SECURITY: Only import `.joblib` files produced by you in a trusted local environment. Loading an untrusted pickle/joblib file can execute arbitrary code.

## 2B. Retrain and evaluate the deployable CatBoost from raw CSV

Use the *raw* `heart_2022_with_nans.csv` used by the notebook, not the transformed `train.csv`. For example:

```powershell
python train.py --data "C:\path\to\heart_2022_with_nans.csv"
```

The training process will:

1. Split raw labeled records into stratified 70% training, 15% validation, 15% untouched test **before fitting any preprocessing**.
2. Learn numerical imputation, missing flags, binary/ordinal/nominal encoding and target-smoothed State encoding from training only. State encodings for training rows are cross-fitted to avoid target leakage.
3. Reproduce all notebook feature engineering: `log1p` for skewed numerics, comorbidity/functional aggregates, health-day aggregates, age × BMI, age × smoker, BMI × diabetes, chest scan × poor health and age/BMI indicators.
4. Run training-only feature selection: stable mutual information → Pearson correlation redundancy filtering → iterative stabilized VIF (threshold 20). **This route evaluates CatBoost with both all features and selected features.**
5. Train CatBoost CPU variants using the notebook's tuned hyperparameters (193 fixed trees to approximate its early-stopped ~193 best iteration, without leaking validation rows into early stopping). No new Optuna optimization is performed.
6. Partition validation into separate **calibration (40%)**, **candidate selection (30%)** and **threshold selection (30%)** splits. Fit Platt calibration on its own split, choose decision thresholds on another and compare candidates on a third independent validation split.
7. Choose the final variant by minimizing false-negative-cost ×5 plus false positives ×1 on the candidate-selection split. Run the untouched test **once**, then save the entire deployable bundle (`model + scaler + preprocessor + selector columns + calibrator + thresholds`).

The final two files are:

- `models/screening_bundle.joblib` — all inference objects, saved automatically.
- `models/training_report.json` — validation comparison, selected features and actual untouched-test metrics.

On a smaller CPU or for a quick functional check, lower work using:

```powershell
python train.py --data "C:\path\to\heart_2022_with_nans.csv" --iterations 80 --mi-repeats 1 --mi-sample 30000 --vif-sample 8000
```

This **changes the model** and its results. Do not assume it matches the notebook benchmark.

Optional `--skip-selection` trains only the original full-feature CatBoost variant.

## 3. Launch on localhost

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

You can also double-click `start_windows.bat` after Python dependencies are installed. Browse to:

**http://127.0.0.1:8000**

API reference: http://127.0.0.1:8000/docs

This listens on `127.0.0.1`, not `0.0.0.0`. There is no public deployment or external network server. The frontend uses local/system fonts only; it does not require any external font or CDN request.

## API

`GET /api/health`: returns server status and whether a local model file exists.

`GET /api/schema`: questionnaire sections, learned category options, model metadata and reference-only notebook metrics.

`POST /api/predict`: accepts original raw BRFSS answers (do **not** send engineered features).

```json
{
  "mode": "cautious",
  "answers": {
    "AgeCategory": "Age 65 to 69",
    "Sex": "Male",
    "GeneralHealth": "Good",
    "BMI": 28.4,
    "HadAngina": "No",
    "HadStroke": "No",
    "SmokerStatus": "Former smoker",
    "PhysicalActivities": "Yes"
  }
}
```

Response fields include `screening_score`, `flagged_for_follow_up`, selected `threshold`, missing input counts and explicit clinical disclaimers. These are example *inputs*, **not** an example diagnosis. Do not store or transmit real patients' identifying information using this prototype.

### Operating modes

- **`cautious` (default):** validation-selected threshold minimizes `5 × false negatives + 1 × false positives`. Matches the notebook's cost-sensitive motivation.
- **`balanced`:** validation-selected F1 threshold.
- **`high_recall`** (available after retraining): attempts ≥80% recall on the threshold-selection validation split; may create substantially more false positives and does not guarantee this recall out of sample.

**Do not use the default threshold `0.5`** for the user’s stated objective: according to the original notebook it recovered only ~24.56% of self-reported positive cases. A threshold never guarantees zero false negatives. Any apparent reassuring output must not override symptoms or medical judgment.

### Notebook comparison (reference, NOT new local results)

| Notebook CatBoost held-out test operating point | Accuracy | Recall | Precision | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Calibrated 0.50 | 94.73% | 24.56% | 58.51% | 98.95% | 34.60% |
| Validation-selected F1 threshold 0.2599 | 93.77% | 51.38% | 45.70% | 96.32% | 48.38% |
| Validation-selected FN×5 threshold 0.1219 | **91.08%** | **62.43%** | 34.31% | 92.80% | 44.28% |

Another notebook model — the validation-weighted calibrated blend — has the best F1 at ~48.51%, but requires many separately stored learners and calibrators; CatBoost is the stronger **single-model recall-oriented** choice at the cost-weighted threshold, and it is substantially easier to package correctly.

**One caution:** the original notebook's later *confusion-matrix visualization* cell appears to use different prediction files/thresholds from its dedicated honest-evaluation table. We use the explicitly labeled held-out cost-weighted results from cell 06 as the benchmark, not the later charts.

## Project structure

```
heart_screening_local/
├── main.py                 # FastAPI endpoints and input validation
├── train.py                # Re-training / evaluating full and selected-feature CatBoost
├── preprocessing.py        # Original notebook leakage-safe feature engineering
├── feature_selection.py    # Original notebook train-only MI, Pearson and VIF
├── import_notebook.py      # Rebuilds missing scaler, loads trained notebook artifacts
├── static/
│   ├── index.html          # Frontend
│   ├── style.css
│   └── app.js
├── models/                  # Model and report generated locally
├── tests/                   # Integration tests
├── requirements.txt
├── start_windows.bat
└── README.md
```

### Reproducibility and limitations

No trained model was included in the uploaded notebook attachment. Training and notebook import therefore cannot be performed here without the *separately stored raw CSV or model/output files*. Until one option is run, the app displays **"Model unavailable"** and the predict button remains disabled rather than generating fabricated predictions. The inference endpoint returns HTTP 503 when no model is present.

The notebook’s reported performance may not reproduce exactly when retraining: the original CatBoost used an Optuna search and early stopping on validation; this local retraining script reuses its tuned parameters but avoids reusing validation examples for early stopping. The 5:1 cost ratio is a demonstration choice, not a clinically validated decision threshold. Cross-sectional survey labels, selection effects, incomplete answers, demographic fairness concerns and prevalence shift limit external validity.

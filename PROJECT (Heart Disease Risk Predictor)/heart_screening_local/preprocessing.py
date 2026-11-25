# Extracted from the user-supplied notebook, cell 01.
# Keeps its exact training-fitted transforms, missing flags,
# one-hot encoding, State OOF encoding and feature engineering.
"""
01_leakage_safe_preprocessing.py
================================
Split raw rows first, then learn every preprocessing statistic from TRAIN only.

v2: adds engineered features directly into the shared transform, so every
downstream model (baselines, advanced boosters, custom MoM, deep learning)
automatically receives them without a separate opt-in step:
  - log1p transforms of right-skewed numerics (BMI, WeightInKilograms,
    PhysicalHealthDays, MentalHealthDays)
  - a severity-weighted comorbidity score (replaces the flat unweighted count)
  - clinically-motivated interaction terms (age x smoker, BMI x diabetes,
    chest-scan x poor general health)
  - clinically-binned age/BMI flags, in addition to the raw values
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET = "HadHeartAttack"
SEED = 42

BINARY_YESNO_COLS = [
    "PhysicalActivities", "HadAngina", "HadStroke", "HadAsthma",
    "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
    "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
    "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
    "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting",
    "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear",
]

ORDINAL_MAPS = {
    "GeneralHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
    "LastCheckupTime": {
        "5 or more years ago": 0,
        "Within past 5 years (2 years but less than 5 years ago)": 1,
        "Within past 2 years (1 year but less than 2 years ago)": 2,
        "Within past year (anytime less than 12 months ago)": 3,
    },
    "RemovedTeeth": {"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2, "All": 3},
    "AgeCategory": {
        "Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2,
        "Age 35 to 39": 3, "Age 40 to 44": 4, "Age 45 to 49": 5,
        "Age 50 to 54": 6, "Age 55 to 59": 7, "Age 60 to 64": 8,
        "Age 65 to 69": 9, "Age 70 to 74": 10, "Age 75 to 79": 11,
        "Age 80 or older": 12,
    },
}

NOMINAL_COLS = [
    "Sex", "HadDiabetes", "SmokerStatus", "ECigaretteUsage",
    "RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos",
]
NUMERIC_COLS = [
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours",
    "HeightInMeters", "WeightInKilograms", "BMI",
]

# --- feature-engineering config (new in v2) --------------------------------
# Right-skewed numerics that benefit from a log1p transform; safe here since
# none of these raw values are negative.
SKEWED_LOG_COLS = ["PhysicalHealthDays", "MentalHealthDays", "BMI", "WeightInKilograms"]

# Severity-weighted comorbidity score. Weights are a rough clinical severity
# ordering (stroke/angina/kidney disease weighted above lower-severity items),
# not a fitted coefficient - this replaces the flat unweighted risk_condition_count.
COMORBIDITY_WEIGHTS = {
    "HadAngina": 3.0, "HadStroke": 3.0, "HadKidneyDisease": 2.5,
    "HadCOPD": 2.0, "HadArthritis": 1.0, "DifficultyWalking": 1.5,
    "ChestScan": 0.5,
}

SENIOR_AGE_CODE = 9    # AgeCategory code for "Age 65 to 69" and above
OBESE_BMI_THRESHOLD = 30.0


def first_existing(candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return candidates[0]


DEFAULT_INPUT = first_existing([
    "/content/heart_2022_with_nans.csv",
    r"C:\Users\Ayushman\OneDrive\Desktop\Heart DD(bigger)\2022\heart_2022_with_nans.csv",
    "heart_2022_with_nans.csv",
])
DEFAULT_OUTDIR = r"C:\Users\Ayushman\Downloads\out_dir"


def encode_target(series):
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    return series.astype("string").str.strip().str.lower().map(mapping)


class LeakageSafeSurveyPreprocessor:
    """Pandas preprocessor whose learned state comes only from the training split."""

    def __init__(self, seed=42, state_smoothing=50.0):
        self.seed = seed
        self.state_smoothing = state_smoothing

    def fit(self, X, y):
        X = X.copy()
        self.numeric_medians_ = {
            c: float(pd.to_numeric(X[c], errors="coerce").median())
            for c in NUMERIC_COLS if c in X
        }
        self.nominal_levels_ = {}
        for c in NOMINAL_COLS:
            if c in X:
                values = X[c].astype("string").fillna("__MISSING__")
                levels = sorted(values.dropna().unique().tolist())
                for special in ["__MISSING__", "__OTHER__"]:
                    if special not in levels:
                        levels.append(special)
                self.nominal_levels_[c] = levels

        self.state_prior_ = float(np.mean(y))
        state = X.get("State", pd.Series("__MISSING__", index=X.index)).astype("string").fillna("__MISSING__")
        stats = pd.DataFrame({"state": state.to_numpy(), "y": np.asarray(y)}) \
            .groupby("state", observed=True)["y"].agg(["mean", "count"])
        smooth = (
            stats["mean"] * stats["count"] + self.state_prior_ * self.state_smoothing
        ) / (stats["count"] + self.state_smoothing)
        self.state_target_map_ = smooth.to_dict()
        self.state_freq_map_ = state.value_counts(normalize=True).to_dict()
        return self

    def _base_transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        for c in NUMERIC_COLS:
            if c in X:
                values = pd.to_numeric(X[c], errors="coerce")
                out[c + "__missing"] = values.isna().astype(np.float32)
                out[c] = values.fillna(self.numeric_medians_[c]).astype(np.float32)

        yes_values = {"yes", "y", "true", "1"}
        no_values = {"no", "n", "false", "0"}
        for c in BINARY_YESNO_COLS:
            if c in X:
                normalized = X[c].astype("string").str.strip().str.lower()
                encoded = np.select(
                    [normalized.isin(yes_values), normalized.isin(no_values)],
                    [1.0, 0.0], default=-1.0,
                )
                out[c] = encoded.astype(np.float32)
                out[c + "__missing"] = (encoded < 0).astype(np.float32)

        for c, mapping in ORDINAL_MAPS.items():
            if c in X:
                encoded = X[c].map(mapping)
                out[c + "__missing"] = encoded.isna().astype(np.float32)
                out[c] = encoded.fillna(-1).astype(np.float32)

        # Cross-fitted target prevalence plus frequency retain State identity without leakage.
        state = X.get("State", pd.Series("__MISSING__", index=X.index)).astype("string").fillna("__MISSING__")
        out["State_target_rate"] = state.map(self.state_target_map_).fillna(self.state_prior_).astype(np.float32)
        out["State_frequency"] = state.map(self.state_freq_map_).fillna(0.0).astype(np.float32)

        for c, levels in self.nominal_levels_.items():
            values = X[c].astype("string").fillna("__MISSING__")
            known = set(levels)
            values = values.where(values.isin(known), "__OTHER__")
            categorical = pd.Categorical(values, categories=levels)
            dummies = pd.get_dummies(categorical, prefix=c, dtype=np.float32)
            dummies.index = X.index
            out = pd.concat([out, dummies], axis=1)

        # Low-variance, clinically interpretable aggregate/interaction features.
        risk_cols = [c for c in [
            "HadAngina", "HadStroke", "HadCOPD", "HadKidneyDisease", "HadArthritis",
            "DifficultyWalking", "ChestScan",
        ] if c in out]
        functional_cols = [c for c in [
            "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
            "DifficultyErrands", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
        ] if c in out]
        if risk_cols:
            out["risk_condition_count"] = out[risk_cols].clip(lower=0).sum(axis=1).astype(np.float32)
        if functional_cols:
            out["functional_limit_count"] = out[functional_cols].clip(lower=0).sum(axis=1).astype(np.float32)
        if {"PhysicalHealthDays", "MentalHealthDays"}.issubset(out.columns):
            out["total_unhealthy_days"] = (
                out["PhysicalHealthDays"] + out["MentalHealthDays"]
            ).clip(upper=60).astype(np.float32)
            out["max_unhealthy_days"] = out[["PhysicalHealthDays", "MentalHealthDays"]].max(axis=1).astype(np.float32)
        if {"AgeCategory", "BMI"}.issubset(out.columns):
            out["age_bmi_interaction"] = (out["AgeCategory"] * out["BMI"]).astype(np.float32)

        # ------------------------------------------------------------------ #
        # New in v2: additional engineered features, added to the SAME shared
        # transform so every downstream model receives them automatically.
        # ------------------------------------------------------------------ #

        # log1p transforms of right-skewed numerics (values are non-negative
        # after imputation above, so log1p is always safe here).
        for c in SKEWED_LOG_COLS:
            if c in out.columns:
                out[f"{c}_log1p"] = np.log1p(out[c].clip(lower=0)).astype(np.float32)

        # Severity-weighted comorbidity score (complements, does not replace,
        # the existing flat risk_condition_count above).
        present_weighted = [c for c in COMORBIDITY_WEIGHTS if c in out.columns]
        if present_weighted:
            out["comorbidity_severity_score"] = sum(
                out[c].clip(lower=0) * w for c, w in COMORBIDITY_WEIGHTS.items() if c in out.columns
            ).astype(np.float32)

        # Clinically-motivated interaction terms. SmokerStatus/HadDiabetes are
        # one-hot encoded above, so their dummy columns are located by prefix.
        smoker_cols = [c for c in out.columns if c.startswith("SmokerStatus_")]
        diabetes_cols = [c for c in out.columns if c.startswith("HadDiabetes_")]
        if "AgeCategory" in out.columns and smoker_cols:
            current_smoker_cols = [c for c in smoker_cols if "Current" in c]
            smoker_current = out[current_smoker_cols].sum(axis=1) if current_smoker_cols else 0.0
            out["age_x_current_smoker"] = (out["AgeCategory"] * smoker_current).astype(np.float32)
        if "BMI" in out.columns and diabetes_cols:
            diabetes_yes_cols = [c for c in diabetes_cols if "Yes" in c]
            diabetes_yes = out[diabetes_yes_cols].sum(axis=1) if diabetes_yes_cols else 0.0
            out["bmi_x_diabetes"] = (out["BMI"] * diabetes_yes).astype(np.float32)
        if {"ChestScan", "GeneralHealth"}.issubset(out.columns):
            out["chestscan_x_poor_health"] = (
                out["ChestScan"] * (out["GeneralHealth"] <= 1).astype(np.float32)
            ).astype(np.float32)

        # Clinically-binned age/BMI flags, kept alongside (not replacing) the
        # raw numeric values - trees can already find these splits, but this
        # helps the linear, custom, and deep-learning models converge faster.
        if "AgeCategory" in out.columns:
            out["age_band_senior"] = (out["AgeCategory"] >= SENIOR_AGE_CODE).astype(np.float32)
        if "BMI" in out.columns:
            out["bmi_band_obese"] = (out["BMI"] >= OBESE_BMI_THRESHOLD).astype(np.float32)

        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    def fit_transform(self, X, y):
        self.fit(X, y)
        out = self._base_transform(X)
        if "State" in X:
            # OOF encoding stops a training row's target from encoding itself.
            state = X["State"].astype("string").fillna("__MISSING__").reset_index(drop=True)
            y_array = np.asarray(y, dtype=np.float32)
            oof = np.full(len(X), self.state_prior_, dtype=np.float32)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            for fit_idx, hold_idx in skf.split(np.zeros(len(y_array)), y_array):
                fold = pd.DataFrame({"state": state.iloc[fit_idx].to_numpy(), "y": y_array[fit_idx]}) \
                    .groupby("state", observed=True)["y"].agg(["mean", "count"])
                fold_map = ((fold["mean"] * fold["count"] + self.state_prior_ * self.state_smoothing) /
                            (fold["count"] + self.state_smoothing)).to_dict()
                oof[hold_idx] = state.iloc[hold_idx].map(fold_map).fillna(self.state_prior_).to_numpy()
            out.loc[:, "State_target_rate"] = oof
        self.feature_names_ = out.columns.tolist()
        return out

    def transform(self, X):
        out = self._base_transform(X)
        return out.reindex(columns=self.feature_names_, fill_value=0.0)



# Extracted from user-supplied notebook, cell 04 (train-only MI/Pearson/VIF).
"""
04_custom_pipeline_improved.py
==============================
Friend's TRAIN-only MI -> Pearson -> VIF selection, applied ONLY to custom models,
followed by robust median-of-means logistic models.
"""

import argparse
import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

TARGET = "HadHeartAttack"


def stable_mi_scores(X, y, sample_size=100_000, repeats=3, seed=42):
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_mi, y_mi = X.iloc[idx], np.asarray(y)[idx]
    else:
        X_mi, y_mi = X, np.asarray(y)
    discrete = np.array([X_mi[c].nunique(dropna=False) <= 24 for c in X_mi.columns])
    scores = []
    for i in range(repeats):
        scores.append(mutual_info_classif(
            X_mi, y_mi, discrete_features=discrete, random_state=seed + i, n_jobs=-1,
        ))
    return pd.Series(np.mean(scores, axis=0), index=X.columns).sort_values(ascending=False)


def mi_filter(scores, percentile=10.0, min_features=25):
    cutoff = float(np.percentile(scores.to_numpy(), percentile))
    kept = scores[scores > cutoff].index.tolist()
    if len(kept) < min(min_features, len(scores)):
        kept = scores.head(min(min_features, len(scores))).index.tolist()
    print(f"[MI] cutoff={cutoff:.6f}; kept {len(kept)}/{len(scores)}")
    return kept


def correlation_filter(X, relevance, threshold=0.98):
    corr = X.corr().abs().fillna(0.0)
    ordered = relevance.loc[X.columns].sort_values(ascending=False).index.tolist()
    dropped = set()
    for i, left in enumerate(ordered):
        if left in dropped:
            continue
        later = [right for right in ordered[i + 1:] if right not in dropped]
        dropped.update(right for right in later if corr.at[left, right] > threshold)
    kept = [c for c in ordered if c not in dropped]
    print(f"[PCC] |r|>{threshold}; dropped {len(dropped)}, kept {len(kept)}")
    return kept


def vif_filter(X, threshold=20.0, sample_size=30_000, max_drops=30, seed=42):
    """Fast iterative VIF using the diagonal of a ridge-stabilized inverse correlation matrix."""
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=seed)
    features = X.columns[X.var() > 1e-12].tolist()
    dropped = []
    while len(features) > 2 and len(dropped) < max_drops:
        Z = X[features].to_numpy(np.float64)
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
        corr = np.corrcoef(Z, rowvar=False)
        inv_corr = np.linalg.inv(corr + np.eye(len(features)) * 1e-8)
        vifs = np.diag(inv_corr)
        worst = int(np.argmax(vifs))
        if float(vifs[worst]) <= threshold:
            break
        dropped.append((features.pop(worst), float(vifs[worst])))
    print(f"[VIF] threshold={threshold}; dropped {len(dropped)}, kept {len(features)}")
    return features, dropped


def select_custom_features(X_train, y_train, args):
    scores = stable_mi_scores(X_train, y_train, args.mi_sample, args.mi_repeats, args.seed)
    mi_kept = mi_filter(scores, args.mi_percentile, args.min_features)
    pcc_kept = correlation_filter(X_train[mi_kept], scores, args.pcc_threshold)
    final_kept, vif_dropped = vif_filter(
        X_train[pcc_kept], args.vif_threshold, args.vif_sample, args.vif_max_drops, args.seed,
    )
    if len(final_kept) < args.min_features:
        final_kept = pcc_kept[:args.min_features]
        print("[VIF] conservative fallback activated to preserve minimum feature count")
    report = {
        "statistics_scope": "training split only",
        "original_count": int(X_train.shape[1]),
        "final_count": len(final_kept),
        "kept_columns": final_kept,
        "mi_scores": scores.to_dict(),
        "dropped_after_mi": sorted(set(X_train.columns) - set(mi_kept)),
        "dropped_by_pcc": sorted(set(mi_kept) - set(pcc_kept)),
        "dropped_by_vif": [name for name, _ in vif_dropped],
    }
    return final_kept, report



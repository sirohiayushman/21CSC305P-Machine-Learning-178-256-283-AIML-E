"""Retrain portable CatBoost variants from the same *raw* BRFSS-2022 CSV.

All feature engineering, MI / PCC / VIF selection and scaling are fitted using
training rows only. Calibration, threshold fitting and variant selection are
separated. Test split is scored exactly once after selection.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import (TARGET, BINARY_YESNO_COLS, NUMERIC_COLS, NOMINAL_COLS,
                           ORDINAL_MAPS, LeakageSafeSurveyPreprocessor, encode_target)
from feature_selection import select_custom_features

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / 'models' / 'screening_bundle.joblib'
REPORT = ROOT / 'models' / 'training_report.json'
NOTEBOOK_BENCHMARK = {
    'model': 'CatBoost (Advanced, Tuned)', 'accuracy': 0.9108,
    'precision': 0.3431, 'recall': 0.6243, 'specificity': 0.9280,
    'f1': 0.4428, 'roc_auc': 0.8912, 'pr_auc': 0.4260,
    'threshold': 0.1219,
    'note': 'Notebook held-out test, cost-weighted FN=5, FP=1; not a result for a newly trained artifact',
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1-1e-6)
    return np.log(p / (1-p)).reshape(-1, 1)


def calibrate(calibrator, raw_p):
    return calibrator.predict_proba(safe_logit(raw_p))[:, 1]


def get_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    labels = (p >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, labels, labels=[0, 1]).ravel())
    return {
        'threshold': round(float(threshold), 6),
        'accuracy': float(accuracy_score(y, labels)),
        'precision': float(precision_score(y, labels, zero_division=0)),
        'recall': float(recall_score(y, labels, zero_division=0)),
        'specificity': tn / max(tn+fp, 1),
        'f1': float(f1_score(y, labels, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier_score': float(brier_score_loss(y, p)),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'weighted_error_cost': int(5 * fn + fp),
    }


def select_thresholds(y, p, target_recall=0.80, fn_cost=5.0):
    # Candidate thresholds are derived solely from the threshold-validation split.
    grid = np.unique(np.r_[0.000001, 0.01, 0.5, 0.99, 0.999999,
                             np.quantile(p, np.linspace(0, 1, 750))])
    y = np.asarray(y, dtype=int)
    best_cost, best_f1 = (float('inf'), 0.5), (-1.0, 0.5)
    best_high = None
    for thr in grid:
        tp = int(((p >= thr) & (y == 1)).sum())
        fp = int(((p >= thr) & (y == 0)).sum())
        fn = int(((p < thr) & (y == 1)).sum())
        recall = tp / max(tp+fn, 1)
        precision = tp / max(tp+fp, 1)
        f1 = (2*recall*precision/(recall+precision)) if (recall+precision) else 0
        cost = fn_cost * fn + fp
        if cost < best_cost[0]:
            best_cost = cost, thr
        if f1 > best_f1[0]:
            best_f1 = f1, thr
        if recall >= target_recall:
            # Highest feasible threshold to limit false alarms at chosen recall.
            best_high = float(thr)
    return {'balanced': float(best_f1[1]),
            'cautious': float(best_cost[1]),
            'high_recall': float(best_high if best_high is not None else 0.000001)}


def schema_for(prep):
    # Values reflect levels actually present in the fitted training split.
    return {'nominal_options': {k: [v for v in levels if not v.startswith('__')]
                                for k, levels in prep.nominal_levels_.items()},
            'state_options': sorted(k for k in prep.state_freq_map_ if k != '__MISSING__')}


def save_bundle(bundle, artifact=ARTIFACT):
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix('.joblib.tmp')
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, artifact)
    return artifact


def fit(args):
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f'Raw CSV not found: {data_path}')
    print(f'[load] {data_path}')
    raw = pd.read_csv(data_path, low_memory=False)
    if TARGET not in raw.columns:
        raise ValueError(f'Missing {TARGET}. Expected raw BRFSS 2022 CSV, not engineered train.csv.')
    required = {'AgeCategory', 'HadAngina', 'HadStroke', 'Sex'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Input is missing expected raw survey fields: {sorted(missing)}')
    raw[TARGET] = encode_target(raw[TARGET])
    raw = raw.dropna(subset=[TARGET]).reset_index(drop=True)
    raw[TARGET] = raw[TARGET].astype('int8')
    X, y = raw.drop(columns=TARGET), raw[TARGET]
    if len(y) < 200 or y.nunique() != 2 or y.sum() < 50:
        raise ValueError('Provide a training dataset with at least 200 rows and 50 positive cases.')
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85,
        stratify=y_tv, random_state=args.seed)
    X_tr = X_tr.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print('[preprocess] fit on training rows only (incl. OOF State encoding)')
    prep = LeakageSafeSurveyPreprocessor(seed=args.seed)
    tr = prep.fit_transform(X_tr, y_tr)
    val = prep.transform(X_val)
    # Do not transform test until after model variant chosen.
    indices = np.arange(len(y_val))
    cal_idx, other = train_test_split(
        indices, test_size=0.60, stratify=y_val, random_state=args.seed)
    sel_idx, thr_idx = train_test_split(
        other, test_size=0.50, stratify=y_val.iloc[other], random_state=args.seed)
    selected_report = None
    candidates = {'all_engineered': list(tr.columns)}
    if not args.skip_selection:
        selargs = SimpleNamespace(mi_sample=args.mi_sample, mi_repeats=args.mi_repeats,
                seed=args.seed, mi_percentile=10.0, min_features=25,
                pcc_threshold=0.98, vif_threshold=20.0,
                vif_sample=args.vif_sample, vif_max_drops=30)
        print('[selection] MI -> Pearson -> VIF (training rows only)')
        chosen, selected_report = select_custom_features(tr, y_tr.to_numpy(), selargs)
        if len(chosen) >= 2 and set(chosen) != set(tr.columns):
            candidates['mi_pearson_vif'] = chosen

    # Original notebook tuned params; original final CatBoost early-stopped near
    # 193 iterations. Fixed iteration count avoids reusing selection/calibration rows.
    specs = {'iterations': args.iterations, 'learning_rate': 0.09219562684963471,
             'depth': 5, 'l2_leaf_reg': 4.3593487625511855,
             'auto_class_weights': 'Balanced', 'random_seed': args.seed,
             'thread_count': args.threads, 'verbose': False,
             'allow_writing_files': False, 'task_type': 'CPU'}
    trained = {}
    candidate_report = {}
    for name, cols in candidates.items():
        print(f'[model] {name}: {len(cols)} features; CatBoost {args.iterations} trees')
        scaler = StandardScaler().fit(tr[cols].to_numpy())
        xt = scaler.transform(tr[cols].to_numpy())
        xv = scaler.transform(val[cols].to_numpy())
        model = CatBoostClassifier(**specs)
        model.fit(xt, y_tr.to_numpy())
        raw_val = model.predict_proba(xv)[:, 1]
        cal = LogisticRegression(C=10.0, max_iter=1000)
        cal.fit(safe_logit(raw_val[cal_idx]), y_val.iloc[cal_idx])
        p_val = calibrate(cal, raw_val)
        thresholds = select_thresholds(y_val.iloc[thr_idx].to_numpy(), p_val[thr_idx],
                                        target_recall=args.target_recall, fn_cost=args.fn_cost)
        sel_metrics = get_metrics(y_val.iloc[sel_idx].to_numpy(), p_val[sel_idx], thresholds['cautious'])
        candidate_report[name] = {
            'feature_count': len(cols), 'thresholds': thresholds,
            'selection_split_metrics': sel_metrics,
            'threshold_split_metrics': {
                k: get_metrics(y_val.iloc[thr_idx], p_val[thr_idx], v)
                for k, v in thresholds.items()},
        }
        trained[name] = (model, scaler, cols, cal, thresholds)
        print(f"  [selection] recall={sel_metrics['recall']:.3f} "
              f"accuracy={sel_metrics['accuracy']:.3f} "
              f"cost={sel_metrics['weighted_error_cost']}")

    # Choose by cost-weighted error on a validation subset that was *not* used
    # for thresholding; no peeking at test set or cherry-picking from test.
    winner = min(trained, key=lambda key: (
        candidate_report[key]['selection_split_metrics']['weighted_error_cost'],
        -candidate_report[key]['selection_split_metrics']['recall'],
        -candidate_report[key]['selection_split_metrics']['pr_auc']))
    model, scaler, cols, cal, thresholds = trained[winner]
    print(f'[select] {winner} (independent validation subset)')
    xtest = prep.transform(X_test)
    test_p = calibrate(cal, model.predict_proba(scaler.transform(xtest[cols].to_numpy()))[:, 1])
    test_metrics = {k: get_metrics(y_test, test_p, threshold) for k, threshold in thresholds.items()}
    train_report = {
        'source': 'retrained_from_raw_csv', 'chosen_variant': winner,
        'raw_rows': len(raw), 'train_rows': len(y_tr), 'validation_rows': len(y_val),
        'test_rows': len(y_test), 'positive_prevalence': float(y.mean()),
        'preprocessed_feature_count': len(tr.columns),
        'selected_feature_count': len(cols),
        'feature_selection_report': selected_report,
        'candidate_comparison': candidate_report,
        'test_metrics': test_metrics,
        'notebook_reference_only': NOTEBOOK_BENCHMARK,
    }
    meta = {
        'model_name': 'CatBoost', 'variant': winner,
        'source': 'Locally retrained on supplied raw BRFSS 2022 dataset',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_train_rows': len(y_tr), 'input_features': len(cols),
        'engineered_features': len(tr.columns),
        'target': TARGET,
        'known_performance': test_metrics,
        'thresholds': thresholds,
        'limit': ('Predicts association with self-reported prior heart attack; '
                  'not a diagnosis or prospective heart attack probability.'),
    }
    bundle = {'model': model, 'scaler': scaler, 'columns': cols,
              'preprocessor_state': prep.__dict__, 'calibrator': cal,
              'thresholds': thresholds, 'schema': schema_for(prep),
              'metadata': meta}
    path = save_bundle(bundle)
    REPORT.write_text(json.dumps(train_report, indent=2, default=str), encoding='utf-8')
    print(f'[saved] {path} \n[report] {REPORT}')
    for mode, metrics in test_metrics.items():
        print(f"[test] {mode}: threshold={metrics['threshold']:.4f}, "
              f"accuracy={metrics['accuracy']:.4f}, recall={metrics['recall']:.4f}, "
              f"specificity={metrics['specificity']:.4f}, FN={metrics['fn']}")
    return train_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Path to raw heart_2022_with_nans.csv')
    parser.add_argument('--iterations', type=int, default=193,
                        help='Final trees; original notebook CatBoost best iteration ~193')
    parser.add_argument('--threads', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mi-sample', type=int, default=100000)
    parser.add_argument('--mi-repeats', type=int, default=3)
    parser.add_argument('--vif-sample', type=int, default=30000)
    parser.add_argument('--skip-selection', action='store_true',
                        help='Train only notebook-faithful full-feature CatBoost')
    parser.add_argument('--fn-cost', type=float, default=5.0)
    parser.add_argument('--target-recall', type=float, default=0.80)
    fit(parser.parse_args())

"""Import the actual saved notebook CatBoost model without retraining.

IMPORTANT: The notebook did NOT save its StandardScaler in cell 03; the importer
reconstructs the exact scaler from the *same notebook-generated* train.csv.
Never fit it on new patient data or substitute a different training split.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing import LeakageSafeSurveyPreprocessor
from train import ARTIFACT, save_bundle, schema_for


def importer(folder):
    folder = Path(folder)
    expected = ['model_catboost.joblib', 'survey_preprocessor.joblib',
                'train.csv', 'probability_calibrators.joblib',
                'validation_selected_thresholds.json']
    missing = [f for f in expected if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError('Missing notebook output files in folder: ' + ', '.join(missing))
    print('[import] Loading original CatBoost, calibration, thresholds and preprocessing')
    model = joblib.load(folder / 'model_catboost.joblib')
    state = joblib.load(folder / 'survey_preprocessor.joblib')
    prep = LeakageSafeSurveyPreprocessor()
    prep.__dict__.update(state)
    cols = list(prep.feature_names_)
    train = pd.read_csv(folder / 'train.csv', usecols=cols, low_memory=False)
    # Matches StandardScaler().fit_transform(X_train) from notebook cell 03.
    scaler = StandardScaler().fit(train[cols].to_numpy())
    if model.n_features_in_ != len(cols):
        raise ValueError(f'Inconsistent model features: {model.n_features_in_} vs {len(cols)}')
    calibrators = joblib.load(folder / 'probability_calibrators.joblib')
    if 'catboost' not in calibrators:
        raise ValueError('Notebook calibrators file lacks catboost entry.')
    threshold_map = json.loads((folder / 'validation_selected_thresholds.json').read_text())
    thresholds = {
        'balanced': float(threshold_map['f1_optimal']['catboost']),
        'cautious': float(threshold_map['cost_weighted']['catboost']),
    }
    meta = {
        'model_name': 'CatBoost', 'variant': 'original_notebook_full_features',
        'source': 'Imported original notebook files (model + scaler reconstructed from train.csv)',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_train_rows': len(train), 'input_features': len(cols),
        'engineered_features': len(cols), 'target': 'HadHeartAttack',
        'thresholds': thresholds, 'known_performance': None,
        'limit': 'Predicts association with self-reported prior heart attack, not future events.',
        'note': 'Notebook MI/Pearson/VIF selection was used for custom models only, NOT this CatBoost.',
    }
    bundle = {'model': model, 'scaler': scaler, 'columns': cols,
              'preprocessor_state': prep.__dict__,
              'calibrator': calibrators['catboost'],
              'thresholds': thresholds, 'schema': schema_for(prep), 'metadata': meta}
    path = save_bundle(bundle)
    print('[saved]', path)
    print('[source] Original notebook CatBoost. No retraining; no feature selection on this model.')
    return path

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', required=True,
                    help='Notebook out_dir holding model_catboost.joblib etc.')
    importer(ap.parse_args().folder)

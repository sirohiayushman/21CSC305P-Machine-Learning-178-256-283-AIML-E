"""Local-only FastAPI inference server for the BRFSS Heart-Attack History Screener."""
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from preprocessing import (BINARY_YESNO_COLS, NOMINAL_COLS, NUMERIC_COLS,
                           ORDINAL_MAPS, LeakageSafeSurveyPreprocessor)
from train import ARTIFACT, NOTEBOOK_BENCHMARK, calibrate

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='BRFSS Heart Attack History Screener', version='1.0',
              description='Research demonstration only; localhost, not clinical decision support.')

RAW_COLS = (['State'] + NUMERIC_COLS + BINARY_YESNO_COLS +
            list(ORDINAL_MAPS) + NOMINAL_COLS)

# Cap extreme input errors; not claims about clinical reference ranges.
VALID_RANGES = {
    'BMI': (8, 100), 'SleepHours': (0, 24),
    'PhysicalHealthDays': (0, 30), 'MentalHealthDays': (0, 30),
    'HeightInMeters': (0.5, 2.7), 'WeightInKilograms': (10, 500),
}
REQUIRED = ['AgeCategory', 'Sex', 'GeneralHealth', 'BMI']


class ScreeningRequest(BaseModel):
    answers: dict[str, Any] = Field(description='Raw BRFSS answers, not engineered inputs')
    mode: Literal['balanced', 'cautious', 'high_recall'] = 'cautious'


@lru_cache(maxsize=2)
def _load_file(path: str, mtime_ns: int):
    # Load only locally created / trusted joblib files. Never accept a user file path via API.
    return joblib.load(path)


def get_bundle():
    if not ARTIFACT.is_file():
        raise HTTPException(503, detail=(
            'Model not prepared. Train with python train.py --data path/to/heart_2022_with_nans.csv '
            'or import notebook artifacts with import_notebook.py.'))
    try:
        return _load_file(str(ARTIFACT), ARTIFACT.stat().st_mtime_ns)
    except Exception as ex:
        raise HTTPException(503, detail=f'Could not load local model: {type(ex).__name__}') from ex


def groups(bundle=None):
    nominal = bundle['schema']['nominal_options'] if bundle else {}
    state = bundle['schema']['state_options'] if bundle else []
    defaults = {
        'Sex': ['Female', 'Male'],
        'HadDiabetes': ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 'Yes, but only during pregnancy (female)'],
        'SmokerStatus': ['Never smoked', 'Former smoker', 'Current smoker - now smokes every day', 'Current smoker - now smokes some days'],
        'ECigaretteUsage': ['Never used e-cigarettes in my entire life', 'Not at all (right now)', 'Every day', 'Some days'],
        'RaceEthnicityCategory': ['White only, Non-Hispanic', 'Black only, Non-Hispanic', 'Hispanic', 'Multiracial, Non-Hispanic', 'Other race only, Non-Hispanic'],
        'TetanusLast10Tdap': ['Yes, received Tdap', 'Yes, received tetanus shot but not sure what type', 'No'],
        'CovidPos': ['Yes', 'No'],
    }
    section_data = [
        ('Personal details', 'user-round', ['AgeCategory', 'Sex', 'RaceEthnicityCategory', 'State']),
        ('Health measures', 'activity', ['GeneralHealth', 'BMI', 'HeightInMeters', 'WeightInKilograms',
                                       'SleepHours', 'PhysicalHealthDays', 'MentalHealthDays',
                                       'LastCheckupTime', 'RemovedTeeth']),
        ('Medical background', 'heart-pulse', ['HadAngina', 'HadStroke', 'HadDiabetes',
            'HadKidneyDisease', 'HadCOPD', 'HadAsthma', 'HadArthritis', 'HadSkinCancer',
            'HadDepressiveDisorder', 'ChestScan', 'HIVTesting']),
        ('Lifestyle & other history', 'clipboard-list', ['PhysicalActivities', 'SmokerStatus',
            'ECigaretteUsage', 'AlcoholDrinkers', 'DifficultyWalking',
            'DifficultyConcentrating', 'DifficultyDressingBathing', 'DifficultyErrands',
            'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'FluVaxLast12',
            'PneumoVaxEver', 'HighRiskLastYear', 'TetanusLast10Tdap', 'CovidPos']),
    ]
    def field_def(c):
        label = ''.join((' ' + x if x.isupper() and i and not (c[i-1].isupper()) else x)
                        for i, x in enumerate(c)).strip().replace('In Meters', '(m)').replace('In Kilograms', '(kg)')
        if c == 'State': opts, typ = state, 'select'
        elif c in NUMERIC_COLS: opts, typ = [], 'number'
        elif c in BINARY_YESNO_COLS: opts, typ = ['Yes', 'No'], 'select'
        elif c in ORDINAL_MAPS: opts, typ = list(ORDINAL_MAPS[c]), 'select'
        else: opts, typ = nominal.get(c, defaults.get(c, [])), 'select'
        return {'name': c, 'label': label, 'type': typ, 'options': opts,
                'required': c in REQUIRED,
                'min': VALID_RANGES.get(c, (None, None))[0],
                'max': VALID_RANGES.get(c, (None, None))[1]}
    return [{'title': title, 'icon': icon, 'fields': [field_def(c) for c in columns]}
            for title, icon, columns in section_data]


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')


@app.get('/static/style.css')
def style():
    return FileResponse(ROOT / 'static' / 'style.css', media_type='text/css')


@app.get('/static/app.js')
def js():
    return FileResponse(ROOT / 'static' / 'app.js', media_type='application/javascript')


@app.get('/api/health')
def health():
    return {'status': 'running', 'model_ready': ARTIFACT.is_file()}


@app.get('/api/schema')
def schema():
    bundle = None
    if ARTIFACT.is_file():
        bundle = get_bundle()
    return {'model_ready': bundle is not None,
            'model': bundle['metadata'] if bundle else None,
            'sections': groups(bundle), 'required': REQUIRED,
            'notebook_reference': NOTEBOOK_BENCHMARK,
            'warning': ('The dataset labels self-reported past heart attacks. '
                        'It cannot diagnose a current event or predict future events.')}


@app.post('/api/predict')
def predict(request: ScreeningRequest):
    bundle = get_bundle()
    raw = request.answers
    extras = sorted(set(raw) - set(RAW_COLS))
    if extras:
        raise HTTPException(422, detail=f'Unknown inputs: {extras}')
    for key in REQUIRED:
        if key not in raw or raw[key] is None or str(raw[key]).strip() == '':
            raise HTTPException(422, detail=f'Missing required field: {key}')
    cleaned = {}
    for c in RAW_COLS:
        value = raw.get(c)
        if value is None or (isinstance(value, str) and not value.strip()):
            value = None
        if c in NUMERIC_COLS and value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise HTTPException(422, detail=f'{c} must be numeric')
            if not np.isfinite(value):
                raise HTTPException(422, detail=f'{c} must be finite')
            if c in VALID_RANGES:
                low, high = VALID_RANGES[c]
                if not (low <= value <= high):
                    raise HTTPException(422, detail=f'{c} must be between {low} and {high}')
        if c in ORDINAL_MAPS and value is not None and value not in ORDINAL_MAPS[c]:
            raise HTTPException(422, detail=f'Invalid option for {c}')
        if c in BINARY_YESNO_COLS and value is not None and value not in ('Yes', 'No'):
            raise HTTPException(422, detail=f'{c} must be Yes or No')
        if c in NOMINAL_COLS and value is not None and value not in bundle['schema']['nominal_options'].get(c, []):
            raise HTTPException(422, detail=f'Invalid option for {c}')
        if c == 'State' and value is not None and value not in bundle['schema']['state_options']:
            raise HTTPException(422, detail='Invalid State value')
        cleaned[c] = value
    missing = [c for c, value in cleaned.items() if value is None]
    prep = LeakageSafeSurveyPreprocessor()
    prep.__dict__.update(bundle['preprocessor_state'])
    engineered = prep.transform(pd.DataFrame([cleaned]))
    missing_model_cols = set(bundle['columns']) - set(engineered.columns)
    if missing_model_cols:
        raise HTTPException(503, detail='Stored feature schema does not match model.')
    features = engineered[bundle['columns']].to_numpy(dtype=np.float64)
    scaled = bundle['scaler'].transform(features)
    raw_proba = float(bundle['model'].predict_proba(scaled)[0, 1])
    prob = float(calibrate(bundle['calibrator'], [raw_proba])[0])
    thresholds = bundle['thresholds']
    if request.mode not in thresholds:
        raise HTTPException(422, detail=f'Mode {request.mode} unavailable for imported model. '
                        f'Available: {list(thresholds)}')
    thr = float(thresholds[request.mode])
    flag = bool(prob >= thr)
    return {
        'flagged_for_follow_up': flag,
        'screening_score': round(prob, 6), 'threshold': round(thr, 6),
        'mode': request.mode, 'input_fields_provided': len(RAW_COLS) - len(missing),
        'input_fields_total': len(RAW_COLS), 'unanswered_fields': missing,
        'model': bundle['metadata']['model_name'],
        'message': ('Screening flag raised: discuss with a healthcare professional.' if flag
                    else 'No flag at the selected threshold. This does not rule out heart disease.'),
        'disclaimer': ('For research and education only. Dataset target is self-reported '
                       'past heart attack. This is NOT the probability of a future heart '
                       'attack and NOT a medical diagnosis. If you have chest pain, severe '
                       'shortness of breath, fainting, or other possible heart-attack '
                       'symptoms, seek urgent emergency medical care regardless of score.'),
    }

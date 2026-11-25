from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'running'


def test_home():
    response = client.get('/')
    assert response.status_code == 200
    assert 'cardioscope' in response.text.lower()


def test_schema_fields():
    response = client.get('/api/schema')
    assert response.status_code == 200
    data = response.json()
    assert 'AgeCategory' in data['required']
    keys = {f['name'] for section in data['sections'] for f in section['fields']}
    assert {'AgeCategory', 'BMI', 'Sex', 'HadStroke', 'ChestScan'} <= keys
    assert len(keys) == 39


def test_missing_model_refuses_to_guess():
    from train import ARTIFACT
    if not ARTIFACT.exists():
        response = client.post('/api/predict', json={
            'mode': 'cautious',
            'answers': {'AgeCategory': 'Age 65 to 69', 'Sex': 'Male', 'BMI': 28,
                        'GeneralHealth': 'Good'},
        })
        assert response.status_code == 503

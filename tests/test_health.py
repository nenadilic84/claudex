from fastapi.testclient import TestClient
import unittest.mock
import os

# Mock environment variables before importing app
with unittest.mock.patch.dict(os.environ, {
    'TARGET_API_BASE': 'https://api.example.com',
    'TARGET_API_KEY': 'mock-api-key',
    'BIG_MODEL_TARGET': 'model-large',
    'SMALL_MODEL_TARGET': 'model-small'
}):
    from claudex.proxy import app

def test_health_check():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data

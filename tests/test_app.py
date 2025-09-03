from fastapi.testclient import TestClient
import src.app as app

client = TestClient(app.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"

def test_predict_one_mock():
    # Отключаем проверку схемы на время теста
    try:
        app.INPUT_SCHEMA = None
    except Exception:
        pass

    class Dummy:
        def predict_proba(self, df):
            import numpy as np
            n = len(df)
            # всегда 0.7 на класс 1
            return np.hstack([np.full((n,1), 0.3), np.full((n,1), 0.7)])
        def predict(self, df):
            return [0]*len(df)

    app.pipe = Dummy()
    body = {"features": {}}
    r = client.post("/predict_one", json=body)
    assert r.status_code == 200
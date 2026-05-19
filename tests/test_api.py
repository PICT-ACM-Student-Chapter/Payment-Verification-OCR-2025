import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_healthcheck():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_create_run_rejects_invalid_registration_type():
    response = client.post(
        "/api/runs",
        files=[
            ("registration_file", ("registration.pdf", io.BytesIO(b"bad"), "application/pdf")),
            ("report_files", ("report.csv", io.BytesIO(b"rrn,amount\n123,500"), "text/csv")),
        ],
        data={"config": '{"regTransactionIdColumn":"transactionId","useFallback":true}'},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_results_endpoint_404_for_unknown_run():
    response = client.get("/api/runs/missing-run/results")
    assert response.status_code == 404


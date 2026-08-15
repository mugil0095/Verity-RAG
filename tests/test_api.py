import pytest
from fastapi.testclient import TestClient

from verityrag import api
from verityrag.pipeline import VerityRAGPipeline


@pytest.fixture()
def client():
    api.pipeline = VerityRAGPipeline()  # fresh pipeline per test -> no cross-test state leakage
    return TestClient(api.app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_endpoint_returns_service_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "VerityRAG"
    assert body["docs"] == "/docs"


def test_ingest_endpoint_adds_document(client):
    resp = client.post("/ingest", json={
        "doc_id": "d1", "title": "Tesla",
        "text": "Nikola Tesla was a Serbian-American inventor and electrical engineer.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "d1"
    assert body["chunks_added"] >= 1
    assert body["index_size"] >= 1


def test_query_endpoint_returns_grounded_answer_schema(client):
    client.post("/ingest", json={
        "doc_id": "d1", "title": "Tesla",
        "text": "Nikola Tesla was a Serbian-American inventor and electrical engineer famous "
                "for his contributions to the design of the modern alternating current system.",
    })
    resp = client.post("/query", json={"question": "What is Nikola Tesla known for?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and "abstained" in body and "claims" in body
    assert body["abstained"] is False
    assert body["answer"]


def test_query_endpoint_abstains_on_empty_index(client):
    resp = client.post("/query", json={"question": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is True
    assert body["answer"] is None


def test_stats_endpoint_reports_index_size(client):
    client.post("/ingest", json={"doc_id": "d1", "title": "T", "text": "Some real sentence content here."})
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["index_size"] >= 1

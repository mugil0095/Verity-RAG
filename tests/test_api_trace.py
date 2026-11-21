import pytest
from fastapi.testclient import TestClient

from verityrag import api
from verityrag.pipeline import VerityRAGPipeline

TESLA_DOC = {"doc_id": "d1", "title": "Tesla",
             "text": "Nikola Tesla was a Serbian-American inventor and electrical engineer. He is best known "
                     "for his contributions to the design of the modern alternating current electricity supply system."}
WARSAW_DOC = {"doc_id": "d2", "title": "Warsaw",
              "text": "Warsaw is the capital and largest city of Poland. It lies on the Vistula river in east-central Poland."}


@pytest.fixture()
def client():
    api.pipeline = VerityRAGPipeline()
    c = TestClient(api.app)
    c.post("/ingest", json=TESLA_DOC)
    c.post("/ingest", json=WARSAW_DOC)
    return c


def test_single_question_has_a_trace_and_no_plan(client):
    body = client.post("/query", json={"question": "What is Nikola Tesla known for?"}).json()
    assert body["abstained"] is False and body["abstain_reason"] is None
    assert body["plan"] == [] and body["parts"] == []
    assert body["trace"][0]["action"] == "retrieve" and body["trace"][-1]["action"] == "answer"
    assert body["evidence"] and {"chunk_id", "doc_id", "title", "text", "dense_score"} <= set(body["evidence"][0])


def test_two_part_question_returns_the_plan_and_each_agents_steps(client):
    body = client.post("/query", json={
        "question": "What is Nikola Tesla known for? Also, where is Warsaw located?"}).json()
    assert body["plan"] == ["What is Nikola Tesla known for?", "Where is Warsaw located?"]
    assert len(body["parts"]) == 2
    assert body["trace"][0]["action"] == "plan"
    for part in body["parts"]:
        assert part["abstained"] is False and part["evidence_chunk_ids"]
        assert part["retriever_steps"][0]["action"] == "retrieve"
        assert part["verifier_steps"][-1]["action"] == "answer"
    assert "Poland" in body["answer"]


def test_abstain_reason_is_returned(client):
    api.pipeline = VerityRAGPipeline()  # empty index
    body = client.post("/query", json={"question": "anything at all"}).json()
    assert body["abstained"] is True and body["abstain_reason"] == "sufficiency"


def test_cors_allows_the_react_dev_server_only(client):
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/query", headers={"Origin": "http://localhost:5173", **ask})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.options("/query", headers={"Origin": "http://evil.example", **ask})
    assert "access-control-allow-origin" not in other.headers

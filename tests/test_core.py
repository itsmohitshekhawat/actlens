"""Fast tests that need no API key: the corpus, search, citation checking, budget and the web routes."""

import json

import pytest
from fastapi.testclient import TestClient

from app import agent, budget
from app.main import app
from app.search import DATA, get_index, ref_of


def test_corpus_has_every_article_annex_and_recital():
    chunks = json.loads((DATA / "corpus.json").read_text())["chunks"]
    assert {c["article"] for c in chunks if c["kind"] == "article"} == set(range(1, 114))
    assert len({c["annex"] for c in chunks if c["kind"] == "annex"}) == 13
    assert len([c for c in chunks if c["kind"] == "recital"]) == 180
    assert max(len(c["text"]) for c in chunks) < 6000  # no giant chunks (Article 3 is split per definition)


@pytest.mark.parametrize("query,expected", [
    ("prohibited AI practices social scoring", "Art 5"),
    ("Article 50", "Art 50"),
    ("penalties administrative fines EUR 35 000 000", "Art 99"),
])
def test_search_finds_the_right_article(query, expected):
    assert expected in [ref_of(h) for h in get_index().search(query, k=5)]


def test_read_references():
    ix = get_index()
    assert ix.article("Art. 99(3)")[0]["label"] == "Art. 99(3)"
    assert ix.article("Annex III")[0]["annex"] == "III"
    assert ix.article("Recital 27")[0]["label"] == "Recital 27"
    assert ix.article("nonsense") == []


def test_citation_check_flags_unseen_citations():
    seen = {"Art. 5(1)": "...", "Annex III (part 1)": "..."}
    result = agent.check_citations("Banned [Art. 5(1)(f)]. High-risk [Annex III]. Made up [Art. 42(2)].", seen)
    assert [c["verified"] for c in result["citations"]] == [True, True, False]
    assert result["grounded"] is False


def test_budget_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "DB_PATH", tmp_path / "b.sqlite")
    monkeypatch.setattr(budget, "PER_VISITOR_PER_DAY", 2)
    assert budget.status("1.1.1.1")["open"]
    budget.record("1.1.1.1", 0.001)
    budget.record("1.1.1.1", 0.001)
    assert not budget.status("1.1.1.1")["open"]
    assert budget.status("2.2.2.2")["open"]


def test_routes():
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"]
        assert "ActLens" in client.get("/").text
        r = client.get("/api/search", params={"q": "deepfake labelling", "mode": "hybrid"})
        assert r.status_code == 200 and r.json()["results"]
        assert client.get("/api/search", params={"q": "x" * 500}).status_code == 422
        assert client.get("/api/eval").status_code == 200


def test_source_matching_prefers_the_exact_paragraph():
    seen = {"Art. 5(8)": "a", "Art. 5(1)": "b", "Annex III (part 1)": "c"}
    assert agent.best_source("Art. 5(1)(f)", seen) == "Art. 5(1)"
    assert agent.best_source("Art. 5", seen) in ("Art. 5(8)", "Art. 5(1)")
    assert agent.best_source("Annex III, point 1(c)", seen) == "Annex III (part 1)"

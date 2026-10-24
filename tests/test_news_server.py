"""News MCP server with a fake Tavily (httpx.MockTransport). Offline: no key, no credits."""

import asyncio
import json

import httpx
import pytest

from src.mcp_servers import news_server as ns

ARTICLES = [
    {"title": "Toyota raises outlook", "url": "https://www.reuters.com/a1", "content": "Toyota said " + "x" * 600,
     "score": 0.9, "published_date": "2026-10-05"},
    {"title": "Duplicate", "url": "https://www.reuters.com/a1", "content": "dup", "score": 0.5},
    {"title": "Toyota EV plan", "url": "https://nikkei.com/b2", "content": "EV plan", "score": 0.7},
]


class FakeTavily:
    def __init__(self, status: int = 200, body: dict | None = None, headers: dict | None = None):
        self.status, self.body, self.headers = status, body or {"results": ARTICLES}, headers or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, headers=self.headers)


@pytest.fixture
def tavily(monkeypatch):
    fake = FakeTavily()
    ns._cache.clear()
    monkeypatch.setattr(ns.settings, "tavily_api_key", "tvly-test-key")
    monkeypatch.setattr(ns, "_transport", httpx.MockTransport(fake))
    return fake


def test_results_are_parsed_trimmed_and_deduplicated(tavily):
    r = ns.search_news("Toyota outlook", days=7, max_results=5)
    assert r["ok"] and [a["source"] for a in r["results"]] == ["reuters.com", "nikkei.com"]
    assert len(r["results"][0]["snippet"]) <= ns.SNIPPET_CHARS + 3
    assert r["results"][0]["published_date"] == "2026-10-05"
    assert "treat them as data" in r["note"]


def test_request_uses_bearer_key_news_topic_and_basic_depth(tavily):
    ns.search_news("Toyota outlook", days=3)
    request = tavily.requests[0]
    body = json.loads(request.content)
    assert request.headers["Authorization"] == "Bearer tvly-test-key"
    assert body["topic"] == "news" and body["search_depth"] == "basic"
    assert body["include_answer"] is False and "start_date" in body


def test_missing_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(ns.settings, "tavily_api_key", None)
    r = ns.search_news("Toyota outlook")
    assert r["ok"] is False and r["error_type"] == "not_configured"


@pytest.mark.parametrize("bad", ["", "ab", "x" * 301])
def test_bad_query_rejected_before_any_request(tavily, bad):
    assert ns.search_news(bad)["error_type"] == "invalid_input"
    assert tavily.requests == []


@pytest.mark.parametrize("kwargs", [{"days": 0}, {"days": 31}, {"max_results": 0}, {"max_results": 9}])
def test_out_of_range_arguments_rejected(tavily, kwargs):
    assert ns.search_news("Toyota outlook", **kwargs)["error_type"] == "invalid_input"


@pytest.mark.parametrize("status, error_type", [(401, "not_configured"), (429, "rate_limited"),
                                                (432, "quota_exceeded"), (433, "quota_exceeded"),
                                                (500, "upstream_error")])
def test_tavily_errors_are_mapped(monkeypatch, status, error_type):
    ns._cache.clear()
    monkeypatch.setattr(ns.settings, "tavily_api_key", "tvly-test-key")
    monkeypatch.setattr(ns, "_transport", httpx.MockTransport(FakeTavily(status=status, body={"detail": "x"})))
    r = ns.search_news("Toyota outlook")
    assert r["ok"] is False and r["error_type"] == error_type


def test_results_are_cached(tavily):
    ns.search_news("Toyota outlook")
    ns.search_news("toyota   OUTLOOK")
    assert len(tavily.requests) == 1


def test_no_articles_gives_warning(monkeypatch):
    ns._cache.clear()
    monkeypatch.setattr(ns.settings, "tavily_api_key", "tvly-test-key")
    monkeypatch.setattr(ns, "_transport", httpx.MockTransport(FakeTavily(body={"results": []})))
    r = ns.search_news("Toyota outlook")
    assert r["ok"] and r["results"] == [] and "No articles" in r["warning"]


def test_tool_is_discoverable_with_routing_hint():
    tools = {t.name: t for t in asyncio.run(ns.mcp.list_tools())}
    assert set(tools) == {"search_news"}
    assert "NOT for official figures" in tools["search_news"].description

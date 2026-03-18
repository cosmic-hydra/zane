"""Tests for online resource reader utilities."""

from drug_discovery.web_scraping.scraper import OnlineResourceReader


class _FakeResponse:
    def __init__(self, text="", content=b"", headers=None):
        self.text = text
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None


def test_read_resource_html(monkeypatch):
    reader = OnlineResourceReader(max_chars=200)

    def _fake_get(url, timeout=0, headers=None):
        assert "example.org" in url
        return _FakeResponse(
            text="<html><body><h1>Route</h1><p>Useful synthesis details here.</p></body></html>",
            headers={"Content-Type": "text/html"},
        )

    monkeypatch.setattr("drug_discovery.web_scraping.scraper.requests.get", _fake_get)

    result = reader.read_resource("https://example.org/route")
    assert result["success"] is True
    assert result["resource_type"] == "html"
    assert "synthesis" in result["text"].lower()


def test_read_resource_pdf_path_dispatch(monkeypatch):
    reader = OnlineResourceReader()

    monkeypatch.setattr(
        reader,
        "read_pdf_url",
        lambda url: {"success": True, "url": url, "resource_type": "pdf", "text": "pdf content"},
    )

    result = reader.read_resource("https://example.org/paper.pdf")
    assert result["success"] is True
    assert result["resource_type"] == "pdf"


def test_enrich_search_hits_adds_preview(monkeypatch):
    reader = OnlineResourceReader()

    monkeypatch.setattr(
        reader,
        "read_resource",
        lambda url: {
            "success": True,
            "url": url,
            "resource_type": "html",
            "text": "Detailed route strategy and reagent discussion.",
        },
    )

    hits = [{"title": "A", "url": "https://example.org/a", "snippet": "", "source": "x"}]
    enriched = reader.enrich_search_hits(hits, max_reads=1)

    assert len(enriched) == 1
    assert enriched[0]["resource_type"] == "html"
    assert enriched[0]["resource_read_success"] == "true"
    assert "route strategy" in enriched[0]["resource_preview"].lower()

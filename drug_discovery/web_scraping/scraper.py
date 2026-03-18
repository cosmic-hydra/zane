"""
Biomedical Web Scraping and Data Ingestion
Collects data from PubMed, bioRxiv, clinical trials, patents
"""

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class PubMedAPI:
    """
    PubMed API client for scientific literature
    """

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: NCBI API key (optional, increases rate limits)
        """
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.34 if api_key else 1.0  # seconds

    def search(self, query: str, max_results: int = 100, date_from: str | None = None) -> list[str]:
        """
        Search PubMed for articles

        Args:
            query: Search query
            max_results: Maximum results
            date_from: Start date (YYYY/MM/DD)

        Returns:
            List of PubMed IDs
        """
        try:
            # Build search URL
            params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}

            if self.api_key:
                params["api_key"] = self.api_key

            if date_from:
                params["datetype"] = "pdat"
                params["mindate"] = date_from

            # Execute search
            search_url = f"{self.base_url}/esearch.fcgi"
            response = requests.get(search_url, params=params)
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            logger.info(f"Found {len(pmids)} articles for query: {query}")
            time.sleep(self.rate_limit_delay)

            return pmids

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """
        Fetch abstracts for PubMed IDs

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article dictionaries
        """
        articles = []

        try:
            # Fetch in batches to respect rate limits
            batch_size = 100

            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                pmid_str = ",".join(batch)

                params = {"db": "pubmed", "id": pmid_str, "retmode": "xml"}

                if self.api_key:
                    params["api_key"] = self.api_key

                fetch_url = f"{self.base_url}/efetch.fcgi"
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()

                # Parse XML (simplified - would use proper XML parsing)
                # Placeholder for actual parsing
                for pmid in batch:
                    article = {
                        "pmid": pmid,
                        "title": f"Article {pmid}",
                        "abstract": "Abstract text...",
                        "date": datetime.now().isoformat(),
                    }
                    articles.append(article)

                time.sleep(self.rate_limit_delay)

            logger.info(f"Fetched {len(articles)} abstracts")
            return articles

        except Exception as e:
            logger.error(f"Error fetching abstracts: {e}")
            return articles


class BiomedicalScraper:
    """
    Comprehensive biomedical data scraper
    """

    def __init__(self):
        self.pubmed_api = PubMedAPI()
        self.trusted_domains = ["nih.gov", "cdc.gov", "who.int", ".edu", "ebi.ac.uk", "ncbi.nlm.nih.gov"]

    def scrape_drug_research(self, keywords: list[str], max_per_keyword: int = 50, days_back: int = 30) -> list[dict]:
        """
        Scrape recent drug research literature

        Args:
            keywords: List of search keywords
            max_per_keyword: Max results per keyword
            days_back: How many days back to search

        Returns:
            List of articles
        """
        all_articles = []
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")

        for keyword in keywords:
            query = f"{keyword} AND (drug OR molecule OR compound)"

            # Search PubMed
            pmids = self.pubmed_api.search(query=query, max_results=max_per_keyword, date_from=date_from)

            # Fetch abstracts
            articles = self.pubmed_api.fetch_abstracts(pmids)
            all_articles.extend(articles)

        logger.info(f"Scraped {len(all_articles)} total articles")
        return all_articles

    def scrape_clinical_trials(self, condition: str, max_results: int = 100) -> list[dict]:
        """
        Scrape ClinicalTrials.gov data

        Args:
            condition: Medical condition
            max_results: Maximum results

        Returns:
            List of clinical trials
        """
        try:
            # ClinicalTrials.gov API
            base_url = "https://clinicaltrials.gov/api/query/study_fields"

            params = {
                "expr": condition,
                "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase",
                "max_rnk": max_results,
                "fmt": "json",
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            trials = data.get("StudyFieldsResponse", {}).get("StudyFields", [])

            logger.info(f"Scraped {len(trials)} clinical trials for {condition}")
            return trials

        except Exception as e:
            logger.error(f"Clinical trials scraping error: {e}")
            return []


class WebDataProcessor:
    """
    Process and clean web-scraped biomedical data
    """

    def __init__(self):
        pass

    def extract_molecules(self, text: str) -> list[str]:
        """
        Extract molecule names and identifiers from text

        Args:
            text: Input text

        Returns:
            List of molecule names
        """
        # Placeholder for NER (Named Entity Recognition)
        # Would use BioBERT, SciBERT, or similar
        molecules = []

        # Simple pattern matching (replace with actual NER)
        import re

        patterns = [r"\b[A-Z][a-z]+\s+(inhibitor|antagonist|agonist)\b", r"\b[A-Z]{2,}\d*\b"]  # Abbreviations

        for pattern in patterns:
            matches = re.findall(pattern, text)
            molecules.extend(matches)

        return list(set(molecules))

    def deduplicate(self, articles: list[dict], key: str = "pmid") -> list[dict]:
        """
        Remove duplicate articles

        Args:
            articles: List of articles
            key: Deduplication key

        Returns:
            Deduplicated articles
        """
        seen = set()
        unique = []

        for article in articles:
            if key in article:
                identifier = article[key]
                if identifier not in seen:
                    seen.add(identifier)
                    unique.append(article)

        logger.info(f"Deduplicated: {len(articles)} → {len(unique)}")
        return unique

    def filter_quality(self, articles: list[dict], min_abstract_length: int = 100) -> list[dict]:
        """
        Filter articles by quality criteria

        Args:
            articles: List of articles
            min_abstract_length: Minimum abstract length

        Returns:
            Filtered articles
        """
        filtered = []

        for article in articles:
            abstract = article.get("abstract", "")

            if len(abstract) >= min_abstract_length:
                filtered.append(article)

        logger.info(f"Quality filter: {len(articles)} → {len(filtered)}")
        return filtered


class InternetSearchClient:
    """
    Internet search client for synthesis research support.

    Supports Google Programmable Search (when configured) with a DuckDuckGo fallback.
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        google_cse_id: str | None = None,
        go_search_bin: str | None = None,
    ):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_CSE_API_KEY")
        self.google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID")
        self.go_search_bin = go_search_bin or os.getenv("ZANE_GO_SEARCH_BIN")

    def search_web(self, query: str, max_results: int = 5, prefer_google: bool = True) -> list[dict[str, str]]:
        if not query.strip():
            return []

        if prefer_google and self.google_api_key and self.google_cse_id:
            results = self._google_search(query=query, max_results=max_results)
            if results:
                return results

        go_results = self._go_fast_search(query=query, max_results=max_results)
        if go_results:
            return go_results

        return self._duckduckgo_search(query=query, max_results=max_results)

    def _go_fast_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        if not self.go_search_bin:
            return []

        try:
            command = [
                self.go_search_bin,
                "--query",
                query,
                "--max-results",
                str(max(1, max_results)),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            payload = json.loads(result.stdout)

            if not isinstance(payload, list):
                return []

            normalized: list[dict[str, str]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue

                normalized.append(
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "snippet": str(item.get("snippet", "")),
                        "source": str(item.get("source", "go-fastsearch")),
                    }
                )

            return normalized
        except Exception as exc:
            logger.warning(f"Go fast search backend unavailable, using Python fallback: {exc}")
            return []

    def _google_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.google_api_key,
                    "cx": self.google_cse_id,
                    "q": query,
                    "num": max(1, min(max_results, 10)),
                },
                timeout=15,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            items = data.get("items", [])

            results: list[dict[str, str]] = []
            for item in items:
                results.append(
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("link", "")),
                        "snippet": str(item.get("snippet", "")),
                        "source": "google-cse",
                    }
                )
            return results
        except Exception as exc:
            logger.warning(f"Google search failed, falling back to DuckDuckGo: {exc}")
            return []

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            html = response.text
            pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)
            matches = pattern.findall(html)

            results: list[dict[str, str]] = []
            for href, title in matches[: max(1, max_results)]:
                cleaned_title = re.sub(r"<[^>]+>", "", title)
                results.append(
                    {
                        "title": cleaned_title,
                        "url": href,
                        "snippet": "",
                        "source": "duckduckgo",
                    }
                )

            return results
        except Exception as exc:
            logger.error(f"DuckDuckGo search failed: {exc}")
            return []


class OnlineResourceReader:
    """Reads and summarizes online resources from URLs, including PDF links."""

    def __init__(self, timeout_seconds: int = 15, max_chars: int = 2000, max_pdf_pages: int = 5):
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars
        self.max_pdf_pages = max_pdf_pages

    def read_resource(self, url: str) -> dict[str, Any]:
        if not url.strip():
            return {"success": False, "url": url, "error": "empty url", "resource_type": "unknown", "text": ""}

        if self._looks_like_pdf(url):
            return self.read_pdf_url(url)

        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" in content_type:
                return self.read_pdf_url(url)

            text = self._extract_html_text(response.text)
            return {
                "success": True,
                "url": url,
                "resource_type": "html",
                "text": text[: self.max_chars],
            }
        except Exception as exc:
            return {
                "success": False,
                "url": url,
                "resource_type": "html",
                "error": str(exc),
                "text": "",
            }

    def read_pdf_url(self, url: str) -> dict[str, Any]:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            return {
                "success": False,
                "url": url,
                "resource_type": "pdf",
                "error": f"pypdf unavailable: {exc}",
                "text": "",
            }

        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            reader = PdfReader(BytesIO(response.content))
            page_count = min(len(reader.pages), max(self.max_pdf_pages, 1))

            chunks: list[str] = []
            for idx in range(page_count):
                extracted = reader.pages[idx].extract_text() or ""
                if extracted:
                    chunks.append(extracted)

            text = "\n".join(chunks)
            return {
                "success": True,
                "url": url,
                "resource_type": "pdf",
                "text": text[: self.max_chars],
            }
        except Exception as exc:
            return {
                "success": False,
                "url": url,
                "resource_type": "pdf",
                "error": str(exc),
                "text": "",
            }

    def enrich_search_hits(self, hits: list[dict[str, str]], max_reads: int = 3) -> list[dict[str, str]]:
        if not hits:
            return []

        enriched: list[dict[str, str]] = []
        for idx, hit in enumerate(hits):
            current = dict(hit)

            if idx < max(0, max_reads):
                resource = self.read_resource(current.get("url", ""))
                current["resource_type"] = str(resource.get("resource_type", "unknown"))
                current["resource_read_success"] = str(bool(resource.get("success", False))).lower()
                current["resource_preview"] = str(resource.get("text", ""))[:500]
                if not resource.get("success", False):
                    current["resource_error"] = str(resource.get("error", ""))

            enriched.append(current)

        return enriched

    @staticmethod
    def _looks_like_pdf(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.path.lower().endswith(".pdf")

    @staticmethod
    def _extract_html_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(" ", strip=True)
        except Exception:
            no_tags = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", no_tags).strip()


class AISynthesisChat:
    """AI chat helper for synthesis strategy generation."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_id = model_id

    def generate_synthesis_brief(
        self,
        smiles: str,
        target_protein: str | None = None,
        research_hits: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        from drug_discovery.ai_support import AISupportConfig, LlamaSupportAssistant

        context_lines = [f"Molecule SMILES: {smiles}"]
        if target_protein:
            context_lines.append(f"Target protein: {target_protein}")

        if research_hits:
            context_lines.append("Relevant web research:")
            for idx, hit in enumerate(research_hits[:5], 1):
                title = hit.get("title", "Unknown")
                url = hit.get("url", "")
                context_lines.append(f"{idx}. {title} ({url})")

        prompt = (
            "Provide a concise medicinal chemistry synthesis strategy with: "
            "(1) route ideas, (2) risk points, (3) building block suggestions, "
            "and (4) immediate next experiments."
        )

        assistant = LlamaSupportAssistant(config=AISupportConfig(model_id=self.model_id))
        response = assistant.respond(
            user_prompt=prompt,
            context="\n".join(context_lines),
            max_new_tokens=300,
            temperature=0.5,
            top_p=0.9,
        )

        return {"model_id": self.model_id, "brief": response}

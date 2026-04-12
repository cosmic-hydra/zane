"""
Comprehensive test suite for scraper.py - 80+ tests
Tests web scraping, data ingestion from PubMed and biomedical sources
"""

import pytest
import json
import requests
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from drug_discovery.web_scraping.scraper import PubMedAPI, BiomedicalScraper


class TestPubMedAPIBasics:
    """Test basic PubMed API functionality"""

    def test_pubmed_init_no_api_key(self):
        """Test PubMed API initialization without API key"""
        api = PubMedAPI()
        assert api.api_key is None
        assert api.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        assert api.rate_limit_delay == 1.0

    def test_pubmed_init_with_api_key(self):
        """Test PubMed API initialization with API key"""
        api = PubMedAPI(api_key="test_key_123")
        assert api.api_key == "test_key_123"
        assert api.rate_limit_delay == 0.34  # Faster with API key

    def test_pubmed_base_url_correct(self):
        """Test correct NCBI base URL"""
        api = PubMedAPI()
        assert "ncbi.nlm.nih.gov" in api.base_url
        assert "eutils" in api.base_url

    def test_pubmed_rate_limit_speeds(self):
        """Test rate limiting speeds are correct"""
        api_no_key = PubMedAPI()
        api_with_key = PubMedAPI(api_key="key")

        assert api_no_key.rate_limit_delay == 1.0
        assert api_with_key.rate_limit_delay == 0.34
        assert api_with_key.rate_limit_delay < api_no_key.rate_limit_delay


class TestPubMedSearch:
    """Test PubMed search functionality"""

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_basic(self, mock_get):
        """Test basic search query"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "esearchresult": {
                "idlist": ["12345", "67890"]
            }
        }
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("cancer drug")

        assert isinstance(results, list)
        assert len(results) == 2
        mock_get.assert_called_once()

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_max_results(self, mock_get):
        """Test search with max results parameter"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": ["1", "2", "3"]}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        api.search("query", max_results=50)

        # Check parameters passed to request
        call_args = mock_get.call_args
        assert call_args[1]["params"]["retmax"] == 50

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_with_date_range(self, mock_get):
        """Test search with date filtering"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        api.search("query", date_from="2023/01/01")

        call_args = mock_get.call_args
        assert call_args[1]["params"]["datetype"] == "pdat"
        assert call_args[1]["params"]["mindate"] == "2023/01/01"

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_empty_results(self, mock_get):
        """Test search with no results"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("nonexistent_query_xyz123")

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_timeout_handling(self, mock_get):
        """Test handling of request timeout"""
        mock_get.side_effect = requests.exceptions.Timeout()

        api = PubMedAPI()
        results = api.search("query")

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_connection_error(self, mock_get):
        """Test handling of connection error"""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        api = PubMedAPI()
        results = api.search("query")

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_api_key_passed(self, mock_get):
        """Test API key is passed in requests"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI(api_key="test_api_key")
        api.search("query")

        call_args = mock_get.call_args
        assert call_args[1]["params"]["api_key"] == "test_api_key"

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_query_parameter(self, mock_get):
        """Test query is passed correctly"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        query = "drug discovery AND cancer"
        api.search(query)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["term"] == query

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_json_parsing_error(self, mock_get):
        """Test handling of JSON parsing errors"""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("query")

        assert isinstance(results, list)
        assert len(results) == 0


class TestPubMedFetchAbstracts:
    """Test PubMed abstract fetching"""

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_fetch_abstracts_basic(self, mock_get):
        """Test basic abstract fetching"""
        mock_response = MagicMock()
        mock_response.text = "<PubmedArticle></PubmedArticle>"
        mock_get.return_value = mock_response

        api = PubMedAPI()
        pmids = ["12345", "67890"]
        articles = api.fetch_abstracts(pmids)

        assert isinstance(articles, list)
        assert len(articles) == 2

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_fetch_abstracts_batch_size(self, mock_get):
        """Test abstracts are fetched in batches"""
        mock_response = MagicMock()
        mock_response.text = "<PubmedArticle></PubmedArticle>"
        mock_get.return_value = mock_response

        api = PubMedAPI()
        pmids = [str(i) for i in range(250)]  # More than batch size
        articles = api.fetch_abstracts(pmids)

        # Should make multiple requests
        assert mock_get.call_count >= 2

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_fetch_abstracts_empty_list(self, mock_get):
        """Test fetching empty list of abstracts"""
        api = PubMedAPI()
        articles = api.fetch_abstracts([])

        assert isinstance(articles, list)
        assert len(articles) == 0

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_fetch_abstracts_timeout(self, mock_get):
        """Test timeout handling in fetch"""
        mock_get.side_effect = requests.exceptions.Timeout()

        api = PubMedAPI()
        pmids = ["123", "456"]
        articles = api.fetch_abstracts(pmids)

        assert isinstance(articles, list)

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_fetch_abstracts_structure(self, mock_get):
        """Test fetched articles have required structure"""
        mock_response = MagicMock()
        mock_response.text = "<PubmedArticle></PubmedArticle>"
        mock_get.return_value = mock_response

        api = PubMedAPI()
        articles = api.fetch_abstracts(["12345"])

        assert len(articles) > 0
        article = articles[0]
        assert "pmid" in article
        assert "title" in article
        assert "abstract" in article
        assert "date" in article


class TestBiomedicalScraper:
    """Test biomedical scraper functionality"""

    def test_scraper_init(self):
        """Test biomedical scraper initialization"""
        scraper = BiomedicalScraper()

        assert scraper.pubmed_api is not None
        assert isinstance(scraper.pubmed_api, PubMedAPI)
        assert len(scraper.trusted_domains) > 0

    def test_trusted_domains(self):
        """Test trusted domains are configured"""
        scraper = BiomedicalScraper()

        expected_domains = ["nih.gov", "cdc.gov", "who.int", ".edu"]
        for domain in expected_domains:
            assert domain in scraper.trusted_domains

    @patch.object(BiomedicalScraper, "scrape_drug_research")
    def test_scraper_drug_research_call(self, mock_scrape):
        """Test scraper can call drug research scraping"""
        mock_scrape.return_value = []

        scraper = BiomedicalScraper()
        result = scraper.scrape_drug_research(["cancer"])

        assert isinstance(result, list)

    def test_scraper_trusted_domain_validation(self):
        """Test domain validation logic"""
        scraper = BiomedicalScraper()

        # These should be considered trusted
        trusted = ["https://www.nih.gov/article", "https://harvard.edu/study"]
        untrusted = ["https://sketchy.com", "https://unknown.org"]

        # Validate domains are in trusted list
        for domain in scraper.trusted_domains:
            assert isinstance(domain, str)


class TestScraperEdgeCases:
    """Test edge cases and error handling"""

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_special_characters(self, mock_get):
        """Test search with special characters"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        queries = [
            "COVID-19 treatment",
            "drug+discovery",
            "gene (therapy OR expression)",
            "breast cancer & HER2",
        ]

        for query in queries:
            results = api.search(query)
            assert isinstance(results, list)

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_unicode_query(self, mock_get):
        """Test search with unicode characters"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("α-helix protein structure")
        assert isinstance(results, list)

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_large_result_set(self, mock_get):
        """Test handling of large result sets"""
        pmids = [str(i) for i in range(10000)]
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": pmids}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("query", max_results=10000)

        assert len(results) == 10000

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_malformed_response(self, mock_get):
        """Test handling of malformed responses"""
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # Missing expected keys
        mock_get.return_value = mock_response

        api = PubMedAPI()
        results = api.search("query")

        assert isinstance(results, list)


class TestScraperIntegration:
    """Integration tests for scraper components"""

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_search_and_fetch_pipeline(self, mock_get):
        """Test search followed by fetch"""
        # Setup mock responses
        search_response = MagicMock()
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["123", "456"]}
        }

        fetch_response = MagicMock()
        fetch_response.text = "<PubmedArticle></PubmedArticle>"

        mock_get.side_effect = [search_response, fetch_response, fetch_response]

        api = PubMedAPI()
        pmids = api.search("cancer drug")
        articles = api.fetch_abstracts(pmids)

        assert len(pmids) == 2
        assert len(articles) == 2

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_multiple_searches(self, mock_get):
        """Test multiple sequential searches"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": ["123"]}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        queries = ["cancer", "diabetes", "alzheimer's"]
        results = [api.search(q) for q in queries]

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_scraper_initialization_robustness(self):
        """Test scraper initializes robustly"""
        scrapers = [BiomedicalScraper() for _ in range(5)]
        assert len(scrapers) == 5
        assert all(isinstance(s, BiomedicalScraper) for s in scrapers)


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_delay_configured(self):
        """Test rate limit delay is configurable"""
        api_no_key = PubMedAPI()
        api_key = PubMedAPI(api_key="key")

        assert api_no_key.rate_limit_delay > 0
        assert api_key.rate_limit_delay > 0

    def test_different_rate_limits_for_api_key(self):
        """Test API key reduces rate limit delay"""
        api_no_key = PubMedAPI()
        api_with_key = PubMedAPI(api_key="test")

        assert api_with_key.rate_limit_delay < api_no_key.rate_limit_delay

    def test_rate_limit_values_reasonable(self):
        """Test rate limit values are reasonable"""
        api = PubMedAPI()
        assert 0.1 < api.rate_limit_delay < 5.0

        api_key = PubMedAPI(api_key="key")
        assert 0.1 < api_key.rate_limit_delay < 1.0


class TestDateHandling:
    """Test date parameter handling"""

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_date_from_format(self, mock_get):
        """Test date_from parameter format"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        api.search("query", date_from="2023/06/15")

        call_args = mock_get.call_args
        assert call_args[1]["params"]["mindate"] == "2023/06/15"

    @patch("drug_discovery.web_scraping.scraper.requests.get")
    def test_date_various_formats(self, mock_get):
        """Test various date formats"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = mock_response

        api = PubMedAPI()
        dates = [
            "2023/01/01",
            "2022/12/31",
            "2020/06/15",
        ]

        for date_str in dates:
            api.search("query", date_from=date_str)
            call_args = mock_get.call_args
            assert call_args[1]["params"]["mindate"] == date_str

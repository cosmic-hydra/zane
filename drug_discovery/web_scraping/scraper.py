"""
Biomedical Web Scraping and Data Ingestion
Collects data from PubMed, bioRxiv, clinical trials, patents
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class PubMedAPI:
    """
    PubMed API client for scientific literature
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: NCBI API key (optional, increases rate limits)
        """
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.34 if api_key else 1.0  # seconds

    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None
    ) -> List[str]:
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
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }

            if self.api_key:
                params['api_key'] = self.api_key

            if date_from:
                params['datetype'] = 'pdat'
                params['mindate'] = date_from

            # Execute search
            search_url = f"{self.base_url}/esearch.fcgi"
            response = requests.get(search_url, params=params)
            response.raise_for_status()

            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])

            logger.info(f"Found {len(pmids)} articles for query: {query}")
            time.sleep(self.rate_limit_delay)

            return pmids

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def fetch_abstracts(
        self,
        pmids: List[str]
    ) -> List[Dict]:
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
                batch = pmids[i:i+batch_size]
                pmid_str = ','.join(batch)

                params = {
                    'db': 'pubmed',
                    'id': pmid_str,
                    'retmode': 'xml'
                }

                if self.api_key:
                    params['api_key'] = self.api_key

                fetch_url = f"{self.base_url}/efetch.fcgi"
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()

                # Parse XML (simplified - would use proper XML parsing)
                # Placeholder for actual parsing
                for pmid in batch:
                    article = {
                        'pmid': pmid,
                        'title': f'Article {pmid}',
                        'abstract': 'Abstract text...',
                        'date': datetime.now().isoformat()
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
        self.trusted_domains = [
            'nih.gov', 'cdc.gov', 'who.int', '.edu',
            'ebi.ac.uk', 'ncbi.nlm.nih.gov'
        ]

    def scrape_drug_research(
        self,
        keywords: List[str],
        max_per_keyword: int = 50,
        days_back: int = 30
    ) -> List[Dict]:
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
        date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')

        for keyword in keywords:
            query = f"{keyword} AND (drug OR molecule OR compound)"

            # Search PubMed
            pmids = self.pubmed_api.search(
                query=query,
                max_results=max_per_keyword,
                date_from=date_from
            )

            # Fetch abstracts
            articles = self.pubmed_api.fetch_abstracts(pmids)
            all_articles.extend(articles)

        logger.info(f"Scraped {len(all_articles)} total articles")
        return all_articles

    def scrape_clinical_trials(
        self,
        condition: str,
        max_results: int = 100
    ) -> List[Dict]:
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
                'expr': condition,
                'fields': 'NCTId,BriefTitle,Condition,InterventionName,Phase',
                'max_rnk': max_results,
                'fmt': 'json'
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            trials = data.get('StudyFieldsResponse', {}).get('StudyFields', [])

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

    def extract_molecules(
        self,
        text: str
    ) -> List[str]:
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
        patterns = [
            r'\b[A-Z][a-z]+\s+(inhibitor|antagonist|agonist)\b',
            r'\b[A-Z]{2,}\d*\b'  # Abbreviations
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            molecules.extend(matches)

        return list(set(molecules))

    def deduplicate(
        self,
        articles: List[Dict],
        key: str = 'pmid'
    ) -> List[Dict]:
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

    def filter_quality(
        self,
        articles: List[Dict],
        min_abstract_length: int = 100
    ) -> List[Dict]:
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
            abstract = article.get('abstract', '')

            if len(abstract) >= min_abstract_length:
                filtered.append(article)

        logger.info(f"Quality filter: {len(articles)} → {len(filtered)}")
        return filtered

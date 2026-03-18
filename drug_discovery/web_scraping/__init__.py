"""
Web-Scale Data Ingestion Pipeline
Scrapes and processes biomedical literature and databases
"""

from .scraper import (
    AISynthesisChat,
    BiomedicalScraper,
    InternetSearchClient,
    OnlineResourceReader,
    PubMedAPI,
    WebDataProcessor,
)

__all__ = [
    "BiomedicalScraper",
    "PubMedAPI",
    "WebDataProcessor",
    "InternetSearchClient",
    "OnlineResourceReader",
    "AISynthesisChat",
]

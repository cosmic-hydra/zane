"""
Biomedical Intelligence Layer - Web-Scale Literature Mining

Ingests and processes biomedical literature from:
- PubMed/PMC (biomedical research)
- arXiv (preprints)
- bioRxiv/medRxiv (biomedical preprints)
- Patent databases (drug patents)
- Clinical trial literature

Performs:
- Named Entity Recognition (NER) for drugs, proteins, diseases
- Relationship extraction
- Knowledge graph construction
- Literature summarization
- Citation analysis
"""

import logging
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class LiteratureDocument:
    """Represents a scientific document."""
    doc_id: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    source: str  # 'pubmed', 'arxiv', 'biorxiv', 'patent'
    doi: Optional[str]
    pmid: Optional[str]
    citations: List[str]
    keywords: List[str]
    full_text: Optional[str] = None


@dataclass
class ExtractedEntity:
    """Extracted named entity from text."""
    entity_type: str  # 'drug', 'protein', 'disease', 'gene'
    entity_text: str
    entity_id: Optional[str]  # External ID (e.g., ChEMBL, UniProt)
    confidence: float
    context: str  # Surrounding text


@dataclass
class ExtractedRelationship:
    """Extracted relationship between entities."""
    subject: ExtractedEntity
    predicate: str  # 'treats', 'inhibits', 'binds', 'causes'
    object: ExtractedEntity
    confidence: float
    source_doc: str


class BiomedicalNER:
    """Named Entity Recognition for biomedical text."""

    def __init__(self):
        """Initialize NER system."""
        # Placeholder patterns - in production, use spaCy, BioBERT, or similar
        self.drug_patterns = [
            r'\b\w+mab\b',  # Monoclonal antibodies
            r'\b\w+tide\b',  # Peptides
            r'\b\w+ine\b',  # Common drug suffix
            r'\b\w+ol\b',  # Alcohols
        ]

        self.protein_patterns = [
            r'\b[A-Z]{2,}[0-9]+\b',  # Gene symbols
            r'\bp[0-9]{2,}\b',  # p53, p21, etc.
            r'\b\w+ receptor\b',
            r'\b\w+ kinase\b',
        ]

        self.disease_patterns = [
            r'\b\w+ cancer\b',
            r'\b\w+ disease\b',
            r'\b\w+ syndrome\b',
            r'\b\w+ disorder\b',
        ]

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[ExtractedEntity]:
        """
        Extract named entities from text.

        Args:
            text: Input text
            entity_types: Types to extract (None = all)

        Returns:
            List of extracted entities
        """
        if entity_types is None:
            entity_types = ['drug', 'protein', 'disease']

        entities = []

        # Drug entities
        if 'drug' in entity_types:
            for pattern in self.drug_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        entity_type='drug',
                        entity_text=match.group(0),
                        entity_id=None,
                        confidence=0.7,
                        context=text[max(0, match.start() - 50):min(len(text), match.end() + 50)],
                    )
                    entities.append(entity)

        # Protein entities
        if 'protein' in entity_types:
            for pattern in self.protein_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        entity_type='protein',
                        entity_text=match.group(0),
                        entity_id=None,
                        confidence=0.6,
                        context=text[max(0, match.start() - 50):min(len(text), match.end() + 50)],
                    )
                    entities.append(entity)

        # Disease entities
        if 'disease' in entity_types:
            for pattern in self.disease_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        entity_type='disease',
                        entity_text=match.group(0),
                        entity_id=None,
                        confidence=0.7,
                        context=text[max(0, match.start() - 50):min(len(text), match.end() + 50)],
                    )
                    entities.append(entity)

        logger.debug(f"Extracted {len(entities)} entities from text")

        return entities


class RelationshipExtractor:
    """Extract relationships between biomedical entities."""

    def __init__(self):
        """Initialize relationship extractor."""
        # Relationship patterns
        self.relation_patterns = {
            'treats': [
                r'(\w+) treats (\w+)',
                r'(\w+) for (\w+ (?:disease|cancer|syndrome))',
                r'treatment of (\w+ (?:disease|cancer)) with (\w+)',
            ],
            'inhibits': [
                r'(\w+) inhibits (\w+)',
                r'inhibition of (\w+) by (\w+)',
                r'(\w+) inhibitor',
            ],
            'binds': [
                r'(\w+) binds (\w+)',
                r'binding of (\w+) to (\w+)',
            ],
            'causes': [
                r'(\w+) causes (\w+)',
                r'(\w+) induces (\w+)',
            ],
        }

    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        source_doc: str,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships from text given known entities.

        Args:
            text: Input text
            entities: Known entities in text
            source_doc: Source document ID

        Returns:
            List of extracted relationships
        """
        relationships = []

        # Create entity map by text
        entity_map = {e.entity_text.lower(): e for e in entities}

        # Pattern-based extraction
        for predicate, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        subj_text = groups[0].lower()
                        obj_text = groups[1].lower()

                        # Find matching entities
                        subj_entity = entity_map.get(subj_text)
                        obj_entity = entity_map.get(obj_text)

                        if subj_entity and obj_entity:
                            relationship = ExtractedRelationship(
                                subject=subj_entity,
                                predicate=predicate,
                                object=obj_entity,
                                confidence=0.6,
                                source_doc=source_doc,
                            )
                            relationships.append(relationship)

        logger.debug(f"Extracted {len(relationships)} relationships from text")

        return relationships


class PubMedIngester:
    """Ingest literature from PubMed."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PubMed ingester.

        Args:
            api_key: NCBI API key (optional but recommended)
        """
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 100,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[str]:
        """
        Search PubMed for articles.

        Args:
            query: Search query
            max_results: Maximum results to return
            date_range: Optional (start_date, end_date) tuple

        Returns:
            List of PMIDs
        """
        # Placeholder - would use actual PubMed API
        logger.info(f"Searching PubMed for: {query}")

        # Simulate PMID results
        pmids = [f"PMID{i:08d}" for i in range(1, min(max_results + 1, 101))]

        return pmids

    async def fetch_article(self, pmid: str) -> Optional[LiteratureDocument]:
        """
        Fetch article metadata from PubMed.

        Args:
            pmid: PubMed ID

        Returns:
            LiteratureDocument or None
        """
        # Placeholder - would use actual PubMed API
        logger.debug(f"Fetching article: {pmid}")

        # Simulate article data
        doc = LiteratureDocument(
            doc_id=pmid,
            title=f"Article {pmid}",
            abstract=f"This is the abstract for article {pmid}. It discusses various biomedical topics.",
            authors=["Author A", "Author B"],
            publication_date=datetime.now().isoformat(),
            source="pubmed",
            doi=f"10.1234/{pmid}",
            pmid=pmid,
            citations=[],
            keywords=["drug discovery", "AI"],
        )

        return doc

    async def batch_fetch_articles(
        self,
        pmids: List[str],
        max_concurrent: int = 10,
    ) -> List[LiteratureDocument]:
        """
        Fetch multiple articles in parallel.

        Args:
            pmids: List of PMIDs
            max_concurrent: Maximum concurrent requests

        Returns:
            List of documents
        """
        # Placeholder for async batch fetching
        documents = []
        for pmid in pmids[:max_concurrent]:
            doc = await self.fetch_article(pmid)
            if doc:
                documents.append(doc)

        logger.info(f"Fetched {len(documents)} articles from PubMed")

        return documents


class ArXivIngester:
    """Ingest preprints from arXiv."""

    def __init__(self):
        """Initialize arXiv ingester."""
        self.base_url = "http://export.arxiv.org/api/query"

    async def search_arxiv(
        self,
        query: str,
        category: str = "q-bio",
        max_results: int = 100,
    ) -> List[LiteratureDocument]:
        """
        Search arXiv for preprints.

        Args:
            query: Search query
            category: arXiv category (e.g., 'q-bio', 'cs.LG')
            max_results: Maximum results

        Returns:
            List of documents
        """
        # Placeholder - would use actual arXiv API
        logger.info(f"Searching arXiv ({category}): {query}")

        documents = []
        for i in range(min(max_results, 50)):
            doc = LiteratureDocument(
                doc_id=f"arXiv:{i:04d}",
                title=f"Preprint {i}",
                abstract=f"Abstract for preprint {i}",
                authors=["Researcher X"],
                publication_date=datetime.now().isoformat(),
                source="arxiv",
                doi=None,
                pmid=None,
                citations=[],
                keywords=[category],
            )
            documents.append(doc)

        return documents


class BiomedicalIntelligence:
    """Comprehensive biomedical intelligence system."""

    def __init__(
        self,
        enable_pubmed: bool = True,
        enable_arxiv: bool = True,
        pubmed_api_key: Optional[str] = None,
    ):
        """
        Initialize biomedical intelligence system.

        Args:
            enable_pubmed: Enable PubMed ingestion
            enable_arxiv: Enable arXiv ingestion
            pubmed_api_key: PubMed API key
        """
        self.ner = BiomedicalNER()
        self.relation_extractor = RelationshipExtractor()

        self.ingesters = {}
        if enable_pubmed:
            self.ingesters['pubmed'] = PubMedIngester(api_key=pubmed_api_key)
        if enable_arxiv:
            self.ingesters['arxiv'] = ArXivIngester()

        # Storage
        self.documents: Dict[str, LiteratureDocument] = {}
        self.entities: Dict[str, List[ExtractedEntity]] = defaultdict(list)
        self.relationships: List[ExtractedRelationship] = []

    async def ingest_literature(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 100,
    ) -> Dict[str, int]:
        """
        Ingest literature from multiple sources.

        Args:
            query: Search query
            sources: List of sources to query (None = all)
            max_results_per_source: Max results per source

        Returns:
            Dictionary with ingestion statistics
        """
        if sources is None:
            sources = list(self.ingesters.keys())

        stats = {}

        for source in sources:
            if source not in self.ingesters:
                logger.warning(f"Unknown source: {source}")
                continue

            ingester = self.ingesters[source]

            try:
                if source == 'pubmed':
                    pmids = await ingester.search_pubmed(query, max_results=max_results_per_source)
                    documents = await ingester.batch_fetch_articles(pmids)
                elif source == 'arxiv':
                    documents = await ingester.search_arxiv(query, max_results=max_results_per_source)
                else:
                    documents = []

                # Store documents
                for doc in documents:
                    self.documents[doc.doc_id] = doc

                stats[source] = len(documents)

            except Exception as e:
                logger.error(f"Ingestion from {source} failed: {e}")
                stats[source] = 0

        logger.info(f"Ingested literature: {stats}")

        return stats

    def process_documents(
        self,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Process documents with NER and relationship extraction.

        Args:
            doc_ids: Document IDs to process (None = all)

        Returns:
            Processing statistics
        """
        if doc_ids is None:
            doc_ids = list(self.documents.keys())

        total_entities = 0
        total_relationships = 0

        for doc_id in doc_ids:
            if doc_id not in self.documents:
                continue

            doc = self.documents[doc_id]

            # Extract entities
            text = f"{doc.title} {doc.abstract}"
            entities = self.ner.extract_entities(text)

            # Store entities
            self.entities[doc_id] = entities
            total_entities += len(entities)

            # Extract relationships
            relationships = self.relation_extractor.extract_relationships(
                text,
                entities,
                doc_id,
            )

            self.relationships.extend(relationships)
            total_relationships += len(relationships)

        stats = {
            "processed_documents": len(doc_ids),
            "total_entities": total_entities,
            "total_relationships": total_relationships,
        }

        logger.info(f"Document processing complete: {stats}")

        return stats

    def build_knowledge_graph(self) -> pd.DataFrame:
        """
        Build knowledge graph from extracted relationships.

        Returns:
            DataFrame with graph edges
        """
        edges = []

        for rel in self.relationships:
            edge = {
                "subject": rel.subject.entity_text,
                "subject_type": rel.subject.entity_type,
                "predicate": rel.predicate,
                "object": rel.object.entity_text,
                "object_type": rel.object.entity_type,
                "confidence": rel.confidence,
                "source": rel.source_doc,
            }
            edges.append(edge)

        df = pd.DataFrame(edges)
        logger.info(f"Built knowledge graph with {len(df)} edges")

        return df

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        all_entities = [e for entities in self.entities.values() for e in entities]

        entity_counts = Counter([e.entity_type for e in all_entities])

        stats = {
            "total_entities": len(all_entities),
            "unique_entities": len(set(e.entity_text.lower() for e in all_entities)),
            "entity_type_counts": dict(entity_counts),
            "avg_confidence": np.mean([e.confidence for e in all_entities]) if all_entities else 0,
        }

        return stats

    def find_drug_disease_associations(
        self,
        min_confidence: float = 0.5,
    ) -> pd.DataFrame:
        """
        Find drug-disease associations from literature.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            DataFrame with drug-disease pairs
        """
        associations = []

        for rel in self.relationships:
            if rel.confidence < min_confidence:
                continue

            # Look for drug-disease relationships
            if rel.predicate == 'treats':
                if rel.subject.entity_type == 'drug' and rel.object.entity_type == 'disease':
                    associations.append({
                        "drug": rel.subject.entity_text,
                        "disease": rel.object.entity_text,
                        "confidence": rel.confidence,
                        "source": rel.source_doc,
                    })

        df = pd.DataFrame(associations)
        logger.info(f"Found {len(df)} drug-disease associations")

        return df

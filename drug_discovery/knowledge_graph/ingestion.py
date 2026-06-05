"""
Parallelized KG Ingestion

Handles terabyte-scale data ingestion into Neo4j using multiprocessing.
"""

import logging
import multiprocessing
import time

import pandas as pd

from .neo4j_adapter import Neo4jAdapter

logger = logging.getLogger(__name__)


def ingest_batch(batch_data: list[dict[str, any]], node_type: str):
    """Worker function to ingest a batch of nodes."""
    adapter = Neo4jAdapter()
    try:
        for item in batch_data:
            node_id = item.pop("id", None)
            if node_id:
                adapter.create_node(node_type, node_id, item)
    finally:
        adapter.close()


class KGIngestor:
    """Orchestrates large-scale ingestion of data into the knowledge graph."""

    def __init__(self, batch_size: int = 1000, num_workers: int | None = None):
        self.batch_size = batch_size
        self.num_workers = num_workers or multiprocessing.cpu_count()

    def parallel_ingest_nodes(self, df: pd.DataFrame, node_type: str):
        """
        Ingest nodes from a DataFrame in parallel.

        Args:
            df: DataFrame containing node data (must have an 'id' column)
            node_type: Label for the nodes
        """
        records = df.to_dict("records")
        batches = [records[i : i + self.batch_size] for i in range(0, len(records), self.batch_size)]

        logger.info(f"Starting parallel ingestion of {len(records)} {node_type} nodes using {self.num_workers} workers")

        start_time = time.time()
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.starmap(ingest_batch, [(batch, node_type) for batch in batches])

        end_time = time.time()
        logger.info(f"Finished ingestion in {end_time - start_time:.2f} seconds")

    def bulk_csv_import(self, csv_path: str, node_type: str):
        """
        Use Neo4j's LOAD CSV for even faster bulk import if applicable.
        Requires CSV to be accessible by Neo4j server.
        """
        adapter = Neo4jAdapter()
        query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
        MERGE (n:{node_type} {{id: row.id}})
        SET n += row
        """
        adapter.execute_query(query)
        adapter.close()

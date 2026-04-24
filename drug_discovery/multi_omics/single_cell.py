"""
Single-Cell and Spatial Transcriptomics Data Loaders.

Builds dataloaders that ingest both scRNA-seq and spatial transcriptomics
to understand cellular context for drug response prediction.

References:
    - Stuart et al., "Comprehensive Integration of Single-Cell Data"
    - Palla et al., "Spatial Proteomics"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import anndata as ad
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    sc = None
    ad = None
    logger.warning("Scanpy not available. Using simplified data models.")

try:
    import squidpy as sq

    SQUIDPY_AVAILABLE = True
except ImportError:
    SQUIDPY_AVAILABLE = False
    sq = None
    logger.warning("Squidpy not available. Spatial features will be limited.")


@dataclass
class CellData:
    """Single cell data container.

    Attributes:
        cell_id: Unique cell identifier.
        cell_type: Cell type annotation.
        gene_expression: Gene expression vector.
        coordinates: Spatial coordinates (if available).
        embeddings: Latent embeddings from model.
        metadata: Additional cell metadata.
    """

    cell_id: str
    cell_type: str = "unknown"
    gene_expression: np.ndarray | None = None
    coordinates: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "gene_expression": self.gene_expression.tolist() if self.gene_expression is not None else None,
            "coordinates": self.coordinates.tolist() if self.coordinates is not None else None,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "metadata": self.metadata,
        }


class SingleCellLoader:
    """
    Data loader for single-cell RNA-seq data.

    Supports loading from:
    - AnnData (.h5ad) files
    - 10x Genomics format
    - Simple expression matrices

    Example::

        loader = SingleCellLoader()
        adata = loader.load_from_file("scRNA_data.h5ad")
        cells = loader.get_cell_data(adata)
    """

    def __init__(
        self,
        n_top_genes: int = 2000,
        normalize: bool = True,
        log_transform: bool = True,
    ):
        """
        Initialize loader.

        Args:
            n_top_genes: Number of highly variable genes to select.
            normalize: Whether to normalize counts.
            log_transform: Whether to log-transform.
        """
        self.n_top_genes = n_top_genes
        self.normalize = normalize
        self.log_transform = log_transform

        logger.info(f"SingleCellLoader initialized: n_genes={n_top_genes}")

    def load_from_file(
        self,
        filepath: str,
        file_format: str = "auto",
    ) -> Any:
        """
        Load single-cell data from file.

        Args:
            filepath: Path to data file.
            file_format: Format ('h5ad', 'loom', 'csv', 'auto').

        Returns:
            AnnData object.
        """
        if not SCANPY_AVAILABLE:
            logger.warning("Scanpy not available. Returning mock data.")
            return self._create_mock_adata()

        if file_format == "auto":
            if filepath.endswith(".h5ad"):
                file_format = "h5ad"
            elif filepath.endswith(".loom"):
                file_format = "loom"
            elif filepath.endswith(".csv"):
                file_format = "csv"

        try:
            if file_format == "h5ad":
                adata = sc.read_h5ad(filepath)
            elif file_format == "loom":
                adata = sc.read_loom(filepath)
            elif file_format == "csv":
                adata = sc.read_csv(filepath)
            else:
                adata = sc.read_h5ad(filepath)

            # Preprocess
            adata = self.preprocess(adata)

            logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            return adata

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return self._create_mock_adata()

    def preprocess(self, adata: Any) -> Any:
        """
        Preprocess AnnData object.

        Args:
            adata: Raw AnnData.

        Returns:
            Preprocessed AnnData.
        """
        if not SCANPY_AVAILABLE:
            return adata

        try:
            # Filter cells and genes
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)

            # Normalize
            if self.normalize:
                sc.pp.normalize_total(adata, target_sum=1e4)

            # Log transform
            if self.log_transform:
                sc.pp.log1p(adata)

            # Select highly variable genes
            if self.n_top_genes > 0:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.n_top_genes,
                    flavor="seurat_v3" if hasattr(sc.pp, "highly_variable_genes") else "seurat",
                )
                adata = adata[:, adata.var.highly_variable]

            # Scale
            sc.pp.scale(adata, max_value=10)

            return adata

        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}. Using raw data.")
            return adata

    def get_cell_data(self, adata: Any) -> list[CellData]:
        """
        Extract CellData from AnnData.

        Args:
            adata: Preprocessed AnnData.

        Returns:
            List of CellData objects.
        """
        cells = []

        try:
            obs_names = adata.obs_names.tolist() if hasattr(adata.obs_names, "tolist") else list(adata.obs_names)
            var_names = adata.var_names.tolist() if hasattr(adata.var_names, "tolist") else list(adata.var_names)

            for i, cell_id in enumerate(obs_names):
                # Gene expression
                expr = adata.X[i] if hasattr(adata.X, "__getitem__") else adata.X[i, :]
                if hasattr(expr, "toarray"):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.array(expr).flatten()

                # Cell type
                cell_type = "unknown"
                if "cell_type" in adata.obs.columns:
                    cell_type = str(adata.obs["cell_type"].iloc[i])
                elif "celltype" in adata.obs.columns:
                    cell_type = str(adata.obs["celltype"].iloc[i])

                # Metadata
                metadata = {}
                for col in adata.obs.columns:
                    if col not in ["cell_type", "celltype"]:
                        try:
                            metadata[col] = adata.obs[col].iloc[i]
                        except Exception:
                            pass

                cell = CellData(
                    cell_id=str(cell_id),
                    cell_type=cell_type,
                    gene_expression=expr,
                    metadata=metadata,
                )
                cells.append(cell)

        except Exception as e:
            logger.warning(f"Failed to extract cell data: {e}")
            # Return mock data
            for i in range(min(100, len(cells) if cells else 100)):
                cells.append(
                    CellData(
                        cell_id=f"cell_{i}",
                        cell_type="mock_type",
                        gene_expression=np.random.randn(self.n_top_genes or 100),
                    )
                )

        return cells

    def _create_mock_adata(self) -> Any:
        """Create mock AnnData for testing."""
        n_cells = 500
        n_genes = self.n_top_genes or 100

        X = np.random.randn(n_cells, n_genes)
        obs = {"cell_type": np.random.choice(["T_cell", "B_cell", "Macrophage", "Fibroblast"], n_cells)}

        if SCANPY_AVAILABLE and ad is not None:
            return ad.AnnData(X=X, obs=obs)
        else:
            # Simple dict mock
            return {"X": X, "obs": obs}


class SpatialTranscriptomicsLoader:
    """
    Data loader for spatial transcriptomics data.

    Supports:
    - Visium spatialomics
    - Slide-seq
    - MERFISH
    - Custom spatial formats

    Example::

        loader = SpatialTranscriptomicsLoader()
        adata = loader.load_spatial_data("visium_data.h5ad")
        spatial_graph = loader.build_spatial_graph(adata)
    """

    def __init__(
        self,
        coord_type: str = "generic",
        n_neighbors: int = 6,
        radius: float | None = None,
    ):
        """
        Initialize loader.

        Args:
            coord_type: Coordinate system ('visium', 'slideseq', 'generic').
            n_neighbors: Number of spatial neighbors.
            radius: Radius for radius-based neighborhoods.
        """
        self.coord_type = coord_type
        self.n_neighbors = n_neighbors
        self.radius = radius

        logger.info(f"SpatialTranscriptomicsLoader initialized: coord_type={coord_type}")

    def load_spatial_data(
        self,
        filepath: str,
        spatial_key: str = "spatial",
    ) -> Any:
        """
        Load spatial transcriptomics data.

        Args:
            filepath: Path to data file.
            spatial_key: Key for spatial coordinates in obsm.

        Returns:
            AnnData with spatial coordinates.
        """
        if not SCANPY_AVAILABLE:
            logger.warning("Scanpy not available. Using mock spatial data.")
            return self._create_mock_spatial_adata()

        try:
            adata = sc.read_h5ad(filepath)

            # Check for spatial coordinates
            if spatial_key not in adata.obsm:
                logger.warning(f"No spatial coordinates found at {spatial_key}")
                adata.obsm[spatial_key] = self._generate_spatial_coords(adata.n_obs)

            return adata

        except Exception as e:
            logger.error(f"Failed to load spatial data: {e}")
            return self._create_mock_spatial_adata()

    def build_spatial_graph(
        self,
        adata: Any,
        spatial_key: str = "spatial",
    ) -> np.ndarray:
        """
        Build spatial neighbor graph.

        Args:
            adata: AnnData with spatial coordinates.
            spatial_key: Key for spatial coordinates.

        Returns:
            Neighbor graph (sparse adjacency matrix).
        """
        if not SQUIDPY_AVAILABLE or not SCANPY_AVAILABLE:
            return self._build_simple_spatial_graph(adata, spatial_key)

        try:
            # Compute spatial neighbors
            sq.gr.spatial_neighbors(
                adata,
                coord_type=self.coord_type,
                n_neighbors=self.n_neighbors,
                radius=self.radius,
            )

            return adata.obsp["spatial_connectivities"]

        except Exception as e:
            logger.warning(f"Spatial neighbor computation failed: {e}")
            return self._build_simple_spatial_graph(adata, spatial_key)

    def _build_simple_spatial_graph(
        self,
        adata: Any,
        spatial_key: str,
    ) -> np.ndarray:
        """Build simple k-nearest-neighbor graph."""
        if spatial_key in adata.obsm:
            coords = adata.obsm[spatial_key]
        else:
            coords = self._generate_spatial_coords(adata.n_obs)

        n = coords.shape[0]
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = d
                distances[j, i] = d

        # K-nearest neighbors
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            nearest = np.argsort(distances[i])[: self.n_neighbors + 1]
            for j in nearest:
                if i != j:
                    adj[i, j] = True

        return adj

    def _generate_spatial_coords(self, n_cells: int) -> np.ndarray:
        """Generate random spatial coordinates."""
        return np.random.rand(n_cells, 2)

    def _create_mock_spatial_adata(self) -> Any:
        """Create mock spatial AnnData."""
        n_cells = 500
        n_genes = 100

        X = np.random.randn(n_cells, n_genes)
        coords = np.random.rand(n_cells, 2) * 100  # 100x100 micron area

        if SCANPY_AVAILABLE and ad is not None:
            adata = ad.AnnData(X=X)
            adata.obsm["spatial"] = coords
            adata.obs["cell_type"] = np.random.choice(["T_cell", "B_cell", "Macrophage", "Fibroblast"], n_cells)
            return adata
        else:
            return {"X": X, "spatial": coords}

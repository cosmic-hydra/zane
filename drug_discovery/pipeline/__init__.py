"""
Autonomous Data Pipeline Module

Provides production-grade data pipeline infrastructure with:
- Streaming data processing
- Asynchronous execution
- Fault tolerance and recovery
- Quality monitoring
- Orchestration and scheduling
"""

# Keep backward compatibility with the legacy high-level pipeline API.
import importlib.util
from pathlib import Path

from drug_discovery.pipeline.autonomous_pipeline import (
    DataBatch,
    DataQualityMonitor,
    FaultTolerantExecutor,
    PipelineCheckpoint,
    PipelineOrchestrator,
    StreamingDataPipeline,
)

_LEGACY_PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"
_legacy_spec = importlib.util.spec_from_file_location("drug_discovery._legacy_pipeline", _LEGACY_PIPELINE_PATH)
if _legacy_spec and _legacy_spec.loader:
    _legacy_module = importlib.util.module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)
    DrugDiscoveryPipeline = _legacy_module.DrugDiscoveryPipeline
else:
    DrugDiscoveryPipeline = None

__all__ = [
    "DrugDiscoveryPipeline",
    "StreamingDataPipeline",
    "PipelineOrchestrator",
    "DataQualityMonitor",
    "FaultTolerantExecutor",
    "DataBatch",
    "PipelineCheckpoint",
]

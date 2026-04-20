"""ZANE — AI-native Drug Discovery Platform."""

__version__ = "2026.4.1"
__all__ = ["__version__", "DrugDiscoveryPipeline"]

try:
    from drug_discovery.pipeline import DrugDiscoveryPipeline
except ImportError:
    pass

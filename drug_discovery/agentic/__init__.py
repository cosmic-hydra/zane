"""Agentic Module — Multi-Agent Swarms and Automated IND Formatting."""

from .fda_formatter import INDGenerator
from .swarm import AgenticSwarm, BioethicsAgent, TranslationAgent

__all__ = ["AgenticSwarm", "BioethicsAgent", "TranslationAgent", "INDGenerator"]

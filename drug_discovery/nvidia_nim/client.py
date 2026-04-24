"""NVIDIA NIM API client for chemistry-focused LLMs and BioNeMo models.

The NVIDIA NIM platform exposes pre-trained LLMs (Llama, Nemotron, Mistral,
…) through an OpenAI-compatible REST API, plus dedicated BioNeMo microservices
for molecular biology.  This module wraps both surfaces:

* ``NvidiaNIMClient.generate_molecules`` — sends an enriched prompt to a
  NIM-hosted chat model and parses SMILES from the response.
* ``NvidiaNIMClient.explain_molecule`` — asks the LLM to reason about a given
  SMILES string (ADMET, target selectivity, SAR, …).
* ``NvidiaNIMClient.generate_molmim`` — calls the BioNeMo MolMIM endpoint to
  perform latent-space optimisation around a seed SMILES.

Environment variables
---------------------
NVIDIA_NIM_API_KEY
    Required.  API key from https://build.nvidia.com/explore/discover.
NVIDIA_NIM_BASE_URL
    Optional override for the chat API base URL
    (default: ``https://integrate.api.nvidia.com/v1``).
NVIDIA_NIM_CHAT_MODEL
    Optional override for the default chat model
    (default: ``nvidia/llama-3.1-nemotron-70b-instruct``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MOLMIM_URL = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

#: Default NIM-hosted model for chemistry/drug-design tasks.
#: Uses Llama-3.1-Nemotron-70B — NVIDIA's flagship instruction-tuned model.
DEFAULT_CHAT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NvidiaNIMConfig:
    """Runtime configuration for the NVIDIA NIM client."""

    api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_NIM_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_NIM_BASE_URL", NVIDIA_NIM_BASE_URL)
    )
    chat_model: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_NIM_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    )
    max_tokens: int = 1024
    temperature: float = 0.4
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NvidiaNIMClient:
    """Client for the NVIDIA NIM API and BioNeMo microservices.

    All methods fail gracefully: when the ``openai`` package is not installed
    or ``NVIDIA_NIM_API_KEY`` is not set, they return a structured error dict
    instead of raising.
    """

    def __init__(self, config: NvidiaNIMConfig | None = None):
        self.config = config or NvidiaNIMConfig()
        self._client: Any | None = None  # openai.OpenAI when initialised

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when the ``openai`` package is importable and an API key is set."""
        return importlib.util.find_spec("openai") is not None and bool(self.config.api_key)

    def _get_client(self) -> Any:
        """Lazy-init the OpenAI-compatible client pointed at NIM."""
        if self._client is None:
            from openai import OpenAI  # type: ignore[import]

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are ZANE ChemAI, an expert computational chemist powered by NVIDIA technology. "
            "You specialize in de novo drug design, molecular optimization, fragment-based design, "
            "ADMET prediction, and multi-target drug combinations. "
            "When asked to generate molecules, always output valid SMILES strings, one per line, "
            "preceded by 'SMILES:'. Provide brief reasoning for each design choice. "
            "Prioritize drug-likeness (Lipinski Ro5), synthetic accessibility, and target selectivity."
        )

    # ------------------------------------------------------------------
    # Generate molecules via NIM chat model
    # ------------------------------------------------------------------

    def generate_molecules(
        self,
        prompt: str,
        num: int = 10,
        context: str | None = None,
        seed_smiles: list[str] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Ask the NIM-hosted LLM to propose novel drug-like molecules.

        Args:
            prompt: Free-form design objective or target description.
            num: Number of candidate molecules to request.
            context: Optional chemical database context block to include.
            seed_smiles: Optional seed / scaffold SMILES to guide generation.
            temperature: Sampling temperature override.

        Returns:
            Dict with keys:
            - ``success`` (bool)
            - ``molecules`` (list[str]) — parsed SMILES
            - ``reasoning`` (str) — full LLM response
            - ``raw_response`` (str) — same as reasoning
            - ``error`` (str | None)
        """
        if not self.is_available():
            return {
                "success": False,
                "molecules": [],
                "reasoning": "",
                "raw_response": "",
                "error": (
                    "NVIDIA NIM not available. "
                    "Install the 'openai' package and set the NVIDIA_NIM_API_KEY "
                    "environment variable (https://build.nvidia.com/explore/discover)."
                ),
            }

        parts: list[str] = []
        if context:
            parts.append(f"Chemical Database Context:\n{context.strip()}")
        if seed_smiles:
            parts.append("Seed / scaffold SMILES:\n" + "\n".join(seed_smiles))
        parts.append(f"Task:\n{prompt.strip()}")
        parts.append(
            f"Generate exactly {num} diverse, drug-like molecules. "
            "For each, output one line starting with 'SMILES: ' followed by the SMILES string, "
            "then a brief one-sentence rationale starting with 'Reason: '."
        )
        user_message = "\n\n".join(parts)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                top_p=self.config.top_p,
            )
            raw: str = response.choices[0].message.content or ""
            molecules = self._extract_smiles(raw)
            return {
                "success": True,
                "molecules": molecules[:num],
                "reasoning": raw,
                "raw_response": raw,
                "error": None,
            }
        except Exception as exc:
            logger.debug("NIM generate_molecules failed: %s", exc)
            return {
                "success": False,
                "molecules": [],
                "reasoning": "",
                "raw_response": "",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Explain a molecule
    # ------------------------------------------------------------------

    def explain_molecule(
        self,
        smiles: str,
        question: str = (
            "Explain the drug-likeness, predicted ADMET properties, "
            "and potential therapeutic applications of this molecule."
        ),
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Ask the LLM to reason about a specific molecule.

        Args:
            smiles: SMILES string of the molecule to explain.
            question: Specific question to ask about the molecule.
            max_tokens: Maximum tokens for the response.

        Returns:
            Dict with keys ``success``, ``explanation``, ``error``.
        """
        if not self.is_available():
            return {
                "success": False,
                "explanation": "",
                "error": "NVIDIA NIM not available.",
            }
        try:
            client = self._get_client()
            user_msg = f"Molecule SMILES: {smiles}\n\nQuestion: {question}"
            response = client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
            )
            raw: str = response.choices[0].message.content or ""
            return {"success": True, "explanation": raw, "error": None}
        except Exception as exc:
            logger.debug("NIM explain_molecule failed: %s", exc)
            return {"success": False, "explanation": "", "error": str(exc)}

    # ------------------------------------------------------------------
    # BioNeMo MolMIM generation
    # ------------------------------------------------------------------

    def generate_molmim(
        self,
        smiles: str,
        num_molecules: int = 10,
        iterations: int = 10,
        algorithm: str = "CMA-ES",
        min_similarity: float = 0.3,
        property_name: str = "QED",
        minimize: bool = False,
    ) -> dict[str, Any]:
        """Call the NVIDIA BioNeMo MolMIM generation endpoint.

        MolMIM encodes the seed SMILES into a continuous latent space and then
        runs a CMA-ES (or similar) optimisation loop to find neighbours that
        maximise (or minimise) a molecular property (default: QED).

        Args:
            smiles: Seed SMILES string.
            num_molecules: Number of molecules to generate.
            iterations: Number of optimisation iterations.
            algorithm: Optimisation algorithm (``"CMA-ES"`` recommended).
            min_similarity: Minimum Tanimoto similarity to the seed.
            property_name: Property to optimise (``"QED"``, ``"logP"``, …).
            minimize: If True, minimise the property instead of maximising.

        Returns:
            Dict with keys ``success``, ``molecules`` (list[str]), ``raw``,
            ``error``.
        """
        if not self.config.api_key:
            return {
                "success": False,
                "molecules": [],
                "raw": {},
                "error": "NVIDIA_NIM_API_KEY not set.",
            }
        try:
            import requests as _req

            payload = {
                "algorithm": algorithm,
                "num_molecules": num_molecules,
                "property_name": property_name,
                "minimize": minimize,
                "min_similarity": min_similarity,
                "particles": 30,
                "iterations": iterations,
                "smi": smiles,
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            resp = _req.post(NVIDIA_MOLMIM_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            molecules = [
                entry.get("smiles", "")
                for entry in data.get("molecules", [])
                if entry.get("smiles")
            ]
            return {"success": True, "molecules": molecules, "raw": data, "error": None}
        except Exception as exc:
            logger.debug("MolMIM request failed: %s", exc)
            return {"success": False, "molecules": [], "raw": {}, "error": str(exc)}

    # ------------------------------------------------------------------
    # SMILES extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_smiles(text: str) -> list[str]:
        """Parse SMILES strings from free-form LLM output.

        Looks for lines containing ``SMILES: <token>`` first; falls back to
        lines that look like bare SMILES strings.
        """
        smiles: list[str] = []

        # Primary: explicit ``SMILES:`` prefix on any line
        for line in text.splitlines():
            m = re.match(r"(?:SMILES:\s*)([^\s]+)", line.strip(), re.IGNORECASE)
            if m:
                candidate = m.group(1).rstrip(".,;")
                if candidate:
                    smiles.append(candidate)

        if smiles:
            return smiles

        # Fallback: lines that look like SMILES (letters, digits, brackets …)
        smiles_pattern = re.compile(r"^[A-Za-z0-9@#%\[\]\(\)\+\-\=\/\\\.]{6,}$")
        for line in text.splitlines():
            stripped = line.strip()
            if smiles_pattern.match(stripped):
                smiles.append(stripped)

        return smiles

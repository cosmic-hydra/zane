"""Llama-based AI support utilities for drug discovery workflows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch

DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class AISupportConfig:
    """Runtime settings for the AI support assistant."""

    model_id: str = DEFAULT_LLAMA_MODEL
    device_map: str = "auto"


def _validate_generation_params(max_new_tokens: int, temperature: float, top_p: float) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")


class LlamaSupportAssistant:
    """Small wrapper around a Meta Llama chat model for project assistance."""

    def __init__(self, config: AISupportConfig | None = None):
        self.config = config or AISupportConfig()
        self._tokenizer: Any | None = None  # AutoTokenizer when loaded
        self._model: Any | None = None  # AutoModelForCausalLM when loaded

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency import issue
            raise RuntimeError("Transformers is required for Llama AI support. Install requirements first.") from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # `device_map` requires accelerate; fall back cleanly when unavailable.
            model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
            if self.config.device_map and find_spec("accelerate") is not None:
                model_kwargs["device_map"] = self.config.device_map

            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
            if (
                getattr(self._tokenizer, "pad_token_id", None) is None
                and getattr(self._tokenizer, "eos_token_id", None) is not None
            ):
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Meta Llama model. Ensure you have accepted model access on Hugging Face "
                "and set a valid token (for example: export HF_TOKEN=...)."
            ) from exc

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are ZANE AI Support, an assistant for computational drug discovery. "
            "Provide practical guidance for molecular modeling, ADMET reasoning, simulation interpretation, "
            "and experimental planning. Keep answers concise, technical, and safety-aware."
        )

    def respond_with_knowledge(
        self,
        user_prompt: str,
        kg: Any | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Respond to user request using Knowledge Graph context if available."""
        context = ""
        if kg is not None:
            try:
                # Attempt to retrieve relevant nodes/triplets from KG
                # Use hybrid_search if available (assuming query embedding is handled internally or simplified)
                if hasattr(kg, "hybrid_search"):
                    # Simplified: just using the prompt text as "embedding" or assuming kg handles it
                    results = kg.hybrid_search(user_prompt, top_k=5)
                elif hasattr(kg, "search"):
                    results = kg.search(user_prompt, limit=5)
                else:
                    results = []

                if results:
                    context = "Knowledge Graph Insights:\n"
                    for item in results:
                        context += f"- {item}\n"
            except Exception as e:
                context = f"Note: Knowledge Graph context partially available. Search error: {e}"

        return self.respond(user_prompt, context=context, max_new_tokens=max_new_tokens)

    def respond(
        self,
        user_prompt: str,
        context: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response using the configured Llama model.

        Args:
            user_prompt: User question or request (cannot be empty).
            context: Optional domain-specific context to prepend to prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (0 = deterministic, higher = more creative).
            top_p: Nucleus sampling parameter (0, 1].

        Returns:
            Generated response text.

        Raises:
            ValueError: If user_prompt is empty or generation parameters are invalid.
            RuntimeError: If model loading or generation fails.
        """
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        _validate_generation_params(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

        self._load()

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Llama model is not initialized.")

        combined_prompt = user_prompt.strip()
        if context:
            combined_prompt = f"Context:\n{context.strip()}\n\nUser Request:\n{combined_prompt}"

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": combined_prompt},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                templated = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                # Tokenizers may return either a tensor or a BatchEncoding-like mapping.
                if isinstance(templated, torch.Tensor):
                    model_input = templated
                    attention_mask = torch.ones_like(model_input)
                else:
                    model_input = templated["input_ids"]
                    attention_mask = templated.get("attention_mask")
            except Exception:
                fallback_text = f"System: {self._system_prompt()}\n\n" f"User: {combined_prompt}\n\n" "Assistant:"
                encoded = self._tokenizer(fallback_text, return_tensors="pt")
                model_input = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
        else:
            fallback_text = f"System: {self._system_prompt()}\n\n" f"User: {combined_prompt}\n\n" "Assistant:"
            encoded = self._tokenizer(fallback_text, return_tensors="pt")
            model_input = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

        model_input = model_input.to(self._model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)

        do_sample = temperature > 0.0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-6),
            "top_p": top_p,
            "attention_mask": attention_mask,
        }

        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            output = self._model.generate(model_input, **generate_kwargs)

        generated = output[0][model_input.shape[-1] :]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)
        cleaned = response.strip()
        if not cleaned:
            raise RuntimeError("Model returned an empty response.")
        return cleaned

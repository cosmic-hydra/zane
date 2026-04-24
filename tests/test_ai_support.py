"""Tests for Llama AI support helpers."""

import pytest

torch = pytest.importorskip("torch")

from drug_discovery.ai_support import LlamaSupportAssistant


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        assert messages and tokenize and add_generation_prompt and return_tensors == "pt"
        return torch.tensor([[10, 11, 12]])

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        if len(token_ids) == 0:
            return ""
        return "ok response"


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, model_input, **kwargs):
        assert kwargs["max_new_tokens"] > 0
        assert kwargs["top_p"] > 0
        # Return prompt tokens + one generated token.
        return torch.cat([model_input, torch.tensor([[99]])], dim=1)


def _assistant_with_fakes() -> LlamaSupportAssistant:
    assistant = LlamaSupportAssistant()
    assistant._tokenizer = _FakeTokenizer()
    assistant._model = _FakeModel()
    return assistant


def test_respond_returns_text_with_fake_backend():
    assistant = _assistant_with_fakes()
    result = assistant.respond("Summarize current candidates")
    assert result == "ok response"


def test_respond_rejects_empty_prompt():
    assistant = _assistant_with_fakes()
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        assistant.respond("  ")


@pytest.mark.parametrize(
    "max_new_tokens,temperature,top_p,error_msg",
    [
        (0, 0.7, 0.9, "max_new_tokens must be > 0"),
        (8, -0.1, 0.9, "temperature must be >= 0"),
        (8, 0.2, 0.0, r"top_p must be in \(0, 1\]"),
    ],
)
def test_respond_validates_generation_params(max_new_tokens, temperature, top_p, error_msg):
    assistant = _assistant_with_fakes()
    with pytest.raises(ValueError, match=error_msg):
        assistant.respond(
            "help",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def test_respond_raises_if_model_returns_empty_string(monkeypatch):
    assistant = _assistant_with_fakes()

    def _empty_decode(*_args, **_kwargs):
        return "   "

    monkeypatch.setattr(assistant._tokenizer, "decode", _empty_decode)

    with pytest.raises(RuntimeError, match="empty response"):
        assistant.respond("help")

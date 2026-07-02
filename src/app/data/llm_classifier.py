from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

from app.schemas.domain import SentimentDirection


@dataclass(frozen=True)
class EventLLMClassification:
    sentiment: SentimentDirection
    summary: str
    key_facts: tuple[str, ...]
    event_labels: tuple[str, ...]
    companies: tuple[str, ...]
    tickers: tuple[str, ...]
    sectors: tuple[str, ...]
    confidence: float
    model: str


class LLMTextClient(Protocol):
    model: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        ...


class JsonEventLLMClassifier:
    def __init__(self, client: LLMTextClient) -> None:
        self.client = client

    def classify(
        self,
        title: str,
        body: str,
        known_tickers: dict[str, str] | None = None,
    ) -> EventLLMClassification:
        known_tickers = known_tickers or {}
        payload = self.client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_user_prompt(title, body, known_tickers),
        )
        data = _parse_json_object(payload)
        sentiment = _sentiment(str(data.get("sentiment", "NEUTRAL")))
        return EventLLMClassification(
            sentiment=sentiment,
            summary=str(data.get("summary") or body[:280]).strip()[:700],
            key_facts=tuple(str(item).strip()[:180] for item in data.get("key_facts", []) if str(item).strip())[:8],
            event_labels=tuple(str(item).strip()[:80] for item in data.get("event_labels", []) if str(item).strip())[:8],
            companies=tuple(str(item).strip()[:120] for item in data.get("companies", []) if str(item).strip())[:12],
            tickers=tuple(str(item).strip().upper()[:20] for item in data.get("tickers", []) if str(item).strip())[:12],
            sectors=tuple(str(item).strip()[:80] for item in data.get("sectors", []) if str(item).strip())[:8],
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            model=self.client.model,
        )


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat-completions adapter.

    Configure with env vars through `build_event_llm_classifier_from_env`.
    This adapter is optional; tests use fake clients and the app falls back to
    keyword classification when no LLM env vars are set.
    """

    def __init__(self, api_key: str, model: str, endpoint: str) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        timeout_seconds = _request_timeout_seconds()
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": _request_max_tokens(),
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


class LocalOpenAICompatibleChatClient:
    """OpenAI-compatible local chat server adapter.

    Works with local servers that expose `/v1/chat/completions`, such as
    Ollama's OpenAI-compatible endpoint or llama.cpp server. No API key is
    required.
    """

    def __init__(self, model: str, endpoint: str) -> None:
        self.model = model
        self.endpoint = endpoint

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        timeout_seconds = _request_timeout_seconds()
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": _request_max_tokens(),
                "stream": False,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


class EmbeddedTransformersChatClient:
    """In-process local LLM adapter backed by Hugging Face Transformers.

    This keeps news/event classification fully local without requiring an
    Ollama or llama.cpp server. The dependency is intentionally optional; if
    transformers/torch or the model files are unavailable, the caller's normal
    keyword fallback remains in charge.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

    @cached_property
    def _tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the local LLM extras: pip install .[local-llm]") from exc
        return AutoTokenizer.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )

    @cached_property
    def _model(self) -> Any:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install the local LLM extras: pip install .[local-llm]") from exc
        kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        if self.device not in {"auto", ""}:
            model = model.to(self.device)
        return model.eval()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        tokenizer = self._tokenizer
        prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        try:
            first_parameter = next(self._model.parameters())
            inputs = inputs.to(first_parameter.device)
        except StopIteration:
            pass
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


class EmbeddedMultimodalTransformersChatClient:
    """In-process local multimodal LLM adapter backed by Hugging Face Transformers.

    Use this for vision-language or other multimodal checkpoints that expose an
    AutoProcessor and AutoModelForMultimodalLM interface.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

    @cached_property
    def _processor(self) -> Any:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the local LLM extras: pip install .[local-llm]") from exc
        return AutoProcessor.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )

    @cached_property
    def _model(self) -> Any:
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Upgrade transformers to a version that provides AutoModelForMultimodalLM"
            ) from exc
        kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForMultimodalLM.from_pretrained(self.model, **kwargs)
        if self.device not in {"auto", ""}:
            model = model.to(self.device)
        return model.eval()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        processor = self._processor
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            model_device = getattr(self._model, "device", None)
            if model_device is not None:
                inputs = inputs.to(model_device)
            else:
                first_parameter = next(self._model.parameters())
                inputs = inputs.to(first_parameter.device)
        except StopIteration:
            pass
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return str(processor.decode(generated, skip_special_tokens=True)).strip()


class EmbeddedOpenVINOChatClient:
    """In-process local LLM adapter using OpenVINO/Optimum Intel.

    Use this when the machine has Intel NPU/OpenVINO support and the model can
    be exported or loaded by Optimum Intel. It is optional and lazy-loaded so
    the default app does not require these packages.
    """

    def __init__(
        self,
        model: str,
        device: str = "NPU",
        max_new_tokens: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

    @cached_property
    def _tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the local LLM extras: pip install .[openvino-llm]") from exc
        return AutoTokenizer.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )

    @cached_property
    def _model(self) -> Any:
        try:
            from optimum.intel.openvino import OVModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install OpenVINO LLM extras: pip install .[openvino-llm]") from exc
        return OVModelForCausalLM.from_pretrained(
            self.model,
            export=True,
            device=self.device,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        tokenizer = self._tokenizer
        prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def configure_default_event_llm_env() -> dict[str, Any]:
    """Enable a local event LLM when no explicit LLM env was provided."""
    if os.getenv("LLM_EVENT_CLASSIFIER_ENABLED"):
        return event_llm_runtime_status()
    if not os.getenv("LLM_EVENT_PROVIDER"):
        os.environ["LLM_EVENT_PROVIDER"] = "local"
    if not os.getenv("LLM_EVENT_MODEL"):
        os.environ["LLM_EVENT_MODEL"] = "qwen2.5:1.5b-instruct"
    if not os.getenv("LLM_EVENT_LOCAL_ENDPOINT"):
        os.environ["LLM_EVENT_LOCAL_ENDPOINT"] = "http://127.0.0.1:11434/v1/chat/completions"
    os.environ.setdefault("LLM_EVENT_MAX_ITEMS_PER_SOURCE", "1")
    os.environ.setdefault("LLM_EVENT_MAX_ITEMS_PER_RUN", "1")
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
            os.environ["LLM_EVENT_CLASSIFIER_ENABLED"] = "true" if response.status == 200 else "false"
    except Exception:
        os.environ["LLM_EVENT_CLASSIFIER_ENABLED"] = "false"
    return event_llm_runtime_status()


def build_event_llm_classifier_from_env() -> JsonEventLLMClassifier | None:
    enabled = os.getenv("LLM_EVENT_CLASSIFIER_ENABLED", "").lower() in {"1", "true", "yes"}
    provider = os.getenv("LLM_EVENT_PROVIDER", "remote").strip().lower()
    model = os.getenv("LLM_EVENT_MODEL")
    if not enabled or not model:
        return None
    if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
        endpoint = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
            "LLM_EVENT_ENDPOINT",
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        return JsonEventLLMClassifier(LocalOpenAICompatibleChatClient(model=model, endpoint=endpoint))
    if provider in {"embedded", "inprocess", "transformers", "local-model", "openvino-llm", "multimodal"}:
        device = os.getenv("LLM_EVENT_DEVICE", "auto").strip()
        max_new_tokens = int(os.getenv("LLM_EVENT_MAX_NEW_TOKENS", "512"))
        cache_dir = os.getenv("LLM_EVENT_MODEL_CACHE_DIR") or None
        local_files_only = os.getenv("LLM_EVENT_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"}
        backend = os.getenv("LLM_EVENT_INFERENCE_BACKEND", "").strip().lower()
        if provider == "openvino-llm" or backend == "openvino" or device.upper() == "NPU":
            return JsonEventLLMClassifier(
                EmbeddedOpenVINOChatClient(
                    model=model,
                    device=device if device != "auto" else "NPU",
                    max_new_tokens=max_new_tokens,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        if provider == "multimodal":
            return JsonEventLLMClassifier(
                EmbeddedMultimodalTransformersChatClient(
                    model=model,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        return JsonEventLLMClassifier(
            EmbeddedTransformersChatClient(
                model=model,
                device=device,
                max_new_tokens=max_new_tokens,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    api_key = os.getenv("LLM_EVENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    endpoint = os.getenv("LLM_EVENT_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    return JsonEventLLMClassifier(OpenAICompatibleChatClient(api_key=api_key, model=model, endpoint=endpoint))


def event_llm_runtime_status() -> dict[str, Any]:
    enabled = os.getenv("LLM_EVENT_CLASSIFIER_ENABLED", "").lower() in {"1", "true", "yes"}
    provider = os.getenv("LLM_EVENT_PROVIDER", "remote").strip().lower()
    model = os.getenv("LLM_EVENT_MODEL", "")
    backend = os.getenv("LLM_EVENT_INFERENCE_BACKEND", "")
    device = os.getenv("LLM_EVENT_DEVICE", "")
    status: dict[str, Any] = {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "backend": backend,
        "device": device,
        "available": False,
        "reason": None,
    }
    if not enabled:
        if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
            status["backend"] = "ollama"
            status["device"] = "ollama-managed"
            status["endpoint"] = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
                "LLM_EVENT_ENDPOINT",
                "http://127.0.0.1:11434/v1/chat/completions",
            )
            try:
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
                    status["available"] = response.status == 200
                    status["reason"] = (
                        "LLM_EVENT_CLASSIFIER_ENABLED is false, but local LLM endpoint is reachable."
                        if status["available"]
                        else f"local LLM status {response.status}"
                    )
            except Exception as exc:
                status["reason"] = f"LLM disabled because local LLM endpoint is unavailable: {exc}"
            return status
        status["reason"] = "LLM_EVENT_CLASSIFIER_ENABLED is false."
        return status
    if not model:
        status["reason"] = "LLM_EVENT_MODEL is not configured."
        return status
    if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
        status["backend"] = "ollama"
        status["device"] = "ollama-managed"
        endpoint = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
            "LLM_EVENT_ENDPOINT",
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        status["endpoint"] = endpoint
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as response:
                status["available"] = response.status == 200
                status["reason"] = None if status["available"] else f"local LLM status {response.status}"
        except Exception as exc:
            status["reason"] = f"local LLM unavailable: {exc}"
        return status
    if provider in {"embedded", "inprocess", "transformers", "local-model", "openvino-llm", "multimodal"}:
        model_path = Path(model)
        if model_path.exists():
            status["available"] = True
            status["reason"] = None
        else:
            status["reason"] = f"embedded model path does not exist: {model}"
        return status
    if os.getenv("LLM_EVENT_API_KEY") or os.getenv("OPENAI_API_KEY"):
        status["available"] = True
        return status
    status["reason"] = "remote provider needs LLM_EVENT_API_KEY or OPENAI_API_KEY."
    return status


_SYSTEM_PROMPT = """You classify financial news and disclosures for a personal investment research system.
Return only valid JSON with keys:
sentiment: POSITIVE, NEGATIVE, or NEUTRAL
summary: concise factual summary
key_facts: array of factual bullet strings
event_labels: array such as EarningsSurprisePositive, GuidanceLowered, MajorSupplyContract, AnalystUpgrade, LitigationRiskHigh, RegulatoryPenaltyNegative, ProductLaunchPositive, RumorRisk
companies: array
tickers: array
sectors: array
confidence: number between 0 and 1
Do not invent facts. Use NEUTRAL and low confidence when the text is ambiguous."""


def _user_prompt(title: str, body: str, known_tickers: dict[str, str]) -> str:
    prompt_limit = _known_ticker_prompt_limit()
    ticker_items = tuple(sorted(known_tickers.items()))[:prompt_limit]
    known = ", ".join(f"{ticker}={company}" for ticker, company in ticker_items)
    if len(known_tickers) > prompt_limit:
        known = f"{known}, ... ({len(known_tickers) - prompt_limit} more tracked tickers omitted from prompt)"
    text = f"{title}\n\n{body}"
    return f"Known tickers: {known or 'none'}\n\nText:\n{text[:2500]}"


def _known_ticker_prompt_limit() -> int:
    try:
        return max(10, int(os.getenv("LLM_EVENT_KNOWN_TICKER_PROMPT_LIMIT", "80")))
    except ValueError:
        return 80


def _request_timeout_seconds() -> float:
    try:
        return max(3.0, float(os.getenv("LLM_EVENT_TIMEOUT_SECONDS", "12")))
    except ValueError:
        return 12.0


def _request_max_tokens() -> int:
    try:
        return max(64, int(os.getenv("LLM_EVENT_RESPONSE_MAX_TOKENS", "180")))
    except ValueError:
        return 180


def _chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{user_prompt}\n"
        "<|assistant|>\n"
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _sentiment(value: str) -> SentimentDirection:
    normalized = value.strip().upper()
    if normalized == "POSITIVE":
        return SentimentDirection.POSITIVE
    if normalized == "NEGATIVE":
        return SentimentDirection.NEGATIVE
    return SentimentDirection.NEUTRAL

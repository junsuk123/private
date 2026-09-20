from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, time
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol
from zoneinfo import ZoneInfo

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
        event_labels = _ground_event_labels(
            tuple(
                str(item).strip()[:80]
                for item in data.get("event_labels", [])
                if str(item).strip()
            )[:8],
            title=title,
            body=body,
        )
        tickers = _normalize_llm_tickers(data.get("tickers", ()), known_tickers)
        return EventLLMClassification(
            sentiment=sentiment,
            summary=str(data.get("summary") or body[:280]).strip()[:700],
            key_facts=tuple(str(item).strip()[:180] for item in data.get("key_facts", []) if str(item).strip())[:8],
            event_labels=event_labels,
            companies=tuple(str(item).strip()[:120] for item in data.get("companies", []) if str(item).strip())[:12],
            tickers=tickers,
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
        payload: dict[str, Any] = {
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
        # Qwen3-class models think by default. Event extraction needs the final
        # JSON, not an internal reasoning trace consuming the complete token
        # budget. Ollama's OpenAI-compatible API exposes this as
        # ``reasoning_effort=none``. Keep it opt-in so other compatible servers
        # that do not implement the field remain usable.
        reasoning_effort = _request_reasoning_effort()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        body = json.dumps(payload).encode("utf-8")
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


def load_shared_local_llm_env(path: str | None = None) -> dict[str, str]:
    """Load the shared local-LLM config file as environment defaults.

    A single `config/local_llm.env` (KEY=VALUE lines) is the one place both the
    Windows launcher and the Raspberry Pi launcher configure the news/event LLM,
    so the model is set once for every machine. Applied with setdefault
    semantics: values already present in the environment (shell / launcher) win
    and are never overwritten. Missing file is a no-op.
    """
    config_path = path or os.getenv("LOCAL_LLM_CONFIG", "config/local_llm.env")
    applied: dict[str, str] = {}
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, OSError):
        return applied
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _local_llm_reachable(endpoint: str | None = None) -> tuple[bool, str]:
    """Probe an OpenAI-compatible local LLM server (Ollama, llama.cpp, ...).

    Derives host:port from the configured endpoint and tries a few common
    liveness paths, so auto-detect works for any local server — not just
    Ollama's `/api/tags`. Returns (reachable, detail).
    """
    from urllib.parse import urlparse

    endpoint = endpoint or os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
        "LLM_EVENT_ENDPOINT", "http://127.0.0.1:11434/v1/chat/completions"
    )
    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "http://127.0.0.1:11434"
    detail = "no endpoint probed"
    for path in ("/health", "/v1/models", "/api/tags"):
        try:
            with urllib.request.urlopen(base + path, timeout=1.5) as response:
                if 200 <= response.status < 500:
                    return True, f"reachable via {path} ({response.status})"
                detail = f"{path} -> {response.status}"
        except Exception as exc:  # noqa: BLE001 - keep probing the next path.
            detail = f"{path} -> {exc}"
    return False, detail


def _ollama_installed_models(endpoint: str) -> tuple[dict[str, Any], ...] | None:
    """Return Ollama's local catalogue, or ``None`` for a non-Ollama endpoint."""
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - llama.cpp and other local servers are valid.
        return None
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    return tuple(row for row in rows if isinstance(row, dict))


def _local_model_quality_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Rank text-instruction models without mistaking embeddings for chat LLMs."""
    name = str(row.get("name") or row.get("model") or "").lower()
    family = 4.0 if "qwen3.5" in name else 3.0 if "qwen3" in name else 2.0 if "qwen2.5" in name else 1.0
    parameter_match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b(?:\b|[-_:])", name)
    parameters = float(parameter_match.group(1)) if parameter_match else 0.0
    instruct = 1.0 if "instruct" in name else 0.0
    return (parameters, family, instruct)


def _resolve_local_ollama_model(
    configured_model: str,
    endpoint: str,
) -> tuple[str, bool, tuple[str, ...]]:
    """Resolve a synced config against models installed on this particular PC.

    Synology shares ``config/local_llm.env`` between unlike machines. A model
    name valid on an RTX desktop must not turn into repeated HTTP 404 responses
    on a laptop. Prefer the configured model when present; otherwise select the
    strongest installed text model and expose that fallback in diagnostics.
    """
    rows = _ollama_installed_models(endpoint)
    if rows is None:
        return configured_model, False, ()
    names = tuple(
        str(row.get("name") or row.get("model") or "").strip()
        for row in rows
        if str(row.get("name") or row.get("model") or "").strip()
    )
    if configured_model in names:
        return configured_model, False, names
    excluded = ("embed", "rerank", "vision", "-vl", "coder", "code-")
    eligible = [
        row
        for row in rows
        if not any(
            token in str(row.get("name") or row.get("model") or "").lower()
            for token in excluded
        )
    ]
    if not eligible:
        return configured_model, False, names
    best = max(eligible, key=_local_model_quality_key)
    resolved = str(best.get("name") or best.get("model") or configured_model).strip()
    return resolved, resolved != configured_model, names


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _market_session_open(now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(ZoneInfo("UTC"))
    kr_now = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    if kr_now.weekday() < 5 and time(9, 0) <= kr_now.time() <= time(15, 30):
        return True
    us_now = now_utc.astimezone(ZoneInfo("America/New_York"))
    if us_now.weekday() < 5 and time(9, 30) <= us_now.time() <= time(16, 0):
        return True
    return False


def _cpu_load_per_core() -> float | None:
    if not hasattr(os, "getloadavg"):
        return None
    try:
        one_min_load = os.getloadavg()[0]
    except OSError:
        return None
    cores = os.cpu_count() or 1
    return max(0.0, float(one_min_load) / float(cores))


def event_llm_opportunistic_gate_status() -> dict[str, Any]:
    """Return whether opportunistic local LLM use is allowed right now."""

    enabled = _env_flag("LLM_EVENT_OPPORTUNISTIC_ENABLED", False)
    status: dict[str, Any] = {
        "enabled": enabled,
        "allowed": True,
        "reason": "opportunistic gate disabled",
        "market_open": False,
        "load_per_core": None,
        "max_load_per_core": _env_float("LLM_EVENT_OPPORTUNISTIC_MAX_LOAD_PER_CORE", 0.60),
    }
    if not enabled:
        return status

    if _env_flag("LLM_EVENT_OPPORTUNISTIC_DISABLE_DURING_MARKET", True):
        market_open = _market_session_open()
        status["market_open"] = market_open
        if market_open:
            status["allowed"] = False
            status["reason"] = "market session open"
            return status

    load_per_core = _cpu_load_per_core()
    status["load_per_core"] = load_per_core
    max_load = float(status["max_load_per_core"])
    if load_per_core is not None and load_per_core > max_load:
        status["allowed"] = False
        status["reason"] = f"cpu load too high ({load_per_core:.2f}>{max_load:.2f})"
        return status

    status["reason"] = "allowed"
    return status


def configure_default_event_llm_env() -> dict[str, Any]:
    """Enable a local event LLM when no explicit LLM env was provided."""
    load_shared_local_llm_env()
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
    reachable, _detail = _local_llm_reachable()
    os.environ["LLM_EVENT_CLASSIFIER_ENABLED"] = "true" if reachable else "false"
    return event_llm_runtime_status()


def build_event_llm_classifier_from_env() -> JsonEventLLMClassifier | None:
    load_shared_local_llm_env()
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
        if _env_flag("LLM_EVENT_OPPORTUNISTIC_ENABLED", False):
            gate = event_llm_opportunistic_gate_status()
            if not gate["allowed"]:
                return None
            reachable, _detail = _local_llm_reachable(endpoint)
            if not reachable:
                return None
        resolved_model, _fallback, _installed = _resolve_local_ollama_model(model, endpoint)
        return JsonEventLLMClassifier(
            LocalOpenAICompatibleChatClient(model=resolved_model, endpoint=endpoint)
        )
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
    load_shared_local_llm_env()
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
    if _env_flag("LLM_EVENT_OPPORTUNISTIC_ENABLED", False):
        status["opportunistic"] = event_llm_opportunistic_gate_status()
    if not enabled:
        if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
            status["backend"] = "openai-compatible"
            status["device"] = "local-server"
            status["endpoint"] = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
                "LLM_EVENT_ENDPOINT",
                "http://127.0.0.1:11434/v1/chat/completions",
            )
            reachable, detail = _local_llm_reachable(status["endpoint"])
            status["available"] = reachable
            status["reason"] = (
                "LLM_EVENT_CLASSIFIER_ENABLED is false, but local LLM endpoint is reachable."
                if reachable
                else f"LLM disabled because local LLM endpoint is unavailable: {detail}"
            )
            return status
        status["reason"] = "LLM_EVENT_CLASSIFIER_ENABLED is false."
        return status
    if not model:
        status["reason"] = "LLM_EVENT_MODEL is not configured."
        return status
    if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
        status["backend"] = "openai-compatible"
        status["device"] = "local-server"
        endpoint = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
            "LLM_EVENT_ENDPOINT",
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        status["endpoint"] = endpoint
        reachable, detail = _local_llm_reachable(endpoint)
        if not reachable:
            status["reason"] = f"local LLM unavailable: {detail}"
            return status
        resolved_model, fallback, installed = _resolve_local_ollama_model(model, endpoint)
        status["configured_model"] = model
        status["model"] = resolved_model
        status["model_fallback_active"] = fallback
        status["installed_models"] = installed
        status["available"] = bool(resolved_model in installed) if installed else True
        status["reason"] = (
            None
            if status["available"]
            else f"configured local model is not installed: {model}"
        )
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
Choose only event labels directly supported by the text; do not add a secondary
label merely because it is often associated with the primary event.
AnalystUpgrade requires an explicit analyst rating or target-price change.
ProductLaunchPositive requires an explicit product launch or unveiling.
EarningsSurprisePositive requires reported earnings that explicitly beat expectations.
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


def _request_reasoning_effort() -> str | None:
    value = os.getenv("LLM_EVENT_REASONING_EFFORT", "").strip().lower()
    return value if value in {"none", "low", "medium", "high"} else None


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
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    # Weak local models (e.g. small quantized GGUF) often ignore the JSON
    # instruction and emit prose. Rather than discard the LLM signal entirely,
    # salvage the sentiment — the field that actually drives the ontology graph —
    # by taking the earliest sentiment keyword mentioned in the output.
    upper = text.upper()
    hits = sorted((upper.find(word), word) for word in ("POSITIVE", "NEGATIVE", "NEUTRAL") if word in upper)
    if hits:
        return {"sentiment": hits[0][1]}
    raise ValueError("LLM output contained neither a JSON object nor a sentiment keyword")


def _sentiment(value: str) -> SentimentDirection:
    normalized = value.strip().upper()
    if normalized == "POSITIVE":
        return SentimentDirection.POSITIVE
    if normalized == "NEGATIVE":
        return SentimentDirection.NEGATIVE
    return SentimentDirection.NEUTRAL


_LABEL_EVIDENCE_TERMS: dict[str, tuple[tuple[str, ...], ...]] = {
    "EarningsSurprisePositive": (
        ("earnings", "beat"), ("earnings", "above expectations"),
        ("실적", "예상 상회"), ("어닝", "서프라이즈"),
    ),
    "GuidanceLowered": (
        ("guidance", "lower"), ("guidance", "cut"),
        ("가이던스", "하향"), ("전망", "하향"),
    ),
    "MajorSupplyContract": (
        ("supply", "contract"), ("supply", "agreement"),
        ("공급", "계약"), ("납품", "계약"),
    ),
    "AnalystUpgrade": (
        ("analyst", "upgrade"), ("analyst", "target price"),
        ("brokerage", "upgrade"), ("증권사", "상향"),
        ("애널리스트", "상향"), ("목표가", "상향"),
    ),
    "LitigationRiskHigh": (
        ("litigation",), ("lawsuit",), ("legal action",),
        ("소송",), ("법적 조치",),
    ),
    "RegulatoryPenaltyNegative": (
        ("regulatory penalty",), ("regulator", "fine"),
        ("규제", "과징금"), ("규제", "벌금"),
    ),
    "ProductLaunchPositive": (
        ("product launch",), ("launched", "product"), ("unveil",),
        ("제품", "출시"), ("신제품", "공개"),
    ),
    "RumorRisk": (
        ("rumor",), ("unconfirmed",), ("speculation",),
        ("루머",), ("미확인",),
    ),
}


def _ground_event_labels(
    labels: tuple[str, ...],
    *,
    title: str,
    body: str,
) -> tuple[str, ...]:
    """Drop specific semantic labels unsupported by an explicit source phrase.

    Small local models are useful for extraction but sometimes append a label
    that is merely correlated with the event (a supply contract became an
    analyst upgrade during hardware validation). Those labels feed graph edges,
    so each known high-impact label needs direct lexical evidence in the source.
    Unknown future labels remain available rather than being silently erased.
    """
    text = f"{title}\n{body}".lower()
    grounded: list[str] = []
    for label in labels:
        evidence_groups = _LABEL_EVIDENCE_TERMS.get(label)
        if evidence_groups and not any(
            all(term.lower() in text for term in group)
            for group in evidence_groups
        ):
            continue
        grounded.append(label)
    return tuple(grounded)


def _normalize_llm_tickers(
    values: Any,
    known_tickers: dict[str, str],
) -> tuple[str, ...]:
    """Keep ticker-shaped identifiers and repair ``TICKER=company`` echoes."""
    known = {str(item).strip().upper() for item in known_tickers if str(item).strip()}
    result: list[str] = []
    for item in values if isinstance(values, (list, tuple)) else ():
        value = str(item).strip().upper()
        if "=" in value:
            prefix = value.split("=", 1)[0].strip()
            if prefix in known:
                value = prefix
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,19}", value):
            continue
        if value not in result:
            result.append(value)
    return tuple(result[:12])

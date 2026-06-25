from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.classifier import classify_text_event, source_now
from app.data.llm_classifier import (
    EmbeddedOpenVINOChatClient,
    EmbeddedMultimodalTransformersChatClient,
    EmbeddedTransformersChatClient,
    JsonEventLLMClassifier,
    LocalOpenAICompatibleChatClient,
    build_event_llm_classifier_from_env,
)
from app.graph import KnowledgeGraph
from app.graph.event_mapper import add_events_to_graph
from app.schemas.domain import EventType, SentimentDirection
from app.storage.local_store import LocalResearchStore


class FakeLLMClient:
    model = "fake-mini-llm"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.payload)


class LLMEventClassifierTest(unittest.TestCase):
    def test_llm_classification_overrides_keyword_fallback_and_extracts_facts(self) -> None:
        classifier = JsonEventLLMClassifier(
            FakeLLMClient(
                {
                    "sentiment": "POSITIVE",
                    "summary": "Company won a large supply contract.",
                    "key_facts": ["Contract amount is material", "Delivery starts next quarter"],
                    "event_labels": ["MajorSupplyContract"],
                    "companies": ["NVIDIA"],
                    "tickers": ["NVDA"],
                    "sectors": ["Semiconductor"],
                    "confidence": 0.91,
                }
            )
        )

        event = classify_text_event(
            title="NVIDIA announces customer update",
            body="The article uses neutral wording but describes a new customer agreement.",
            source=source_now("unit", "local://llm", "llm:1"),
            known_tickers={"NVDA": "NVIDIA"},
            llm_classifier=classifier,
        )

        self.assertEqual(event.sentiment, SentimentDirection.POSITIVE)
        self.assertEqual(event.tickers, ("NVDA",))
        self.assertIn("MajorSupplyContract", event.event_labels)
        self.assertIn("Contract amount is material", event.key_facts)
        self.assertEqual(event.classification_model, "fake-mini-llm")
        self.assertGreater(event.classification_confidence, 0.9)

    def test_llm_event_labels_are_mapped_into_graph(self) -> None:
        event = classify_text_event(
            title="AAPL faces penalty",
            body="Apple was fined by a regulator.",
            source=source_now("unit", "local://llm", "llm:2"),
            event_type=EventType.NEWS,
            known_tickers={"AAPL": "Apple"},
            llm_classifier=JsonEventLLMClassifier(
                FakeLLMClient(
                    {
                        "sentiment": "NEGATIVE",
                        "summary": "Apple faces a regulatory penalty.",
                        "key_facts": ["Regulator imposed a fine"],
                        "event_labels": ["RegulatoryPenaltyNegative"],
                        "companies": ["Apple"],
                        "tickers": ["AAPL"],
                        "sectors": ["Technology"],
                        "confidence": 0.84,
                    }
                )
            ),
        )

        graph = add_events_to_graph(KnowledgeGraph(), (event,))

        self.assertGreater(len(graph.matching(predicate="increasesRiskOf", object_="RegulatoryPenaltyNegative")), 0)
        self.assertGreater(len(graph.matching(predicate="generatesSemanticFeature")), 0)

    def test_llm_classification_survives_storage_roundtrip(self) -> None:
        event = classify_text_event(
            title="MSFT upgrade",
            body="Analysts raised the target price.",
            source=source_now("unit", "local://llm", "llm:3"),
            known_tickers={"MSFT": "Microsoft"},
            llm_classifier=JsonEventLLMClassifier(
                FakeLLMClient(
                    {
                        "sentiment": "POSITIVE",
                        "summary": "Analysts upgraded Microsoft.",
                        "key_facts": ["Target price raised"],
                        "event_labels": ["AnalystUpgrade", "TargetPriceRaised"],
                        "companies": ["Microsoft"],
                        "tickers": ["MSFT"],
                        "sectors": ["Technology"],
                        "confidence": 0.77,
                    }
                )
            ),
        )

        class Result:
            events = (event,)
            raw_records = ()
            market_snapshots = ()
            macro_metrics = ()

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalResearchStore(root=Path(tmp), retention_days=3650)
            store.save_research_result(Result())
            loaded = store.load().events[0]

        self.assertEqual(loaded.event_labels, ("AnalystUpgrade", "TargetPriceRaised"))
        self.assertEqual(loaded.key_facts, ("Target price raised",))
        self.assertEqual(loaded.classification_model, "fake-mini-llm")

    def test_local_llm_classifier_can_be_configured_without_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_EVENT_CLASSIFIER_ENABLED": "true",
                "LLM_EVENT_PROVIDER": "local",
                "LLM_EVENT_MODEL": "qwen2.5:1.5b-instruct",
                "LLM_EVENT_LOCAL_ENDPOINT": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=False,
        ):
            classifier = build_event_llm_classifier_from_env()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier.client, LocalOpenAICompatibleChatClient)
        self.assertEqual(classifier.client.model, "qwen2.5:1.5b-instruct")

    def test_embedded_local_model_classifier_can_be_configured_without_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_EVENT_CLASSIFIER_ENABLED": "true",
                "LLM_EVENT_PROVIDER": "embedded",
                "LLM_EVENT_MODEL": "models/local-llm/event-classifier",
                "LLM_EVENT_DEVICE": "cpu",
                "LLM_EVENT_LOCAL_FILES_ONLY": "true",
                "LLM_EVENT_MAX_NEW_TOKENS": "128",
            },
            clear=False,
        ):
            classifier = build_event_llm_classifier_from_env()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier.client, EmbeddedTransformersChatClient)
        self.assertEqual(classifier.client.model, "models/local-llm/event-classifier")
        self.assertEqual(classifier.client.device, "cpu")
        self.assertTrue(classifier.client.local_files_only)
        self.assertEqual(classifier.client.max_new_tokens, 128)

    def test_multimodal_local_model_classifier_can_be_configured_without_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_EVENT_CLASSIFIER_ENABLED": "true",
                "LLM_EVENT_PROVIDER": "multimodal",
                "LLM_EVENT_MODEL": "google/diffusiongemma-26B-A4B-it",
                "LLM_EVENT_DEVICE": "cpu",
                "LLM_EVENT_LOCAL_FILES_ONLY": "true",
                "LLM_EVENT_MAX_NEW_TOKENS": "128",
            },
            clear=False,
        ):
            classifier = build_event_llm_classifier_from_env()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier.client, EmbeddedMultimodalTransformersChatClient)
        self.assertEqual(classifier.client.model, "google/diffusiongemma-26B-A4B-it")
        self.assertEqual(classifier.client.device, "cpu")
        self.assertTrue(classifier.client.local_files_only)
        self.assertEqual(classifier.client.max_new_tokens, 128)

    def test_openvino_local_model_classifier_can_target_npu(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_EVENT_CLASSIFIER_ENABLED": "true",
                "LLM_EVENT_PROVIDER": "openvino-llm",
                "LLM_EVENT_MODEL": "models/local-llm/event-classifier",
                "LLM_EVENT_DEVICE": "NPU",
                "LLM_EVENT_LOCAL_FILES_ONLY": "true",
            },
            clear=False,
        ):
            classifier = build_event_llm_classifier_from_env()

        self.assertIsNotNone(classifier)
        self.assertIsInstance(classifier.client, EmbeddedOpenVINOChatClient)
        self.assertEqual(classifier.client.device, "NPU")


if __name__ == "__main__":
    unittest.main()

"""Read-only data collectors."""

from app.data.classifier import classify_text_event
from app.data.llm_classifier import (
    EmbeddedOpenVINOChatClient,
    EmbeddedMultimodalTransformersChatClient,
    EmbeddedTransformersChatClient,
    JsonEventLLMClassifier,
    LLMTextClient,
    LocalOpenAICompatibleChatClient,
    OpenAICompatibleChatClient,
    build_event_llm_classifier_from_env,
)
from app.data.public_collectors import (
    AlphaVantageDailyMarketDataCollector,
    DynamicPageCollector,
    EcosMacroCollector,
    FredMacroCollector,
    HtmlResearchCollector,
    OpenDartDisclosureCollector,
    RssNewsCollector,
    StooqMarketDataCollector,
    YahooChartMarketDataCollector,
    extract_focus_sections,
)
from app.data.raw_archive import RawArchive

__all__ = [
    "DynamicPageCollector",
    "AlphaVantageDailyMarketDataCollector",
    "EmbeddedOpenVINOChatClient",
    "EmbeddedMultimodalTransformersChatClient",
    "EmbeddedTransformersChatClient",
    "EcosMacroCollector",
    "FredMacroCollector",
    "HtmlResearchCollector",
    "JsonEventLLMClassifier",
    "LLMTextClient",
    "LocalOpenAICompatibleChatClient",
    "OpenDartDisclosureCollector",
    "OpenAICompatibleChatClient",
    "RawArchive",
    "RssNewsCollector",
    "StooqMarketDataCollector",
    "YahooChartMarketDataCollector",
    "classify_text_event",
    "build_event_llm_classifier_from_env",
    "extract_focus_sections",
]

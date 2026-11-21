"""Kitten-ASR -- lightweight speech-to-text."""

from kittenml.kittenasr.audio import SUPPORTED_LANGUAGES
from kittenml.kittenasr.enriched import (
    plain_text,
    postprocess_enriched_text,
    reference_context,
)

__all__ = [
    "get_model",
    "KittenASR",
    "Transcription",
    "SUPPORTED_LANGUAGES",
    "register_auto_classes",
    "plain_text",
    "postprocess_enriched_text",
    "reference_context",
]


def __getattr__(name):
    if name in {"get_model", "KittenASR", "Transcription"}:
        from kittenml.kittenasr.get_model import KittenASR, Transcription, get_model

        return {"get_model": get_model, "KittenASR": KittenASR, "Transcription": Transcription}[name]
    if name == "register_auto_classes":
        from kittenml.kittenasr.auto import register_auto_classes

        return register_auto_classes
    raise AttributeError(f"module 'kittenml.kittenasr' has no attribute {name!r}")

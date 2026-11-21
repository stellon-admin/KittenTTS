"""Prompting and output parsing for the enriched Kitten-ASR checkpoints.

Enriched checkpoints (e.g. `KittenML/kitten-asr-small-enhanced`) are trained to
emit inline markup alongside the words:

    [Happy]Congratulations were (poured) in upon her.[/Happy] [Accent='british']

`[Tag]...[/Tag]` spans carry emotion and delivery, standalone `[tag]` markers
carry non-verbal events and pauses, and nested parentheses mark emphasis. They
also expect a specific conditioning prompt, built by `reference_context`.

Adapted from the reference inference script for KitenASR-Enriched.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

NEUTRAL = "Neutral"

EMOTION_TAGS = {
    "amusement", "happiness", "excitement", "tenderness", "sadness", "anger",
    "frustration", "fear", "anxiety", "disgust", "contempt", "surprise",
    "embarrassment", "happy", "sad", "angry", "annoyed", "fearful",
    "disgusted", "surprised",
}
DELIVERY_TAGS = {
    "shout", "raised_voice", "whisper", "lowered_voice", "sarcastic",
    "deadpan", "sing_song", "laughing_voice", "crying_voice", "breathy",
    "strained", "hesitant", "rushed", "slow", "drawn_out", "calm",
    "laughing", "normal", "shouting", "whispering",
}
NON_VERBAL_EVENTS = {
    "breath", "chuckle", "cough", "cry", "gasp", "groan", "hmm", "laugh",
    "other_vocalization", "sigh", "sniff", "sob", "throat_clear", "ugh",
    "yawn",
}

TAG_RE = re.compile(r"\[(/?)([^\[\]=]+)(?:=([^\]]+))?\]")
ACCENT_RE = re.compile(r"\[accent\s*=\s*(['\"]?)(.*?)\1\]", re.IGNORECASE)
WORD_PATTERN = r"[^\W_]+(?:['’][^\W_]+)*"
TOKEN_RE = re.compile(rf"\[/?[^\[\]]+\]|\({{1,3}}|\){{1,3}}|{WORD_PATTERN}", re.UNICODE)


def reference_context(transcript: Optional[str] = None) -> str:
    """Build the conditioning prompt these checkpoints were fine-tuned with.

    Args:
        transcript: Optional plain-text transcript to condition decoding on.
    """
    reference = transcript.strip() if transcript else ""
    return f"Reference Transcript: {reference}\nEnriched Transcript:"


def parse_accent(enriched_text: str) -> Optional[str]:
    """Pull the accent out of an `[Accent='...']` marker, if the model emitted one."""
    match = ACCENT_RE.search(enriched_text or "")
    return match.group(2).strip() if match else None


def plain_text(enriched_text: str) -> str:
    """Strip the enrichment markup, keeping readable punctuation."""
    text = TAG_RE.sub(
        lambda match: " " if _category(match.group(2)) == "event" else "",
        enriched_text or "",
    )
    text = re.sub(r"\({1,3}|\){1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def postprocess_enriched_text(enriched_text: str) -> Dict[str, Any]:
    """Parse inline markup into per-word annotations and clip-level spans.

    Returns a dict with `plain_text`, `word_level_annotations`, `emotions`, and
    `delivery`. Standalone events get their own annotation with a null word.
    """
    words: List[Dict[str, Any]] = []
    spans: Dict[str, List[Dict[str, Any]]] = {"emotion": [], "delivery": []}
    active: Dict[str, List[Tuple[str, int]]] = {"emotion": [], "delivery": []}
    emphasis: List[int] = []

    def current_value(category: str) -> str:
        return active[category][-1][0] if active[category] else NEUTRAL

    def close_span(category: str, tag: str, end: int) -> None:
        normalized = _display_tag(tag)
        stack = active[category]
        for position in range(len(stack) - 1, -1, -1):
            label, start = stack[position]
            if label == normalized:
                del stack[position]
                spans[category].append({
                    "tag": label,
                    "segment": plain_text(enriched_text[start:end]),
                })
                return

    for match in TOKEN_RE.finditer(enriched_text or ""):
        token = match.group()
        tag_match = TAG_RE.fullmatch(token)
        if tag_match:
            closing, raw_tag, _ = tag_match.groups()
            category = _category(raw_tag)
            if category == "event":
                if not closing:
                    words.append({
                        "index": len(words),
                        "word": None,
                        "nv-event": raw_tag.casefold(),
                        "emotion": current_value("emotion"),
                        "delivery": current_value("delivery"),
                        "emphasis": max(emphasis, default=0),
                    })
            elif category in active:
                if closing:
                    close_span(category, raw_tag, match.start())
                else:
                    active[category].append((_display_tag(raw_tag), match.end()))
            continue
        if token.startswith("("):
            emphasis.append(min(len(token), 3))
            continue
        if token.startswith(")"):
            if emphasis:
                emphasis.pop()
            continue

        words.append({
            "index": len(words),
            "word": token,
            "nv-event": None,
            "emotion": current_value("emotion"),
            "delivery": current_value("delivery"),
            "emphasis": max(emphasis, default=0),
        })

    # Treat a missing closing boundary as a span running to the final word;
    # generation can stop just before the closing tag.
    for category, stack in active.items():
        for label, start in stack:
            spans[category].append({
                "tag": label,
                "segment": plain_text(enriched_text[start:]),
            })

    return {
        "plain_text": plain_text(enriched_text or ""),
        "word_level_annotations": words,
        "emotions": spans["emotion"],
        "delivery": spans["delivery"],
    }


def _display_tag(tag: str) -> str:
    return tag.casefold().replace("_", " ").title()


def _category(tag: str) -> Optional[str]:
    normalized = tag.casefold()
    if normalized in EMOTION_TAGS:
        return "emotion"
    if normalized in DELIVERY_TAGS:
        return "delivery"
    if normalized in NON_VERBAL_EVENTS or normalized.startswith("pause_"):
        return "event"
    return None

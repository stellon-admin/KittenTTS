"""High-level speech-to-text interface for Kitten-ASR."""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .audio import (
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)
from .enriched import parse_accent, postprocess_enriched_text, reference_context
from .quantization import find_quantized_files, load_quantized_state_dict

DEFAULT_MODEL = "KittenML/kitten-asr-tiny"

# Checkpoints whose name contains one of these emit enriched markup and need the
# prompting and parsing in `enriched.py`.
ENRICHED_MARKERS = ("enhanced", "enriched")

# Audio longer than this is split on a low-energy boundary before inference.
# The model accepts far longer inputs; this default keeps memory use within
# reach of a CPU-only machine.
DEFAULT_MAX_CHUNK_SECONDS = 300.0


def _resolve_model_path(model_name):
    """Resolve a model name to a local directory or a Hugging Face repo id."""
    if os.path.isdir(model_name):
        return model_name
    return model_name if "/" in model_name else f"KittenML/{model_name}"


def _is_enriched_model(model_name):
    """Guess whether a checkpoint emits enriched markup, from its name."""
    name = os.path.basename(str(model_name).rstrip("/\\")).casefold()
    return any(marker in name for marker in ENRICHED_MARKERS)


def _load_quantized_model(model_path, quantized_files, dtype, cache_dir):
    """Build the model from config and fill it with unpacked quantized weights."""
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import no_init_weights

    config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
    # Every weight is about to be overwritten, so skip the random initialization.
    # Anything the checkpoint fails to supply is caught below, not left as noise.
    with no_init_weights():
        model = AutoModel.from_config(config).to(dtype)

    state = load_quantized_state_dict(*quantized_files, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"Quantized checkpoint {model_path!r} is missing {len(missing)} weights, "
            f"first: {missing[:3]}. Loading it would leave those layers randomly "
            "initialized."
        )
    if unexpected:
        warnings.warn(
            f"Quantized checkpoint {model_path!r} carries {len(unexpected)} weights "
            f"the model does not use, first: {unexpected[:3]}.",
            stacklevel=3,
        )
    return model


def _require_torch():
    """Import torch, pointing at the right install command when it is missing."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Kitten-ASR needs PyTorch and transformers. "
            "Install them with: pip install 'torch>=2.6' 'transformers>=4.55'"
        ) from exc
    return torch


@dataclass
class Transcription:
    """One transcription result.

    Attributes:
        text: The transcribed text, with any enrichment markup stripped.
        language: Detected (or forced) language, e.g. "English". Comma separated
            when a long recording switched language between chunks. Empty when
            the audio is silent or the language could not be determined.

    The remaining attributes are populated only by enriched checkpoints:

        enriched_text: The raw output, markup included.
        accent: Accent the model reported, if any.
        emotions: Clip-level emotion spans, as `{"tag", "segment"}` dicts.
        delivery: Clip-level delivery spans, in the same shape.
        word_annotations: Per-word emotion, delivery, emphasis, and non-verbal
            events.
        reference_transcript: The transcript decoding was conditioned on.
    """

    text: str
    language: str = ""
    enriched_text: str = ""
    accent: Optional[str] = None
    emotions: List[Dict[str, Any]] = field(default_factory=list)
    delivery: List[Dict[str, Any]] = field(default_factory=list)
    word_annotations: List[Dict[str, Any]] = field(default_factory=list)
    reference_transcript: Optional[str] = None

    def __str__(self):
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the enriched-transcription JSON schema."""
        return {
            "lang": self.language,
            "accent": self.accent,
            "plain_text": self.text,
            "reference_transcript_provided": self.reference_transcript is not None,
            "reference_transcript": self.reference_transcript,
            "enrich_text": self.enriched_text,
            "word_level_annotations": self.word_annotations,
            "emotions": self.emotions,
            "delivery": self.delivery,
        }


class KittenASR:
    """Main Kitten-ASR class for speech-to-text transcription."""

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        cache_dir=None,
        device=None,
        dtype=None,
        max_new_tokens=448,
        max_batch_size=8,
        enriched=None,
    ):
        """Initialize Kitten-ASR with a model from Hugging Face or a local directory.

        Args:
            model_name: Hugging Face repository ID, model name, or a path to a
                local model directory.
            cache_dir: Directory to cache downloaded files.
            device: Torch device to run on ("cuda", "cpu", ...). Defaults to
                CUDA when available.
            dtype: Torch dtype for the weights. Defaults to bfloat16 on CUDA and
                float32 on CPU.
            max_new_tokens: Generation cap per audio chunk.
            max_batch_size: How many chunks to run through the model at once.
            enriched: Whether this checkpoint emits enriched markup (emotion,
                delivery, emphasis). Inferred from the model name by default.
        """
        torch = _require_torch()
        from transformers import AutoProcessor

        from .auto import register_auto_classes

        register_auto_classes()

        model_path = _resolve_model_path(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        quantized = find_quantized_files(model_path, cache_dir=cache_dir)
        self.quantized = quantized is not None
        if self.quantized:
            model = _load_quantized_model(model_path, quantized, dtype, cache_dir)
        else:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_path, dtype=dtype, cache_dir=cache_dir)

        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=cache_dir, fix_mistral_regex=True
        )
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.enriched = _is_enriched_model(model_path) if enriched is None else bool(enriched)

    @property
    def supported_languages(self) -> List[str]:
        """Language names accepted by the `language` argument.

        This is the full list the architecture understands. A given checkpoint
        may transcribe only a subset of it well.
        """
        return list(SUPPORTED_LANGUAGES)

    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        max_chunk_sec: float = DEFAULT_MAX_CHUNK_SECONDS,
    ) -> Union[str, List[str]]:
        """Transcribe audio to text.

        Args:
            audio: A file path, URL, base64 audio string, a (waveform, sample_rate)
                tuple, or a list of any of those.
            context: Optional prompt biasing the transcription (e.g. a list of
                names or jargon likely to appear). Broadcast over a batch. On an
                enriched checkpoint this is the reference transcript instead,
                and is wrapped in the prompt that checkpoint expects.
            language: Force the output language instead of letting the model
                detect it. Must be one of `supported_languages`.
            max_chunk_sec: Longer audio is split into chunks of at most this
                many seconds.

        Returns:
            The transcribed text, or a list of strings when `audio` is a list.
        """
        results = self.transcribe_detailed(
            audio, context=context, language=language, max_chunk_sec=max_chunk_sec
        )
        texts = [r.text for r in results]
        return texts if isinstance(audio, list) else texts[0]

    def transcribe_detailed(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        max_chunk_sec: float = DEFAULT_MAX_CHUNK_SECONDS,
    ) -> List[Transcription]:
        """Transcribe audio and return text plus detected language.

        Takes the same arguments as `transcribe`, and always returns a list with
        one `Transcription` per input audio.
        """
        torch = _require_torch()

        wavs = normalize_audios(audio)
        n = len(wavs)

        contexts = self._broadcast(context, n, "context")
        languages = [self._normalize_language(lang) for lang in self._broadcast(language, n, "language")]

        # Split long audio, remembering which input each chunk came from.
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=max_chunk_sec)
            for j, (chunk_wav, offset_sec) in enumerate(parts):
                chunks.append(
                    AudioChunk(orig_index=i, chunk_index=j, wav=chunk_wav, sr=SAMPLE_RATE, offset_sec=offset_sec)
                )

        with torch.no_grad():
            raw_outputs = self._generate(
                [contexts[c.orig_index] for c in chunks],
                [c.wav for c in chunks],
                [languages[c.orig_index] for c in chunks],
            )

        # Stitch the per-chunk outputs back into one result per input audio.
        texts: List[List[str]] = [[] for _ in range(n)]
        langs: List[List[str]] = [[] for _ in range(n)]
        for chunk, raw in zip(chunks, raw_outputs):
            lang, text = parse_asr_output(raw, user_language=languages[chunk.orig_index])
            texts[chunk.orig_index].append(text)
            langs[chunk.orig_index].append(lang)

        return [
            self._build_result(
                text="".join(t for t in texts[i] if t),
                language=merge_languages(langs[i]),
                reference_transcript=contexts[i],
            )
            for i in range(n)
        ]

    def _build_result(self, text: str, language: str, reference_transcript: str) -> Transcription:
        """Wrap one stitched output, unpacking the markup for enriched models."""
        if not self.enriched:
            return Transcription(text=text, language=language)

        parsed = postprocess_enriched_text(text)
        return Transcription(
            text=parsed["plain_text"],
            language=language,
            enriched_text=text,
            accent=parse_accent(text),
            emotions=parsed["emotions"],
            delivery=parsed["delivery"],
            word_annotations=parsed["word_level_annotations"],
            reference_transcript=reference_transcript or None,
        )

    def transcribe_file(self, audio_path: str, **kwargs) -> str:
        """Transcribe a single audio file. Alias for `transcribe`."""
        return self.transcribe(audio_path, **kwargs)

    def _generate(self, contexts, wavs, languages) -> List[str]:
        """Run the model over a list of chunks and return the raw decoded strings."""
        prompts = [self._build_prompt(c, lang) for c, lang in zip(contexts, languages)]
        batch_size = self.max_batch_size if self.max_batch_size and self.max_batch_size > 0 else len(prompts)

        outputs: List[str] = []
        for i in range(0, len(prompts), batch_size):
            inputs = self.processor(
                text=prompts[i : i + batch_size],
                audio=wavs[i : i + batch_size],
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device).to(self.dtype)

            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            outputs.extend(
                self.processor.batch_decode(
                    generated.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return outputs

    def _build_prompt(self, context: str, language: Optional[str]) -> str:
        """Build the chat-template prompt for one chunk."""
        if self.enriched:
            # These checkpoints were fine-tuned behind a fixed prompt, which
            # carries the reference transcript rather than a free-form context.
            context = reference_context(context)
        messages = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            # Pre-filling the language makes the model emit the transcription only.
            prompt = prompt + f"language {language}<asr_text>"
        return prompt

    @staticmethod
    def _broadcast(value: Any, n: int, name: str) -> List[Any]:
        values = value if isinstance(value, list) else [value]
        if len(values) == 1 and n > 1:
            values = values * n
        if len(values) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, {name}={len(values)}")
        return values

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if language is None or str(language).strip() == "":
            return None
        normalized = normalize_language_name(str(language))
        validate_language(normalized)
        return normalized


def get_model(repo_id=DEFAULT_MODEL, cache_dir=None, **kwargs):
    """Get a Kitten-ASR model."""
    return KittenASR(repo_id, cache_dir=cache_dir, **kwargs)

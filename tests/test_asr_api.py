"""Public API coverage without downloading model weights."""

import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from kittenml import KittenASR


def make_model(outputs, enriched=False):
    model = KittenASR.__new__(KittenASR)
    model.enriched = enriched
    model._generate = Mock(return_value=outputs)
    return model


def audio():
    return np.zeros(16000, dtype=np.float32), 16000


def test_root_import_keeps_architecture_lazy():
    subprocess.run(
        [sys.executable, "-c", "import sys; import kittenml; "
         "assert 'kittenml.kittenasr' not in sys.modules; "
         "from kittenml import KittenASR; "
         "assert 'kittenml.kittenasr.modeling' not in sys.modules"],
        check=True,
    )


def test_single_and_batch_return_types():
    model = make_model(["language English<asr_text>Hello."])
    assert model.transcribe(audio()) == "Hello."
    assert model.transcribe([audio()]) == ["Hello."]


def test_batch_broadcasts_context_and_normalizes_language():
    model = make_model(["Hello.", "Bonjour."])
    results = model.transcribe_detailed(
        [audio(), audio()], context="Names", language=["english", "french"]
    )
    assert [(r.text, r.language) for r in results] == [
        ("Hello.", "English"), ("Bonjour.", "French")
    ]
    contexts, wavs, languages = model._generate.call_args.args
    assert contexts == ["Names", "Names"]
    assert len(wavs) == 2
    assert languages == ["English", "French"]


def test_batch_mismatch_fails_before_generation():
    model = make_model([])
    with pytest.raises(ValueError, match="Batch size mismatch"):
        model.transcribe([audio(), audio()], context=["a", "b", "c"])
    model._generate.assert_not_called()


def test_enriched_transcription_is_parsed_and_serializable():
    model = make_model(
        ["language English<asr_text>[Happy]Hello (there).[/Happy] [Accent='british']"],
        enriched=True,
    )
    result = model.transcribe_detailed(audio(), context="Hello there.")[0]
    assert result.text == "Hello there."
    assert result.accent == "british"
    assert result.emotions == [{"tag": "Happy", "segment": "Hello there."}]
    assert result.to_dict()["reference_transcript"] == "Hello there."


def test_auto_registration_is_idempotent():
    from transformers import AutoConfig
    from kittenml.kittenasr.auto import register_auto_classes
    from kittenml.kittenasr.configuration import KittenASRConfig

    register_auto_classes()
    register_auto_classes()
    assert isinstance(AutoConfig.for_model("kitten_asr"), KittenASRConfig)

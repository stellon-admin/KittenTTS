"""Registration of the Kitten-ASR architecture with the transformers Auto classes."""

def register_auto_classes():
    """Make `AutoModel`/`AutoProcessor` able to load a `kitten_asr` checkpoint.

    Reassert the mappings on each Torch load: vLLM can register its compatible
    Qwen3-ASR config under the same checkpoint model_type in this process.
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor

    from .configuration import KittenASRConfig
    from .modeling import KittenASRForConditionalGeneration
    from .processing import KittenASRProcessor

    AutoConfig.register("kitten_asr", KittenASRConfig, exist_ok=True)
    AutoModel.register(KittenASRConfig, KittenASRForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(KittenASRConfig, KittenASRProcessor, exist_ok=True)

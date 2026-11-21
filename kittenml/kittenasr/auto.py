"""Registration of the Kitten-ASR architecture with the transformers Auto classes."""

_REGISTERED = False


def register_auto_classes():
    """Make `AutoModel`/`AutoProcessor` able to load a `kitten_asr` checkpoint.

    Safe to call more than once; the first call does the work.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoModel, AutoProcessor

    from .configuration import KittenASRConfig
    from .modeling import KittenASRForConditionalGeneration
    from .processing import KittenASRProcessor

    AutoConfig.register("kitten_asr", KittenASRConfig, exist_ok=True)
    AutoModel.register(KittenASRConfig, KittenASRForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(KittenASRConfig, KittenASRProcessor, exist_ok=True)

    _REGISTERED = True

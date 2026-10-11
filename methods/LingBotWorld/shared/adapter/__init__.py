"""LingBot-World conditioning and inference adapters."""

from .lingbot_v2 import (
    LingBotV2Adapter,
    decode_lingbot_video,
    encode_lingbot_condition,
    initialize_lingbot_frozen_models,
    to_runtime_config,
)

__all__ = [
    "LingBotDiffusionWrapper",
    "LingBotFastScheduler",
    "LingBotV2Adapter",
    "decode_lingbot_video",
    "encode_lingbot_condition",
    "initialize_lingbot_frozen_models",
    "to_runtime_config",
]


def __getattr__(name: str):
    if name == "LingBotFastScheduler":
        from .scheduler import LingBotFastScheduler

        return LingBotFastScheduler
    if name == "LingBotDiffusionWrapper":
        from .wrapper import LingBotDiffusionWrapper

        return LingBotDiffusionWrapper
    raise AttributeError(name)

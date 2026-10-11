"""Errors raised before a concrete model adapter is imported."""


class CoreError(RuntimeError):
    """Base class for user-facing orchestration errors."""


class RegistryError(CoreError):
    """A model manifest is absent, duplicated, or malformed."""


class UnknownModelError(RegistryError):
    """The requested model ID is not registered."""


class UnsupportedStageError(CoreError):
    """The requested stage is not part of the supported stage registry."""


class ConfigurationError(CoreError):
    """Configuration loading or validation failed."""


class AdapterError(CoreError):
    """A model adapter cannot be loaded or does not satisfy its contract."""


class CheckpointError(CoreError):
    """A checkpoint artifact does not satisfy the versioned contract."""


class LegacyDataError(CoreError):
    """A legacy latent LMDB does not satisfy the read contract."""

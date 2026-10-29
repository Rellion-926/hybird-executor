class HybridExecutorError(RuntimeError):
    """Base class for user-facing executor failures."""


class ConfigurationError(HybridExecutorError):
    """Raised when a required toolchain setting is invalid or missing."""


class ManifestError(HybridExecutorError):
    """Raised when a plugin manifest violates the public schema."""


class BuildError(HybridExecutorError):
    """Raised when an external build step fails."""

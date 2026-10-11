"""Model-family workspaces for Causal-Forcing.

The package intentionally keeps model-independent orchestration under
``methods.core``.  Importing :mod:`methods` never imports a concrete model
family; concrete adapters are resolved lazily from their manifests.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

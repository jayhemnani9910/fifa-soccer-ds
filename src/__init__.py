"""Top-level package for FIFA Soccer DS analytics pipeline."""

import os

# Ultralytics otherwise loads PyTorch checkpoints through unrestricted pickle.
# Deployments may opt out explicitly only for reviewed legacy checkpoints.
os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "true")

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analytics",
    "calib",
    "data",
    "detect",
    "graph",
    "live",
    "models",
    "track",
    "utils",
]

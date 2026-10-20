"""CV Backend implementations.

Provides pluggable computer-vision backends that all conform to the
``CVBackend`` Protocol. Choose one based on your deployment:

- ``MockBackend`` — no dependencies, deterministic fake results
- ``OpenCVBackend`` — local inference, requires opencv-python + numpy
- ``HTTPBackend`` — forwards to a remote CV service

Or implement your own and ``register()`` it on the default registry.
"""

from extraction.operators.backends.base import (
    CVBackend,
    CVBackendError,
    Detection,
    require_image_bytes,
)
from extraction.operators.backends.mock_backend import MockBackend

# Optional backends — import lazily so missing deps don't break the
# package import.
try:
    from extraction.operators.backends.opencv_backend import OpenCVBackend
except ImportError:  # pragma: no cover
    OpenCVBackend = None  # type: ignore[assignment]

try:
    from extraction.operators.backends.http_backend import HTTPBackend
except ImportError:  # pragma: no cover
    HTTPBackend = None  # type: ignore[assignment]

__all__ = [
    "CVBackend",
    "CVBackendError",
    "Detection",
    "MockBackend",
    "OpenCVBackend",
    "HTTPBackend",
    "require_image_bytes",
]

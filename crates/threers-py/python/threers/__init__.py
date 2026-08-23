"""threers — three.js-shaped 3D toolkit for Python."""

from __future__ import annotations

from threers._native import (
    Color,
    PerspectiveCamera,
    Scene,
    Vector3,
    version,
)

try:
    from threers._native import HeadlessRenderer
except ImportError:  # built without `headless`
    HeadlessRenderer = None  # type: ignore[misc, assignment]

try:
    from threers._native import Tween
except ImportError:
    Tween = None  # type: ignore[misc, assignment]

try:
    from threers._native import PhysicsWorld
except ImportError:
    PhysicsWorld = None  # type: ignore[misc, assignment]

__version__ = version()

__all__ = [
    "Color",
    "HeadlessRenderer",
    "PerspectiveCamera",
    "PhysicsWorld",
    "Scene",
    "Tween",
    "Vector3",
    "__version__",
    "version",
]

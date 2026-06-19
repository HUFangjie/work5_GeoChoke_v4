from __future__ import annotations

from typing import Any, Callable

ATTACK_REGISTRY: dict[str, Callable[[Any], Any]] = {}

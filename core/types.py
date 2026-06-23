from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from crypto.ciphertext_payload import EncryptedUpdate


@dataclass
class LocalUpdateRecord:
    client_id: int
    num_samples: int
    local_update: np.ndarray
    train_loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def clean_update(self) -> np.ndarray:
        """Backward-compatible alias for older non-DBA code paths."""
        return self.local_update


@dataclass
class ClientUpload:
    client_id: int
    num_samples: int
    profile_id: str
    encrypted_update: EncryptedUpdate
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_payload: Dict[str, Any] = field(default_factory=dict)

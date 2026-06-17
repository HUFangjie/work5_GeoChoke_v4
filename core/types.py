from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from crypto.ciphertext_payload import EncryptedUpdate


@dataclass
class LocalUpdateRecord:
    client_id: int
    num_samples: int
    clean_update: np.ndarray
    train_loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientUpload:
    client_id: int
    num_samples: int
    profile_id: str
    encrypted_update: EncryptedUpdate
    metadata: Dict[str, Any] = field(default_factory=dict)
    plaintext_reference: Optional[np.ndarray] = None

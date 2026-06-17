from __future__ import annotations

import numpy as np

from crypto.ciphertext_payload import EncryptedUpdate
from crypto.ckks_context_manager import CKKSContextManager


class AuthorizedDecryptionService:
    """Secret-key holder that decrypts aggregate ciphertexts only."""

    def __init__(self, context_manager: CKKSContextManager) -> None:
        self._context_manager = context_manager

    def decrypt_aggregate(self, aggregated_ciphertext: EncryptedUpdate, profile_id: str) -> np.ndarray:
        if aggregated_ciphertext.profile_id != profile_id:
            raise ValueError("profile mismatch for aggregate decryption")
        values: list[float] = []
        secret_key = self._context_manager.secret_key(profile_id)
        for chunk in aggregated_ciphertext.chunks:
            if chunk.valid_length <= 0:
                raise ValueError("ciphertext chunk valid length must be positive")
            if chunk.profile_id != profile_id:
                raise ValueError("ciphertext chunk profile mismatch during aggregate decryption")
            decrypted = chunk.payload.decrypt(secret_key)[: chunk.valid_length]
            values.extend(float(x) for x in decrypted)
        result = np.asarray(values, dtype=np.float64)
        if result.size != aggregated_ciphertext.total_dimension:
            raise ValueError("decrypted aggregate dimension mismatch")
        if not np.all(np.isfinite(result)):
            raise ValueError("decrypted aggregate contains non-finite values")
        return result

    def decrypt_for_offline_calibration(self, aggregated_ciphertext: EncryptedUpdate, profile_id: str) -> np.ndarray:
        """Offline-only calibration/validation decrypt path; never called by AggregationServer."""
        return self.decrypt_aggregate(aggregated_ciphertext, profile_id)

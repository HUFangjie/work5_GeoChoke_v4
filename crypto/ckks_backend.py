from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import tenseal as ts

from crypto.base import CryptoBackend
from crypto.ciphertext_payload import CiphertextChunk, EncryptedUpdate
from crypto.ckks_context_manager import PublicCKKSContextBundle
from utils.validation import ensure_finite_array


class CKKSBackend(CryptoBackend):
    """Public-key CKKS backend used by clients and the aggregation server."""

    def __init__(self, public_bundles: Mapping[str, PublicCKKSContextBundle]) -> None:
        self._public_bundles = dict(public_bundles)
        self.last_encryption_time = 0.0

    def initialize_profiles(self) -> None:
        if not self._public_bundles:
            raise ValueError("CKKSBackend requires initialized public context bundles")

    def get_public_context(self, profile_id: str) -> Any:
        return self._public_bundles[profile_id].public_context

    def get_profile_parameters(self, profile_id: str) -> Dict[str, Any]:
        return dict(self._public_bundles[profile_id].parameters)

    def slot_count(self, profile_id: str) -> int:
        return self._public_bundles[profile_id].slot_count

    def encrypt_update(self, flat_update: np.ndarray, profile_id: str) -> EncryptedUpdate:
        ensure_finite_array(flat_update, "flat_update")
        if profile_id not in self._public_bundles:
            raise KeyError(f"unknown CKKS profile {profile_id}")
        vector = np.asarray(flat_update, dtype=np.float64)
        slot_count = self.slot_count(profile_id)
        chunks: list[CiphertextChunk] = []
        start_time = time.perf_counter()
        for chunk_index, start in enumerate(range(0, vector.size, slot_count)):
            chunk = vector[start : start + slot_count]
            ciphertext = ts.ckks_vector(self.get_public_context(profile_id), chunk.tolist())
            chunks.append(
                CiphertextChunk(
                    chunk_index=chunk_index,
                    valid_length=int(chunk.size),
                    profile_id=profile_id,
                    total_dimension=int(vector.size),
                    payload=ciphertext,
                )
            )
        self.last_encryption_time = time.perf_counter() - start_time
        return EncryptedUpdate(chunks=chunks, profile_id=profile_id, total_dimension=int(vector.size))

    def multiply_plain(self, encrypted_update: EncryptedUpdate, weight: float) -> EncryptedUpdate:
        if not np.isfinite(weight):
            raise ValueError("aggregation weight must be finite")
        self._validate_encrypted_update(encrypted_update)
        chunks = [
            CiphertextChunk(
                chunk_index=chunk.chunk_index,
                valid_length=chunk.valid_length,
                profile_id=chunk.profile_id,
                total_dimension=chunk.total_dimension,
                payload=chunk.payload * float(weight),
            )
            for chunk in encrypted_update.chunks
        ]
        return EncryptedUpdate(chunks=chunks, profile_id=encrypted_update.profile_id, total_dimension=encrypted_update.total_dimension)

    def add_ciphertexts(self, ciphertexts: Iterable[EncryptedUpdate]) -> EncryptedUpdate:
        encrypted_updates = list(ciphertexts)
        if not encrypted_updates:
            raise ValueError("cannot aggregate an empty ciphertext list")
        first = encrypted_updates[0]
        self._validate_encrypted_update(first)
        for update in encrypted_updates[1:]:
            self._validate_compatible(first, update)
        aggregated_chunks: list[CiphertextChunk] = []
        for chunk_index in range(first.block_count):
            accumulator = encrypted_updates[0].chunks[chunk_index].payload
            for encrypted_update in encrypted_updates[1:]:
                accumulator = accumulator + encrypted_update.chunks[chunk_index].payload
            template = first.chunks[chunk_index]
            aggregated_chunks.append(
                CiphertextChunk(
                    chunk_index=chunk_index,
                    valid_length=template.valid_length,
                    profile_id=template.profile_id,
                    total_dimension=template.total_dimension,
                    payload=accumulator,
                )
            )
        return EncryptedUpdate(
            chunks=aggregated_chunks,
            profile_id=first.profile_id,
            total_dimension=first.total_dimension,
        )

    def serialize(self, ciphertext: EncryptedUpdate) -> Dict[str, Any]:
        self._validate_encrypted_update(ciphertext)
        return {
            "profile_id": ciphertext.profile_id,
            "total_dimension": ciphertext.total_dimension,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "valid_length": chunk.valid_length,
                    "profile_id": chunk.profile_id,
                    "total_dimension": chunk.total_dimension,
                    "payload": chunk.payload.serialize(),
                }
                for chunk in ciphertext.chunks
            ],
        }

    def deserialize(self, payload: Mapping[str, Any], profile_id: str) -> EncryptedUpdate:
        if payload["profile_id"] != profile_id:
            raise ValueError("serialized ciphertext profile does not match requested profile")
        context = self.get_public_context(profile_id)
        chunks = [
            CiphertextChunk(
                chunk_index=int(chunk["chunk_index"]),
                valid_length=int(chunk["valid_length"]),
                profile_id=profile_id,
                total_dimension=int(chunk["total_dimension"]),
                payload=ts.ckks_vector_from(context, chunk["payload"]),
            )
            for chunk in payload["chunks"]
        ]
        result = EncryptedUpdate(chunks=chunks, profile_id=profile_id, total_dimension=int(payload["total_dimension"]))
        self._validate_encrypted_update(result)
        return result

    def _validate_encrypted_update(self, update: EncryptedUpdate) -> None:
        if update.profile_id not in self._public_bundles:
            raise ValueError(f"unknown profile {update.profile_id}")
        if not update.chunks:
            raise ValueError("encrypted update has no chunks")
        expected_chunks = (update.total_dimension + self.slot_count(update.profile_id) - 1) // self.slot_count(update.profile_id)
        if update.block_count != expected_chunks:
            raise ValueError("ciphertext block count does not match total dimension and slot count")
        for expected_index, chunk in enumerate(update.chunks):
            if chunk.profile_id != update.profile_id:
                raise ValueError("ciphertext chunk profile mismatch")
            if chunk.chunk_index != expected_index:
                raise ValueError("ciphertext chunk index mismatch")
            if chunk.total_dimension != update.total_dimension:
                raise ValueError("ciphertext chunk total dimension mismatch")
            if chunk.valid_length <= 0 or chunk.valid_length > self.slot_count(update.profile_id):
                raise ValueError("invalid ciphertext chunk valid length")

    def _validate_compatible(self, left: EncryptedUpdate, right: EncryptedUpdate) -> None:
        self._validate_encrypted_update(right)
        if left.profile_id != right.profile_id:
            raise ValueError("profile mismatch during encrypted aggregation")
        if left.total_dimension != right.total_dimension:
            raise ValueError("total dimension mismatch during encrypted aggregation")
        if left.block_count != right.block_count:
            raise ValueError("chunk count mismatch during encrypted aggregation")
        for left_chunk, right_chunk in zip(left.chunks, right.chunks):
            if left_chunk.valid_length != right_chunk.valid_length:
                raise ValueError("chunk valid length mismatch during encrypted aggregation")

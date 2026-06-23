from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DefenseStrategy(ABC):
    @abstractmethod
    def initialize(self, model, crypto_backend, proxy_loader, decrypt_aggregate_fn=None): ...

    @abstractmethod
    def get_profile_for_round(self, round_id): ...

    @abstractmethod
    def after_aggregate(self, previous_model, candidate_model, current_profile_id, round_id): ...

    def prepare_client_upload(self, client_id: int, update_vector: Any, round_id: int, profile_id: str, num_samples: int, metadata: dict[str, Any] | None = None):
        return update_vector, {}

    def filter_uploads_before_aggregation(self, uploads: list[Any], round_id: int):
        return uploads, {}

    def use_uniform_aggregation_weights(self) -> bool:
        return False

    def unmask_aggregate_update(self, aggregate_update: Any, uploads: list[Any], round_id: int, weights: list[float]):
        return aggregate_update, {}

    def get_round_metrics(self, round_id: int) -> dict[str, Any]:
        return {}

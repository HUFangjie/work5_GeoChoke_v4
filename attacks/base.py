from abc import ABC,abstractmethod
class AttackStrategy(ABC):
    @abstractmethod
    def craft_update(self, client_id, clean_update, global_model, attacker_context): ...

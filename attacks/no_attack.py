from attacks.base import AttackStrategy
class NoAttack(AttackStrategy):
    def craft_update(self, client_id, clean_update, global_model, attacker_context): return clean_update.copy()

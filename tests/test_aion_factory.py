from config import ExperimentConfig
from defenses.aion.defense import AionDefense
from factories.defense_factory import create_defense


def test_aion_factory_registration():
    cfg = ExperimentConfig(defense_name="aion", aggregation="mean")
    defense = create_defense(cfg)
    assert isinstance(defense, AionDefense)
    assert defense.get_profile_for_round(0) == cfg.aion.profile_id

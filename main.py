import os

from config import CONFIG
from factories import defaults  # registers built-in components
from factories.attack_factory import create_attack
from factories.crypto_factory import create_crypto_backend
from factories.dataset_factory import create_dataset_provider
from factories.defense_factory import create_defense
from factories.model_factory import create_model_factory
from core.coordinator import FederatedCoordinator
from utils.logger import setup_logger
from utils.seed import set_seed


def _configure_experiment_output_dir() -> None:
    attack_mode = CONFIG.dba_attack_mode if CONFIG.attack_name == "dba" else "na"
    run_dir = f"attack={CONFIG.attack_name}_defense={CONFIG.defense_name}_mode={attack_mode}_seed={CONFIG.seed}"
    if os.path.basename(os.path.normpath(CONFIG.output_dir)) != run_dir:
        CONFIG.output_dir = os.path.join(CONFIG.output_dir, run_dir)


def main() -> None:
    _configure_experiment_output_dir()
    set_seed(CONFIG.seed)
    logger = setup_logger(CONFIG.log_level)
    dataset_provider = create_dataset_provider(CONFIG)
    model_factory = create_model_factory(CONFIG)
    crypto_bundle = create_crypto_backend(CONFIG)
    defense = create_defense(CONFIG, crypto_bundle)
    attack = create_attack(CONFIG)
    coordinator = FederatedCoordinator(
        cfg=CONFIG,
        dataset_provider=dataset_provider,
        model_factory=model_factory,
        crypto_backend=crypto_bundle.public_backend,
        decryption_service=crypto_bundle.decryption_service,
        defense=defense,
        attack=attack,
        logger=logger,
    )
    coordinator.run()


if __name__ == "__main__":
    main()

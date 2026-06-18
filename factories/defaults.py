from __future__ import annotations

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack, SignFlipScaledAttack
from attacks.no_attack import NoAttack
from core.decryption_service import AuthorizedDecryptionService
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager
from data.mnist import MNISTProvider
from defenses.geochoke.defense import GeoChokeDefense
from defenses.no_defense import NoDefense
from factories.attack_factory import ATTACK_REGISTRY
from factories.crypto_factory import CRYPTO_REGISTRY, CryptoBundle
from factories.dataset_factory import DATASET_REGISTRY
from factories.defense_factory import DEFENSE_REGISTRY
from factories.model_factory import MODEL_REGISTRY
from models.mnist_cnn import MNISTCNN


def register_defaults() -> None:
    """Idempotently register built-in components.

    This function is intentionally callable multiple times. It fixes cases where
    tests or notebooks mutate a registry after `factories.defaults` has already
    been imported: create_* factories call this function before lookup, so built-
    ins such as `mnist` and `mnist_cnn` are restored if missing.
    """

    DATASET_REGISTRY.setdefault("mnist", MNISTProvider)
    MODEL_REGISTRY.setdefault("mnist_cnn", _create_mnist_cnn)
    ATTACK_REGISTRY.setdefault("none", _create_no_attack)
    ATTACK_REGISTRY.setdefault("alie", _create_alie)
    ATTACK_REGISTRY.setdefault("fang_mean", _create_fang_mean)
    ATTACK_REGISTRY.setdefault("sign_flip_scaled", _create_sign_flip_scaled)
    CRYPTO_REGISTRY.setdefault("ckks", _create_ckks)
    DEFENSE_REGISTRY.setdefault("geochoke", _create_geochoke)
    DEFENSE_REGISTRY.setdefault("none", _create_no_defense)
    DEFENSE_REGISTRY.setdefault("no_defense", _create_no_defense)


def _create_mnist_cnn():
    return MNISTCNN()


def _create_no_attack(cfg):
    return NoAttack()


def _create_alie(cfg):
    return ALIEAttack(
        cfg.alie_z,
        oracle_all_updates=cfg.alie_oracle_all_updates,
        whitebox=cfg.attack_whitebox,
        oracle_mean_replacement=cfg.oracle_mean_replacement,
        whitebox_z=cfg.alie_whitebox_z,
        strength=cfg.alie_strength,
    )


def _create_fang_mean(cfg):
    return FangMeanAttack(
        cfg.aggregation,
        cfg.fang_max_norm,
        cfg.fang_search_steps,
        whitebox=cfg.attack_whitebox,
        target_scale=cfg.fang_target_scale,
        oracle_mean_replacement=cfg.oracle_mean_replacement,
    )


def _create_sign_flip_scaled(cfg):
    return SignFlipScaledAttack(
        cfg.aggregation,
        cfg.fang_max_norm,
        cfg.fang_search_steps,
        target_scale=cfg.fang_target_scale,
    )


def _create_ckks(cfg):
    manager = CKKSContextManager(cfg.ckks_profiles)
    manager.initialize()
    service = AuthorizedDecryptionService(manager)
    public_backend = CKKSBackend(manager.public_bundles())
    public_backend.initialize_profiles()
    return CryptoBundle(public_backend=public_backend, context_manager=manager, decryption_service=service)


def _create_geochoke(cfg):
    return GeoChokeDefense(cfg.geochoke, cfg.ckks_profiles, device=cfg.device, output_dir=f"{cfg.output_dir}/crypto_calibration")


def _create_no_defense(cfg):
    return NoDefense(cfg.geochoke.initial_profile_id)


register_defaults()

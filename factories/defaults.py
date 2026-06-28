from __future__ import annotations

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack
from attacks.dba import DBAAttack
from attacks.no_attack import NoAttack
from attacks.neurotoxin import NeurotoxinAttack
from attacks.a3fl import A3FLAttack
from attacks.three_dfed import ThreeDFedAttack
from attacks.adaptive_geochoke import AdaptiveGeoChokeAttack
from core.decryption_service import AuthorizedDecryptionService
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager
from data.mnist import MNISTProvider
from data.fashion_mnist import FashionMNISTProvider
from data.cifar10 import CIFAR10Provider
from defenses.geochoke.defense import GeoChokeDefense
from defenses.no_defense import NoDefense
from factories.registries import ATTACK_REGISTRY
from factories.crypto_factory import CRYPTO_REGISTRY, CryptoBundle
from factories.dataset_factory import DATASET_REGISTRY
from factories.defense_factory import DEFENSE_REGISTRY
from factories.model_factory import MODEL_REGISTRY
from models.mnist_cnn import MNISTCNN
from models.cifar10_cnn import CIFAR10CNN


def register_defaults() -> None:
    """Idempotently register built-in components.

    This function is intentionally callable multiple times. It fixes cases where
    tests or notebooks mutate a registry after `factories.defaults` has already
    been imported: create_* factories call this function before lookup, so built-
    ins such as `mnist` and `mnist_cnn` are restored if missing.
    """

    DATASET_REGISTRY.setdefault("mnist", MNISTProvider)
    DATASET_REGISTRY.setdefault("fashion_mnist", FashionMNISTProvider)
    DATASET_REGISTRY.setdefault("cifar10", CIFAR10Provider)
    MODEL_REGISTRY.setdefault("mnist_cnn", _create_mnist_cnn)
    MODEL_REGISTRY.setdefault("cifar10_cnn", _create_cifar10_cnn)
    ATTACK_REGISTRY.setdefault("none", _create_no_attack)
    ATTACK_REGISTRY.setdefault("alie", _create_alie)
    ATTACK_REGISTRY.setdefault("fang_mean", _create_fang_mean)
    ATTACK_REGISTRY.setdefault("dba", _create_dba)
    ATTACK_REGISTRY.setdefault("neurotoxin", _create_neurotoxin)
    ATTACK_REGISTRY.setdefault("a3fl", _create_a3fl)
    ATTACK_REGISTRY.setdefault("three_dfed", _create_three_dfed)
    ATTACK_REGISTRY.setdefault("adaptive_geochoke", _create_adaptive_geochoke)
    CRYPTO_REGISTRY.setdefault("ckks", _create_ckks)
    DEFENSE_REGISTRY.setdefault("geochoke", _create_geochoke)
    DEFENSE_REGISTRY.setdefault("none", _create_no_defense)
    DEFENSE_REGISTRY.setdefault("no_defense", _create_no_defense)


def _create_mnist_cnn():
    return MNISTCNN()


def _create_cifar10_cnn():
    return CIFAR10CNN()


def _create_no_attack(cfg):
    return NoAttack()


def _create_alie(cfg):
    return ALIEAttack(cfg.alie_z, oracle_all_updates=cfg.alie_oracle_all_updates)


def _create_fang_mean(cfg):
    return FangMeanAttack(cfg.aggregation, cfg.fang_max_norm, cfg.fang_search_steps)


def _create_dba(cfg):
    return DBAAttack(cfg)


def _create_neurotoxin(cfg):
    return NeurotoxinAttack(cfg)


def _create_a3fl(cfg):
    return A3FLAttack(cfg)


def _create_three_dfed(cfg):
    return ThreeDFedAttack(cfg)


def _create_adaptive_geochoke(cfg):
    return AdaptiveGeoChokeAttack(cfg)


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

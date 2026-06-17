from __future__ import annotations

from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack
from attacks.no_attack import NoAttack
from core.decryption_service import AuthorizedDecryptionService
from crypto.ckks_backend import CKKSBackend
from crypto.ckks_context_manager import CKKSContextManager
from data.mnist import MNISTProvider
from defenses.geochoke.defense import GeoChokeDefense
from defenses.no_defense import NoDefense
from factories.attack_factory import register_attack
from factories.crypto_factory import CryptoBundle, register_crypto
from factories.dataset_factory import register_dataset
from factories.defense_factory import register_defense
from factories.model_factory import register_model
from models.mnist_cnn import MNISTCNN

register_dataset("mnist")(MNISTProvider)

@register_model("mnist_cnn")
def _create_mnist_cnn():
    return MNISTCNN()

@register_attack("none")
def _create_no_attack(cfg):
    return NoAttack()

@register_attack("alie")
def _create_alie(cfg):
    return ALIEAttack(cfg.alie_z, oracle_all_updates=cfg.alie_oracle_all_updates)

@register_attack("fang_mean")
def _create_fang_mean(cfg):
    return FangMeanAttack(cfg.aggregation, cfg.fang_max_norm, cfg.fang_search_steps)

@register_crypto("ckks")
def _create_ckks(cfg):
    manager = CKKSContextManager(cfg.ckks_profiles)
    manager.initialize()
    service = AuthorizedDecryptionService(manager)
    public_backend = CKKSBackend(manager.public_bundles())
    public_backend.initialize_profiles()
    return CryptoBundle(public_backend=public_backend, context_manager=manager, decryption_service=service)

@register_defense("geochoke")
def _create_geochoke(cfg):
    return GeoChokeDefense(cfg.geochoke, cfg.ckks_profiles, device=cfg.device, output_dir=f"{cfg.output_dir}/crypto_calibration")

@register_defense("none")
def _create_no_defense(cfg):
    return NoDefense(cfg.geochoke.initial_profile_id)

@register_defense("no_defense")
def _create_named_no_defense(cfg):
    return NoDefense(cfg.geochoke.initial_profile_id)

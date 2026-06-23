# GeoChoke CKKS Secure Aggregation FL

A minimal but real single-server encrypted-aggregation federated learning project for MNIST. Clients train independent local model copies, apply optional plaintext attacks before encryption, encrypt model updates with TenSEAL CKKS, and the aggregation server performs only ciphertext-domain weighted mean aggregation. An independent authorized decryption service decrypts only the aggregate ciphertext. GeoChoke calibrates CKKS profiles with real CKKS residuals and selects the next-round profile from CFI control variables.

## Directory tree

```text
.
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── core/
│   ├── coordinator.py
│   ├── client.py
│   ├── server.py
│   ├── decryption_service.py
│   ├── local_trainer.py
│   └── types.py
├── crypto/
│   ├── base.py
│   ├── ckks_backend.py
│   ├── ckks_context_manager.py
│   ├── ciphertext_payload.py
│   └── update_codec.py
├── defenses/
│   ├── base.py
│   └── geochoke/
│       ├── defense.py
│       ├── calibration.py
│       ├── cfi_estimator.py
│       └── controller.py
├── attacks/
│   ├── base.py
│   ├── no_attack.py
│   ├── alie.py
│   └── fang.py
├── data/
│   ├── base.py
│   ├── mnist.py
│   └── partition.py
├── models/
│   ├── registry.py
│   └── mnist_cnn.py
├── evaluation/
│   ├── metrics.py
│   └── evaluator.py
├── utils/
│   ├── logger.py
│   ├── seed.py
│   └── validation.py
└── tests/
    ├── test_update_codec.py
    ├── test_ckks_roundtrip.py
    ├── test_encrypted_aggregation.py
    ├── test_security_boundary.py
    ├── test_attacks.py
    ├── test_geochoke_controller.py
    └── test_end_to_end.py
```

## Security boundary

This is a CKKS-based single-server encrypted aggregation system. It relies on an independent `AuthorizedDecryptionService` that does not collude with the `AggregationServer`. It is **not** threshold CKKS and **not** Bonawitz masked SecAgg.

Key separation is explicit: `CKKSContextManager` owns secret contexts only on the authorized decryption side, exports public-only bundles, and `CKKSBackend` stores only those public bundles. Clients and the aggregation server receive the public backend only. The decryption service exposes only:

```python
decrypt_aggregate(aggregated_ciphertext, profile_id)
```

It intentionally does not expose `decrypt_client_update(...)` or `decrypt_ciphertext_list(...)`.

## Implemented workflow

1. Server samples clients and broadcasts the global model plus the current profile ID.
2. Each client trains an independent model copy on its own MNIST split.
3. Selected clients compute clean updates; malicious selected clients craft ALIE or FangMean updates before encryption.
4. Every selected client encrypts chunks with real TenSEAL CKKS under the public context.
5. The server computes weighted mean using CKKS scalar multiplication and ciphertext addition only.
6. The authorized service decrypts only the aggregate ciphertext.
7. The server applies the approximate aggregate update to a candidate model.
8. GeoChoke compares previous and candidate CFI using the fixed proxy set and fixed perturbation bank, then selects the next-round profile.
9. The candidate model is accepted as the next global model.

## Fang Mean adaptation

Original Fang attacks are usually framed against robust aggregators such as Krum or trimmed mean. This project's aggregator is CKKS weighted mean, so `FangMeanAttack` performs a mean-compatible bounded search that maximizes aggregate deviation from the observed benign center under a norm budget. Krum/TrimmedMean-specific Fang variants must reject `aggregation="weighted_mean"` rather than silently running the wrong objective.

## Configuration

All experiment settings live in `config.py`; there are no YAML/JSON/Hydra configuration files. The default enables a DBA MNIST run with four malicious clients, explicit GeoChoke reference-profile calibration, and three CKKS profiles.

For a fuller experiment, edit `config.py` and increase values such as:

```python
quick_data_limit = 6000
num_rounds = 30
clients_per_round = 5
attack_name = "fang_mean"  # or "alie" / "dba" / "none"
```

## Installation and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
pytest -q
```

## Outputs

The coordinator logs each round's test accuracy and writes:

- `client_partitions.csv`: sample count, label distribution, malicious flag per client.
- `fl_ckks_geochoke_metrics.csv`: unified per-round FL, CKKS, and GeoChoke metrics.
- `attack_metrics.csv`: per-client attack norm/cosine/time metrics.
- `crypto_calibration/direct_profile_calibration.csv`: per-profile direct CKKS residual MSE, max residual, relative L2, residual mean, and residual standard deviation.
- `crypto_calibration/aggregation_pipeline_validation.csv`: full Enc → plaintext-weight multiply → ciphertext addition → aggregate decrypt validation metrics, including norm ratio.
- `crypto_calibration/reference_residual_bank.npz`: direct residual samples separated by calibrated profile.
- `crypto_calibration/calibration_tensor_sources.csv`: representative calibration tensor source and statistic records.
- `crypto_calibration/perturbation_bank.csv`: fixed protocol-aligned reference perturbation bank with selected ξ_m norms, source profile, and SHA-256 hashes.

Plaintext aggregate reference metrics are optional experiment-only validation fields computed by the coordinator-side experiment verifier and are never used to update the global model.


## Component registry and extension

Built-in components are registered in `factories/defaults.py`. The coordinator depends on injected dataset, model factory, crypto backend, decryption service, defense, and attack objects; it does not import MNIST, CKKS, GeoChoke, or concrete attacks. To switch components, register a new implementation with `register_dataset`, `register_model`, `register_crypto`, `register_defense`, or `register_attack`, then change only the corresponding name in `config.py` (`dataset_name`, `model_name`, `crypto_backend_name`, `defense_name` (`"geochoke"`, `"none"`, or another registered defense), `attack_name`). Unknown names raise `ValueError` rather than silently falling back.

### No-defense mode

Set `defense_name = "none"` in `config.py` to disable GeoChoke profile adaptation while keeping the encrypted aggregation pipeline unchanged. `NoDefense` always returns the configured initial CKKS profile and records `defense_enabled=False` in round metrics.

## Aion single-mask secure aggregation simulator

This repository also includes an optional `defense_name = "aion"` mode that simulates the protocol logic of **Aion: Robust and Efficient Multi-Round Single-Mask Secure Aggregation Against Malicious Participants** inside this single-process FL testbed. Aion is separate from GeoChoke: it does not use CFI, CKKS profile control, tangent commitment, patch detection, or any client-level unmasked update inspection.

To run Aion, keep configuration in `config.py` and select the Aion example configuration or set the fields manually:

```python
from config import aion_config
CONFIG = aion_config()
```

or:

```python
defense_name = "aion"
attack_name = "dba"
aggregation = "mean"
aion.reconstruction_mode = "amr"
```

Aion's HPRF masks are additively homomorphic only for uniform sums/means. In Aion mode, the server uses uniform mean weights. If `aggregation="weighted_mean"` is used with unequal participant sample counts, the run raises `ValueError` instead of applying sample-count weights to masks.

Implemented Aion components:

- Feldman VSS over Shamir shares in a 2048-bit RFC 3526 safe-prime subgroup.
- Deterministic key-homomorphic PRF masks derived from SHAKE256 public coefficients.
- AMR (Aggregated Mask Reconstruction) as the default reconstruction path; ASR is intentionally rejected in strict mode unless CCS/disjoint client scheduling is implemented because it exposes the aggregated secret across overlapping rounds.
- Masked Gradient Filtering (MGF) over masked update payload statistics, with evolving bounds.
- Dynamic Mask Coverage / Dynamic Mask Removal style extra digits and post-unmask rounding.
- A local BFT commitment simulator that checks `n >= 3f + 1`, commits online/valid sets and a model hash, and simulates aggregator HMAC signatures.

Security boundary: this Aion integration is a **single-process protocol-faithful simulator**, not a distributed multi-aggregator network deployment. It preserves the project’s CKKS wrapper for transport and server aggregate decryption service, but the Aion defense logic treats uploaded values as masked updates and reconstructs only the aggregated mask. The server must not use sample-weighted masks, must not access unmasked individual client updates, and must not rely on GeoChoke calibration when `defense_name="aion"`.

Aion writes protocol artifacts under `outputs/.../aion/`:

- `aion_protocol_metrics.csv`
- `aion_vss_init_summary.json`
- `aion_hprf_validation.json`

Important per-round CSV/log fields include `secure_aggregation_scheme`, `aion_reconstruction_mode`, `aion_valid_client_count`, `aion_filtered_client_count`, `aion_filter_rate`, `aion_bound`, `aion_alpha`, `aion_unmasked_aggregate_norm`, `aion_aggregated_mask_norm`, `aion_hprf_additivity_max_error`, `aion_bft_quorum_size`, and `aion_bft_required_quorum`.

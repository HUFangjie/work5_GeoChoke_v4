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

All experiment settings live in `config.py`; there are no YAML/JSON/Hydra configuration files. The default is a quick smoke setup: 3 clients, 2 rounds, a small MNIST subset, ALIE enabled for client 1, and three CKKS profiles.

For a fuller experiment, edit `config.py` and increase values such as:

```python
quick_data_limit = 6000
num_rounds = 10
clients_per_round = 5
attack_type = "fang_mean"  # or "alie" / "none"
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
- `crypto_calibration/reference_residual_bank.npz`: fixed reference residual samples used to build the CFI perturbation bank.

Plaintext aggregate reference metrics are optional experiment-only validation fields computed by the coordinator-side experiment verifier and are never used to update the global model.


## Component registry and extension

Built-in components are registered in `factories/defaults.py`. The coordinator depends on injected dataset, model factory, crypto backend, decryption service, defense, and attack objects; it does not import MNIST, CKKS, GeoChoke, or concrete attacks. To switch components, register a new implementation with `register_dataset`, `register_model`, `register_crypto`, `register_defense`, or `register_attack`, then change only the corresponding name in `config.py` (`dataset_name`, `model_name`, `crypto_backend_name`, `defense_name`, `attack_name`). Unknown names raise `ValueError` rather than silently falling back.

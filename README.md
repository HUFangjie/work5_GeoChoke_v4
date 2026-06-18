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
7. The server builds aggregate-level candidate models from scales such as `1.0`, `0.5`, `0.25`, `0.1`, and `0.0`.
8. GeoChoke gates the current aggregate using only the previous global model, decrypted aggregate update, unlabeled proxy set, and fixed CKKS residual perturbation bank. It computes separately normalized drift, CFI, and CFI-injection risks, selects the largest candidate scale whose normalized risk is at most `1.0`, and may reject the current aggregate with scale `0.0`.
9. GeoChoke maintains CUSUM cumulative risk plus a last-safe checkpoint; sustained abnormal risk can trigger rollback and a recovery period with capped update scale. It then selects the next-round profile via normalized log-energy control and one-step high→medium→low / low→medium→high transitions.

## Fang Mean adaptation

Original Fang attacks are usually framed against robust aggregators such as Krum or trimmed mean. This project's aggregator is CKKS weighted mean. Without oracle access, the non-oracle fallback is explicitly reported as `sign_flip_scaled` rather than being mislabeled as standard Fang. The default threat model does **not** expose benign clients' plaintext updates or the clean aggregate to the attacker. The optional `oracle_mean_replacement=True` mode keeps the older mean-replacement oracle only as an explicit strongest pressure test. Krum/TrimmedMean-specific Fang variants must reject `aggregation="weighted_mean"` rather than silently running the wrong objective.

## Configuration

All experiment settings live in `config.py`; there are no YAML/JSON/Hydra configuration files. The default is a 30-round MNIST setup with attack rounds 10--29, `fang_mean` configured for client 1, `attack_whitebox=False`, and three CKKS profiles. With `oracle_mean_replacement=False`, the effective non-oracle attack is logged as `sign_flip_scaled`.

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
- `crypto_calibration/direct_profile_calibration.csv`: per-profile offline CKKS aggregation-pipeline residual MSE, max residual, relative L2, norm ratio, residual mean, and residual standard deviation.
- `crypto_calibration/aggregation_pipeline_validation.csv`: full Enc → plaintext-weight multiply → ciphertext addition → aggregate decrypt validation metrics, including norm ratio.
- `crypto_calibration/reference_residual_bank.npz`: fixed reference residual samples used to build the CFI perturbation bank.

Plaintext aggregate reference metrics are optional experiment-only validation fields computed by the coordinator-side experiment verifier and are never used to update the global model.


## Component registry and extension

Built-in components are registered in `factories/defaults.py`. The coordinator depends on injected dataset, model factory, crypto backend, decryption service, defense, and attack objects; it does not import MNIST, CKKS, GeoChoke, or concrete attacks. To switch components, register a new implementation with `register_dataset`, `register_model`, `register_crypto`, `register_defense`, or `register_attack`, then change only the corresponding name in `config.py` (`dataset_name`, `model_name`, `crypto_backend_name`, `defense_name` (`"geochoke"`, `"none"`, or another registered defense), `attack_name`). Unknown names raise `ValueError` rather than silently falling back.

### No-defense mode

Set `defense_name = "none"` in `config.py` to disable GeoChoke profile adaptation while keeping the encrypted aggregation pipeline unchanged. `NoDefense` always returns the configured initial CKKS profile and records `defense_enabled=False` in round metrics.

### Oracle mean-replacement stress-test mode

`attack_whitebox` defaults to `False`; ordinary ALIE/FangMean attacks receive only compromised-client observable updates and round metadata. Set `attack_start_round` and `attack_end_round` in `config.py` to bound when malicious clients actually modify updates. If you intentionally want the older oracle pressure test, set `oracle_mean_replacement = True`; only then do attack modules receive `all_clean_updates`, `clean_aggregate`, and `benign_weighted_sum`. This oracle mode is not the default threat model and is never used by GeoChoke's defense decision. Per-round metrics include raw and accepted candidate risk, normalized drift/CFI/injection risks, CUSUM score, rollback flag, accepted update scale, CKKS residual ratio, and profile transition reason.

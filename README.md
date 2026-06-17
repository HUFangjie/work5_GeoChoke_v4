# GeoChoke CKKS Secure Aggregation FL

A minimal, runnable, modular single-server encrypted aggregation federated-learning project for MNIST. It uses real TenSEAL CKKS vectors for client update encryption, plaintext-ciphertext scalar multiplication, ciphertext addition, and aggregate-only decryption.

## Security boundary

This is a CKKS-based single-server encrypted aggregation system. It relies on an independent `AuthorizedDecryptionService` that does not collude with the `AggregationServer`. It is **not** threshold CKKS and **not** Bonawitz masked SecAgg. Clients and the aggregation server receive only public CKKS contexts. The decryption service exposes only `decrypt_aggregate(aggregated_ciphertext, profile_id)` and does not provide single-client decryption APIs.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
pytest -q
```

All experiment settings live in `config.py`. The default is a quick smoke configuration: MNIST subset, 3 clients, 2 rounds, ALIE enabled for client 1.

## Directory tree

See the repository folders `core/`, `crypto/`, `defenses/geochoke/`, `attacks/`, `data/`, `models/`, `evaluation/`, `utils/`, and `tests/`.

## Fang Mean adaptation

Original Fang attacks are usually framed against robust aggregators such as Krum or trimmed mean. This project's aggregator is CKKS weighted mean, so `FangMeanAttack` performs a mean-compatible malicious-update search that maximizes deviation from the observed benign center under a norm budget. Krum/TrimmedMean-specific variants must reject `aggregation="weighted_mean"` rather than silently running the wrong objective.

## Implemented workflow

1. Server samples clients and broadcasts the global model plus the current profile ID.
2. Each client trains an independent model copy on its own MNIST split.
3. Malicious clients apply ALIE or FangMean to their plaintext update before encryption.
4. Every client encrypts chunks with real TenSEAL CKKS under the public context.
5. The server computes weighted mean using CKKS scalar multiplication and ciphertext addition only.
6. The decryption service decrypts only the aggregate ciphertext.
7. GeoChoke compares previous and candidate CFI using the fixed proxy set and fixed perturbation bank, then selects the next-round profile.

## Configuration example

Edit `config.py` only. For a fuller run, set `quick_data_limit = 6000`, `num_rounds = 10`, `clients_per_round = 5`, and choose `attack_type = "fang_mean"` or `"alie"`.

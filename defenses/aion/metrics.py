from __future__ import annotations

import csv
import json
import os
from typing import Any


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def append_csv(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as handle:
            fieldnames = list(csv.DictReader(handle).fieldnames or [])
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    else:
        fieldnames = sorted(row.keys())
    rows = []
    if exists:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append(row)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

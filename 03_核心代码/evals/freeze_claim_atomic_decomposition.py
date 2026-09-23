"""Validate human-approved claim decomposition and freeze immutable atomic IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from evals.claim_atomic_contract import FIELD_NAMES, validate_atomic_definition
except ModuleNotFoundError:
    from claim_atomic_contract import FIELD_NAMES, validate_atomic_definition  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze(review_path: Path, dataset_path: Path) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_rows = {row["example_id"]: row for row in dataset}
    if review.get("artifact_type") != "claim_atomic_decomposition_review_candidates":
        raise ValueError("unexpected review artifact type")
    if review.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("review dataset hash mismatch")
    if review.get("field_names") != list(FIELD_NAMES):
        raise ValueError("fixed field contract mismatch")
    review_rows = review.get("records")
    if not isinstance(review_rows, list) or {row.get("example_id") for row in review_rows} != set(dataset_rows):
        raise ValueError("review IDs do not match dataset")
    frozen_records = []
    all_atomic_ids = set()
    for review_row in review_rows:
        example_id = review_row["example_id"]
        if review_row.get("review_status") != "approved":
            raise ValueError(f"decomposition is not approved: {example_id}")
        if review_row.get("source_record_sha256") != dataset_rows[example_id]["record_sha256"]:
            raise ValueError(f"source record hash mismatch: {example_id}")
        atomics = review_row.get("atomics")
        if not isinstance(atomics, list) or not atomics:
            raise ValueError(f"no atomics: {example_id}")
        validated = []
        for index, atomic in enumerate(atomics, start=1):
            if not isinstance(atomic, Mapping):
                raise ValueError(f"invalid atomic: {example_id}/{index}")
            expected_id = f"{example_id}::a{index:03d}"
            if atomic.get("atomic_id") != expected_id or expected_id in all_atomic_ids:
                raise ValueError(f"non-canonical or duplicate atomic_id: {expected_id}")
            all_atomic_ids.add(expected_id)
            fragment = atomic.get("claim_fragment")
            fields = atomic.get("fields")
            if not isinstance(fragment, str) or not fragment.strip():
                raise ValueError(f"empty claim_fragment: {expected_id}")
            if not isinstance(fields, Mapping) or set(fields) != set(FIELD_NAMES):
                raise ValueError(f"fixed fields mismatch: {expected_id}")
            if not any(isinstance(fields[name], str) and fields[name].strip() for name in FIELD_NAMES):
                raise ValueError(f"all fields are not_applicable: {expected_id}")
            if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in fields.values()):
                raise ValueError(f"field values must be non-empty strings or null: {expected_id}")
            validate_atomic_definition(atomic, label=expected_id)
            validated.append({
                "atomic_id": expected_id,
                "claim_fragment": fragment.strip(),
                "fields": {name: fields[name] for name in FIELD_NAMES},
            })
        frozen_records.append({
            "example_id": example_id,
            "source_record_sha256": review_row["source_record_sha256"],
            "atomics": validated,
            "atomics_sha256": _canonical_sha256(validated),
        })
    frozen_records.sort(key=lambda row: row["example_id"])
    return {
        "schema_version": "1.0",
        "artifact_type": "claim_atomic_decomposition_frozen",
        "dataset_sha256": review["dataset_sha256"],
        "field_names": list(FIELD_NAMES),
        "record_count": len(frozen_records),
        "atomic_count": sum(len(row["atomics"]) for row in frozen_records),
        "review_source_sha256": _sha256(review_path),
        "records": frozen_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    artifact = freeze(args.review.resolve(), args.dataset.resolve())
    _atomic_json(output, artifact)
    _atomic_json(args.manifest.resolve(), {
        "schema_version": "1.0",
        "artifact_type": "claim_atomic_decomposition_manifest",
        "dataset_sha256": artifact["dataset_sha256"],
        "output": str(output),
        "output_sha256": _sha256(output),
        "record_count": artifact["record_count"],
        "atomic_count": artifact["atomic_count"],
        "review_source_sha256": artifact["review_source_sha256"],
    })
    print(json.dumps({
        "output_sha256": _sha256(output),
        "record_count": artifact["record_count"],
        "atomic_count": artifact["atomic_count"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

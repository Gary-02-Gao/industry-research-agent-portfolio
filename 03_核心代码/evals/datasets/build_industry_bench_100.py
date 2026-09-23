#!/usr/bin/env python3
"""Build the fixed 40/30/20/10 IndustryBench-100 from reviewed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from pypdf import PdfReader


TARGETS = {
    "single_document_fact": 40,
    "same_document_multi_hop": 30,
    "cross_document_comparison": 20,
    "unanswerable": 10,
}
SPLIT_TARGETS = {
    "dev": {"single_document_fact": 16, "same_document_multi_hop": 12, "cross_document_comparison": 8, "unanswerable": 4},
    "validation": {"single_document_fact": 12, "same_document_multi_hop": 9, "cross_document_comparison": 6, "unanswerable": 3},
    "test": {"single_document_fact": 12, "same_document_multi_hop": 9, "cross_document_comparison": 6, "unanswerable": 3},
}

# Human reviewers repaired these table excerpts by joining the visual header
# and value row. PDF text extraction serializes some table columns separately,
# so the joined transcription is not a literal contiguous substring. Keep the
# reviewed meaning while materializing each referenced extracted line as its
# own traceable evidence span. Values are zero-based pypdf line ranges.
SUPPORT_LINE_REPAIRS: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {
    ("IBV1-029", "data/华电科工.pdf"): ((4, 4), (23, 23)),
    ("IBV1-032", "data/民银国际.pdf"): ((55, 56),),
    ("IBV1-046", "data/民银国际.pdf"): ((2, 2), (13, 13), (46, 46), (64, 64)),
    ("IBV1-047", "data/民银国际.pdf"): ((2, 2), (13, 13), (22, 22), (64, 64)),
    ("IBV1-048", "data/民银国际.pdf"): ((2, 2), (13, 13), (29, 29), (64, 64)),
    ("IBV1-059", "data/民银国际.pdf"): ((55, 55), (60, 61)),
    ("IBV1-060", "data/民银国际.pdf"): ((4, 4), (7, 8), (14, 15), (20, 20)),
    ("IBV1-061", "data/民银国际.pdf"): ((4, 4), (7, 8), (14, 15), (18, 18)),
    ("IBV1-069", "data/中邮证券.pdf"): ((28, 28), (35, 35)),
    ("IBV1-069", "data/华电科工.pdf"): ((30, 30), (43, 43)),
    ("IBV1-071", "data/中邮证券.pdf"): ((28, 28), (32, 33)),
    ("IBV1-071", "data/华鑫证券.pdf"): ((20, 20), (23, 24)),
    ("IBV1-072", "data/中邮证券.pdf"): ((28, 28), (35, 35)),
    ("IBV1-072", "data/华鑫证券.pdf"): ((10, 12),),
    ("IBV1-073", "data/中邮证券.pdf"): ((28, 28), (34, 34)),
    ("IBV1-073", "data/华鑫证券.pdf"): ((20, 20), (25, 25)),
    ("IBV1-077", "data/华电科工.pdf"): ((30, 30), (39, 39)),
    ("IBV1-078", "data/中邮证券.pdf"): ((2, 2), (8, 8)),
    ("IBV1-078", "data/华电科工.pdf"): ((30, 30), (36, 36)),
    ("IBV1-079", "data/中邮证券.pdf"): ((2, 2), (19, 21)),
    ("IBV1-079", "data/华电科工.pdf"): ((30, 30), (33, 33)),
    ("IBV1-080", "data/华电科工.pdf"): ((30, 30), (39, 39)),
    ("IBV1-081", "data/华电科工.pdf"): ((30, 30), (43, 43)),
    ("IBV1-082", "data/华电科工.pdf"): ((30, 31),),
    ("IBV1-082", "data/华鑫证券.pdf"): ((20, 20), (25, 25)),
}


class DatasetBuildError(ValueError):
    pass


def _normalize_pdf_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _materialize_traceable_supports(
    row: dict[str, Any], case_id: str, root: Path, reader_cache: dict[Path, PdfReader]
) -> None:
    materialized: list[dict[str, Any]] = []
    for support in row.get("support", []):
        source_path = str(support.get("source_path", ""))
        path = root / source_path
        if path not in reader_cache:
            reader_cache[path] = PdfReader(path)
        page_number = int(support["page"])
        page_text = reader_cache[path].pages[page_number - 1].extract_text() or ""
        if _normalize_pdf_text(str(support.get("support_text", ""))) in _normalize_pdf_text(page_text):
            materialized.append(support)
            continue
        ranges = SUPPORT_LINE_REPAIRS.get((case_id, source_path))
        if ranges is None:
            raise DatasetBuildError(f"{case_id}: support is not traceable and has no audited repair")
        lines = page_text.splitlines()
        for start, end in ranges:
            if start < 0 or end < start or end >= len(lines):
                raise DatasetBuildError(f"{case_id}: invalid support line repair for {source_path}")
            repaired = deepcopy(support)
            repaired.pop("chunk_id", None)
            repaired["support_text"] = "\n".join(lines[start : end + 1]).strip()
            if not repaired["support_text"]:
                raise DatasetBuildError(f"{case_id}: empty repaired support for {source_path}")
            materialized.append(repaired)
    row["support"] = materialized


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pool_candidate(record: Mapping[str, Any], root: Path, cache: dict[Path, dict[str, Any]]) -> Mapping[str, Any]:
    provenance = record.get("candidate_provenance")
    if not isinstance(provenance, Mapping):
        raise DatasetBuildError(f"{record.get('review_id')}: missing candidate provenance")
    pool_value = provenance.get("source_pool") or provenance.get("pool")
    candidate_id = provenance.get("original_candidate_id") or provenance.get("candidate_id")
    if not pool_value or not candidate_id:
        raise DatasetBuildError(f"{record.get('review_id')}: incomplete candidate provenance")
    pool = Path(str(pool_value))
    if not pool.is_absolute():
        pool = root / pool
    if pool not in cache:
        artifact = _read_json(pool)
        cache[pool] = {str(item["candidate_id"]): item for item in artifact.get("candidates", [])}
    try:
        return cache[pool][str(candidate_id)]
    except KeyError as exc:
        raise DatasetBuildError(f"{record.get('review_id')}: source candidate not found") from exc


def _candidate_case(record: Mapping[str, Any], root: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    source = _pool_candidate(record, root, cache)
    category = str(record.get("proposed_category", ""))
    if category not in TARGETS:
        raise DatasetBuildError(f"{record.get('review_id')}: invalid category")
    return {
        "category": category,
        "query": deepcopy(record.get("query")),
        "gold_claims": deepcopy(record.get("gold_claims")),
        "answerable": category != "unanswerable",
        "allowed_tools": deepcopy(source.get("allowed_tools") or ["local_retrieval"]),
        "required_tools": deepcopy(source.get("required_tools") or ["local_retrieval"]),
        "support": deepcopy(record.get("support") or []),
    }


def _split_ids(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(str(row["id"]))
    offsets = Counter()
    result: dict[str, Any] = {}
    for split, targets in SPLIT_TARGETS.items():
        ids: list[str] = []
        for category in TARGETS:
            start, count = offsets[category], targets[category]
            ids.extend(by_category[category][start : start + count])
            offsets[category] += count
        result[split] = {"hidden": split == "test", "ids": sorted(ids)}
    if set().union(*(set(value["ids"]) for value in result.values())) != {str(row["id"]) for row in rows}:
        raise DatasetBuildError("split allocation is not exhaustive")
    return result


def build(
    base_rows: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Any],
    schema: Mapping[str, Any],
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [deepcopy(dict(row)) for row in base_rows]
    cache: dict[Path, dict[str, Any]] = {}
    remaining = Counter(TARGETS)
    remaining.subtract(Counter(str(row.get("category")) for row in rows))
    for artifact in stages:
        if artifact.get("artifact_type") != "industry_bench_candidate_final":
            raise DatasetBuildError("all stage artifacts must be finalized")
        for record in artifact.get("accepted", []):
            category = str(record.get("proposed_category", ""))
            if remaining[category] <= 0:
                continue
            rows.append(_candidate_case(record, root, cache))
            remaining[category] -= 1
    if any(value != 0 for value in remaining.values()):
        raise DatasetBuildError(f"category quotas are not filled: {dict(remaining)}")
    if len(rows) != 100:
        raise DatasetBuildError(f"expected 100 cases, got {len(rows)}")

    document_map = {item["source_path"]: item for item in documents.get("documents", [])}
    span_counters: Counter[tuple[str, int]] = Counter()
    for row in rows[: len(base_rows)]:
        for support in row.get("support", []):
            key = (str(support["document_id"]), int(support["page"]))
            suffix = str(support["chunk_id"]).rsplit(":s", 1)[-1]
            if suffix.isdigit():
                span_counters[key] = max(span_counters[key], int(suffix))

    seen_queries: set[str] = set()
    validator = Draft202012Validator(schema)
    reader_cache: dict[Path, PdfReader] = {}
    for index, row in enumerate(rows, 1):
        case_id = f"IBV1-{index:03d}"
        row["id"] = row["case_id"] = case_id
        normalized_query = "".join(str(row.get("query") or "").split()).casefold()
        if not normalized_query or normalized_query in seen_queries:
            raise DatasetBuildError(f"{case_id}: empty or duplicate query")
        seen_queries.add(normalized_query)
        _materialize_traceable_supports(row, case_id, root, reader_cache)
        for support in row.get("support", []):
            source_path = str(support.get("source_path", ""))
            document = document_map.get(source_path)
            if document is None:
                raise DatasetBuildError(f"{case_id}: unknown source {source_path}")
            if support.get("sha256") != document.get("sha256"):
                raise DatasetBuildError(f"{case_id}: source hash mismatch")
            support["document_id"] = document["document_id"]
            if not support.get("chunk_id"):
                key = (str(document["document_id"]), int(support["page"]))
                span_counters[key] += 1
                support["chunk_id"] = f"ibv1:{key[0]}:p{key[1]:03d}:s{span_counters[key]:02d}"
            if _normalize_pdf_text(str(support["support_text"])) not in _normalize_pdf_text(
                reader_cache[root / source_path].pages[int(support["page"]) - 1].extract_text() or ""
            ):
                raise DatasetBuildError(f"{case_id}: materialized support is not traceable")
        errors = sorted(validator.iter_errors(row), key=lambda value: list(value.path))
        if errors:
            raise DatasetBuildError(f"{case_id}: schema error: {errors[0].message}")

    counts = Counter(str(row["category"]) for row in rows)
    if dict(counts) != TARGETS:
        raise DatasetBuildError(f"unexpected category composition: {dict(counts)}")
    splits = {
        "dataset_id": "industry_bench_100",
        "dataset_version": "1.0.0",
        "splits": _split_ids(rows),
    }
    return rows, splits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="evals/datasets/industry_bench_v1_1.jsonl")
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--documents", default="evals/datasets/documents.json")
    parser.add_argument("--schema", default="evals/datasets/industry_bench_v1.schema.json")
    parser.add_argument("--output", default="evals/datasets/industry_bench_100.jsonl")
    parser.add_argument("--splits", default="evals/datasets/industry_bench_100_splits.json")
    args = parser.parse_args(argv)
    root = Path.cwd()
    base_path, output_path = Path(args.base), Path(args.output)
    base_rows = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_paths = [Path(value) for value in args.stage]
    schema = _read_json(Path(args.schema))
    rows, splits = build(
        base_rows,
        [_read_json(path) for path in stage_paths],
        _read_json(Path(args.documents)),
        schema,
        root,
    )
    dataset_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n").encode("utf-8")
    _write_atomic(output_path, dataset_bytes)
    _write_atomic(Path(args.splits), (json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    sidecar = output_path.with_suffix(".schema.json")
    _write_atomic(sidecar, (json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    manifest = {
        "artifact_type": "industry_bench_dataset_manifest",
        "dataset_id": "industry_bench_100",
        "dataset_version": "1.0.0",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "case_count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "sources": {
            "base": {"path": str(base_path), "sha256": _sha256(base_path)},
            "stages": [{"path": str(path), "sha256": _sha256(path)} for path in stage_paths],
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    _write_atomic(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(output_path), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

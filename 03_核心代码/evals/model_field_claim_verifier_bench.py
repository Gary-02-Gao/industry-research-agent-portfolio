"""Benchmark verification against a pre-reviewed, frozen atomic decomposition."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError

try:
    from evals.claim_atomic_contract import (
        FIELD_NAMES, FIELD_RESULT_STATUSES, validate_atomic_definition,
    )
    from evals.model_claim_verifier_bench import (
        _atomic, _evidence_text, _extract_json, _load_rows, _metrics, _post,
        _service_fingerprint, _sha256,
    )
except ModuleNotFoundError:  # Standalone server bundle.
    from claim_atomic_contract import (  # type: ignore[no-redef]
        FIELD_NAMES, FIELD_RESULT_STATUSES, validate_atomic_definition,
    )
    from model_claim_verifier_bench import (  # type: ignore[no-redef]
        _atomic, _evidence_text, _extract_json, _load_rows, _metrics, _post,
        _service_fingerprint, _sha256,
    )


CHECK_STATUSES = set(FIELD_RESULT_STATUSES)
SYSTEM_PROMPT = """你是逐字段 Claim-Evidence 核验器，只能使用给定证据。
原子项已经人工审核并冻结；不得增删、合并、拆分、改写或遗漏 request_id。
request_id 只是本次请求内的无语义短 ID，不包含数据集、标签或审核来源信息。
只判定 applicable_fields 中列出的字段：matched=证据明确支持；missing=证据未说明；
conflict=证据明确给出不同信息。未列出的固定字段由程序确定性填充为 not_applicable，禁止输出它们。
数字逐位核对，除完全等价的单位换算外不允许近似；年份、日期和已完成/即将/计划严格区分。
同一主体、指标、时间下数值完全相同必须判 matched，不能因证据中还有其他数字而判 conflict。
只有证据明确绑定到当前原子且表达相反值/状态才判 conflict；没有明确绑定或只可推断时判 missing。
不得把总合同、母公司、其他年度或其他对象的属性推断给当前原子。
results 按 applicable_fields 的给定顺序返回四元数组：[字段名,状态,置信度,证据片段]。
状态只能是 matched、missing、conflict；置信度为 0 到 1；matched/conflict 的证据片段必须是证据原文。
输出结构由服务端 JSON Schema 强制约束。
不要输出总分类，不要输出新的原子项，不要 Markdown，不要思维过程。"""
FEW_SHOT = """

示例1原子：[{"request_id":"A1","claim_fragment":"甲公司2025年收入10亿元","applicable_fields":{"subject":"甲公司","predicate":"收入","value":"10","unit":"亿元","currency":"CNY","time_scope":"2025"}}]
证据：甲公司2025年收入为9亿元。
输出：{"request_id":"A1","results":[["subject","matched",0.99,"甲公司"],["predicate","matched",0.99,"收入"],["value","conflict",0.99,"9亿元"],["unit","matched",0.99,"亿元"],["currency","matched",0.99,"亿元"],["time_scope","matched",0.99,"2025年"]]}

示例2原子：[{"request_id":"A1","claim_fragment":"乙公司2025年利润5亿元","applicable_fields":{"subject":"乙公司","predicate":"利润","value":"5","unit":"亿元","currency":"CNY","time_scope":"2025"}}]
证据：乙公司2025年实现利润5亿元，另有收入20亿元。
输出：{"request_id":"A1","results":[["subject","matched",0.99,"乙公司"],["predicate","matched",0.99,"利润"],["value","matched",0.99,"5亿元"],["unit","matched",0.99,"亿元"],["currency","matched",0.99,"亿元"],["time_scope","matched",0.99,"2025年"]]}

示例3原子：[{"request_id":"A1","claim_fragment":"丙公司的设备合同金额8亿元为含税金额","applicable_fields":{"subject":"设备合同金额","predicate":"税务口径","object":"含税","value":"8","unit":"亿元","currency":"CNY"}}]
证据：丙公司签署10亿元（含税）总合同，其中设备合同金额为8亿元。
输出：{"request_id":"A1","results":[["subject","matched",0.99,"设备合同金额为8亿元"],["predicate","missing",0.95,""],["object","missing",0.95,""],["value","matched",0.99,"8亿元"],["unit","matched",0.99,"亿元"],["currency","matched",0.99,"亿元"]]}
"""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_decomposition(
    path: Path,
    expected_sha256: str | None,
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    observed = _sha256(path)
    if expected_sha256 and observed != expected_sha256:
        raise ValueError(f"decomposition hash mismatch: expected={expected_sha256} observed={observed}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "claim_atomic_decomposition_frozen":
        raise ValueError("decomposition is not a frozen artifact")
    if artifact.get("field_names") != list(FIELD_NAMES):
        raise ValueError("decomposition fixed field contract mismatch")
    source = {row["example_id"]: row for row in rows}
    records = artifact.get("records")
    if not isinstance(records, list) or {row.get("example_id") for row in records} != set(source):
        raise ValueError("decomposition IDs do not match dataset")
    if artifact.get("record_count") != len(records):
        raise ValueError("decomposition record_count mismatch")
    if not isinstance(artifact.get("review_source_sha256"), str) or len(artifact["review_source_sha256"]) != 64:
        raise ValueError("decomposition has no review source hash")
    result: dict[str, list[dict[str, Any]]] = {}
    atomic_ids = set()
    for record in records:
        example_id = record["example_id"]
        if record.get("source_record_sha256") != source[example_id]["record_sha256"]:
            raise ValueError(f"decomposition source hash mismatch: {example_id}")
        atomics = record.get("atomics")
        if not isinstance(atomics, list) or not atomics:
            raise ValueError(f"decomposition has no atomics: {example_id}")
        if record.get("atomics_sha256") != _canonical_sha256(atomics):
            raise ValueError(f"atomic hash mismatch: {example_id}")
        for index, atomic in enumerate(atomics, start=1):
            if not isinstance(atomic, Mapping) or set(atomic) != {"atomic_id", "claim_fragment", "fields"}:
                raise ValueError(f"invalid frozen atomic: {example_id}")
            atomic_id = atomic["atomic_id"]
            fields = atomic["fields"]
            if atomic_id != f"{example_id}::a{index:03d}" or atomic_id in atomic_ids:
                raise ValueError(f"duplicate or invalid atomic_id: {atomic_id}")
            atomic_ids.add(atomic_id)
            if not isinstance(fields, Mapping) or set(fields) != set(FIELD_NAMES):
                raise ValueError(f"fixed fields mismatch: {atomic_id}")
            if not isinstance(atomic["claim_fragment"], str) or not atomic["claim_fragment"].strip():
                raise ValueError(f"empty claim_fragment: {atomic_id}")
            if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in fields.values()):
                raise ValueError(f"invalid fixed field value: {atomic_id}")
            validate_atomic_definition(atomic, label=atomic_id)
        result[example_id] = atomics
    if artifact.get("atomic_count") != len(atomic_ids):
        raise ValueError("decomposition atomic_count mismatch")
    return result


def _apply_diagnostic_scope(
    path: Path,
    dataset_path: Path,
    decomposition_path: Path,
    rows: list[dict[str, Any]],
    decomposition: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "claim_verifier_diagnostic_scope":
        raise ValueError("unexpected diagnostic scope artifact")
    if artifact.get("holdout_used") is not False:
        raise ValueError("diagnostic scope must not use holdout")
    if artifact.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("diagnostic scope dataset hash mismatch")
    if artifact.get("decomposition_sha256") != _sha256(decomposition_path):
        raise ValueError("diagnostic scope decomposition hash mismatch")
    records = artifact.get("records")
    if not isinstance(records, list) or artifact.get("records_sha256") != _canonical_sha256(records):
        raise ValueError("diagnostic scope records hash mismatch")
    if artifact.get("row_count") != len(records):
        raise ValueError("diagnostic scope row_count mismatch")
    source = {row["example_id"]: row for row in rows}
    ids = []
    seen_ids = set()
    atomic_count = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "example_id", "source_record_sha256", "atomics_sha256", "atomic_count",
        }:
            raise ValueError("invalid diagnostic scope record")
        example_id = record["example_id"]
        if example_id in seen_ids or example_id not in source:
            raise ValueError(f"invalid diagnostic scope ID: {example_id}")
        if record["source_record_sha256"] != source[example_id]["record_sha256"]:
            raise ValueError(f"diagnostic scope source hash mismatch: {example_id}")
        atomics = decomposition[example_id]
        if record["atomics_sha256"] != _canonical_sha256(atomics):
            raise ValueError(f"diagnostic scope atomic hash mismatch: {example_id}")
        if record["atomic_count"] != len(atomics):
            raise ValueError(f"diagnostic scope atomic_count mismatch: {example_id}")
        ids.append(example_id)
        seen_ids.add(example_id)
        atomic_count += len(atomics)
    if artifact.get("atomic_count") != atomic_count:
        raise ValueError("diagnostic scope total atomic_count mismatch")
    selected_ids = set(ids)
    selected_rows = [row for row in rows if row["example_id"] in selected_ids]
    selected_decomposition = {
        row["example_id"]: decomposition[row["example_id"]] for row in selected_rows
    }
    return selected_rows, selected_decomposition, artifact


def _prompt(row: Mapping[str, Any], atomic: dict[str, Any], prompt_variant: str) -> str:
    public_atomic = {
        "request_id": "A1",
        "claim_fragment": atomic["claim_fragment"],
        "applicable_fields": {
            field: atomic["fields"][field]
            for field in FIELD_NAMES
            if atomic["fields"][field] is not None
        },
    }
    return (
        SYSTEM_PROMPT
        + (FEW_SHOT if prompt_variant == "few-shot" else "")
        + "\n\n<FROZEN_ATOMICS>\n"
        + json.dumps([public_atomic], ensure_ascii=False)
        + "\n</FROZEN_ATOMICS>\n\n证据：\n"
        + _evidence_text(row)
    )


def _validate_checks(
    value: Mapping[str, Any],
    atomics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(atomics) != 1:
        raise ValueError("compact wire response requires exactly one frozen atomic")
    if set(value) != {"request_id", "results"} or value.get("request_id") != "A1":
        raise ValueError("response must return request_id A1 and results only")
    atomic = atomics[0]
    atomic_id = atomic["atomic_id"]
    field_results = value["results"]
    expected_fields = [
        field for field in FIELD_NAMES if atomic["fields"][field] is not None
    ]
    if not isinstance(field_results, list) or len(field_results) != len(expected_fields):
        raise ValueError(f"results must contain exactly all applicable fields: {atomic_id}")
    validated_fields = {}
    for field in FIELD_NAMES:
        if field not in expected_fields:
            validated_fields[field] = {
                "status": "not_applicable",
                "confidence": 1.0,
                "evidence_fragment": "",
            }
        else:
            position = expected_fields.index(field)
            check = field_results[position]
            if not isinstance(check, list) or len(check) != 4 or check[0] != field:
                raise ValueError(f"invalid field result: {atomic_id}/{field}")
            _, status, confidence, evidence = check
            if status not in CHECK_STATUSES - {"not_applicable"}:
                raise ValueError(f"invalid status: {atomic_id}/{field}")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"invalid confidence: {atomic_id}/{field}")
            if not isinstance(evidence, str):
                raise ValueError(f"invalid evidence_fragment: {atomic_id}/{field}")
            if status in {"matched", "conflict"} and not evidence.strip():
                raise ValueError(f"evidence is required: {atomic_id}/{field}")
            validated_fields[field] = {
                "status": status,
                "confidence": float(confidence),
                "evidence_fragment": evidence.strip(),
            }
    return [{"atomic_id": atomic_id, "field_results": validated_fields}]


def _response_format(atomic: Mapping[str, Any]) -> dict[str, Any]:
    applicable_fields = [
        field for field in FIELD_NAMES if atomic["fields"][field] is not None
    ]

    def field_tuple(field: str) -> dict[str, Any]:
        return {
            "type": "array",
            "prefixItems": [
                {"type": "string", "const": field},
                {"type": "string", "enum": ["matched", "missing", "conflict"]},
                {"type": "number", "minimum": 0, "maximum": 1},
                {"type": "string"},
            ],
            "minItems": 4,
            "maxItems": 4,
        }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frozen_atomic_field_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "request_id": {"type": "string", "const": "A1"},
                    "results": {
                        "type": "array",
                        "prefixItems": [field_tuple(field) for field in applicable_fields],
                        "minItems": len(applicable_fields),
                        "maxItems": len(applicable_fields),
                    },
                },
                "required": ["request_id", "results"],
            },
        },
    }


def aggregate_checks(checks: list[Mapping[str, Any]]) -> tuple[str, float]:
    applicable = [
        value
        for item in checks
        for value in item["field_results"].values()
        if value["status"] != "not_applicable"
    ]
    if not applicable:
        raise ValueError("frozen decomposition has no applicable fields")
    statuses = [item["status"] for item in applicable]
    confidence = min(float(item["confidence"]) for item in applicable)
    if "conflict" in statuses:
        return "contradicted", confidence
    if all(status == "matched" for status in statuses):
        return "fully_supported", confidence
    if "matched" in statuses:
        return "partially_supported", confidence
    return "insufficient", confidence


def _row_field_predictions(checks: list[Mapping[str, Any]]) -> dict[str, str]:
    result = {}
    for field in FIELD_NAMES:
        statuses = [item["field_results"][field]["status"] for item in checks]
        if "conflict" in statuses:
            result[field] = "contradicted"
        elif "missing" in statuses:
            result[field] = "missing"
        elif "matched" in statuses:
            result[field] = "supported"
        else:
            result[field] = "not_applicable"
    return result


def _field_metrics(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    scoped = [row for row in rows if row.get("gold_field_results") is not None]
    scope_ids = sorted(row["example_id"] for row in scoped)
    records_by_id = {record["example_id"]: record for record in records}
    confusion = {field: Counter() for field in FIELD_NAMES}
    value_gold = value_hit = time_gold = time_hit = 0
    schema_valid_rows = 0
    for row in scoped:
        record = records_by_id.get(row["example_id"])
        gold = row["gold_field_results"]
        if record and record.get("checks"):
            schema_valid_rows += 1
            predicted = _row_field_predictions(record["checks"])
        else:
            predicted = {field: "__error__" for field in FIELD_NAMES}
        for field in FIELD_NAMES:
            confusion[field][f"{gold[field]}->{predicted[field]}"] += 1
        if gold["value"] == "contradicted":
            value_gold += 1
            value_hit += predicted["value"] == "contradicted"
        if gold["time_scope"] == "contradicted":
            time_gold += 1
            time_hit += predicted["time_scope"] == "contradicted"
    return {
        "gold_scope": "rows_with_non_null_gold_field_results_only",
        "gold_row_count": len(scoped),
        "prediction_row_count": len(scoped),
        "schema_valid_prediction_row_count": schema_valid_rows,
        "dataset_row_count": len(rows),
        "gold_row_coverage": len(scoped) / len(rows),
        "gold_scope_sha256": hashlib.sha256("\n".join(scope_ids).encode("utf-8")).hexdigest(),
        "confusion": {field: dict(confusion[field]) for field in FIELD_NAMES},
        "value_conflict": {
            "support": value_gold,
            "recall": value_hit / value_gold if value_gold else 0.0,
        },
        "time_scope_conflict": {
            "support": time_gold,
            "recall": time_hit / time_gold if time_gold else 0.0,
        },
    }


def run(
    rows: list[dict[str, Any]], decomposition: dict[str, list[dict[str, Any]]], *,
    base_url: str, api_key: str, model: str, prompt_variant: str, workers: int,
    timeout: float, retries: int, max_tokens: int, confidence_cutoff: float,
) -> dict[str, Any]:
    def evaluate_atomic(task: tuple[Mapping[str, Any], dict[str, Any]]) -> dict[str, Any]:
        row, atomic = task
        started = time.perf_counter()
        raw = ""
        checks = None
        prediction = None
        usage: Mapping[str, Any] = {}
        error = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                raw, usage = _post(
                    base_url=base_url, api_key=api_key, model=model,
                    prompt=_prompt(row, atomic, prompt_variant), timeout=timeout,
                    max_tokens=max_tokens,
                    response_format=_response_format(atomic),
                )
                checks = _validate_checks(_extract_json(raw), [atomic])
                prediction = aggregate_checks(checks)
                error = None
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError, OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(0.5 * (2 ** attempt))
        return {
            "example_id": row["example_id"], "atomic_id": atomic["atomic_id"],
            "checks": checks, "schema_valid": prediction is not None,
            "attempts": attempts, "usage": dict(usage), "raw_content": raw,
            "error": error,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    started = time.perf_counter()
    tasks = [
        (row, atomic)
        for row in rows
        for atomic in decomposition[row["example_id"]]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        atomic_records = list(pool.map(evaluate_atomic, tasks))
    by_example: dict[str, list[dict[str, Any]]] = {}
    for atomic_record in atomic_records:
        by_example.setdefault(atomic_record["example_id"], []).append(atomic_record)
    records = []
    for row in rows:
        parts = by_example[row["example_id"]]
        checks = [
            check
            for part in parts
            for check in (part["checks"] or [])
        ]
        valid = all(part["schema_valid"] for part in parts)
        prediction = aggregate_checks(checks) if valid else None
        usage_keys = {key for part in parts for key in part["usage"]}
        records.append({
            "example_id": row["example_id"], "case_id": row.get("case_id"),
            "gold_status": row["gold_status"],
            "predicted_status": prediction[0] if prediction else None,
            "confidence": prediction[1] if prediction else None,
            "checks": checks if valid else None, "schema_valid": valid,
            "attempts": sum(part["attempts"] for part in parts),
            "usage": {
                key: sum(int(part["usage"].get(key) or 0) for part in parts)
                for key in usage_keys
            },
            "raw_content": [part["raw_content"] for part in parts],
            "error": "; ".join(part["error"] for part in parts if part["error"]) or None,
            "duration_ms": round(sum(part["duration_ms"] for part in parts), 3),
        })
    metrics = _metrics(records, confidence_cutoff)
    metrics["field_evaluation"] = _field_metrics(records, rows)
    prompt_payload = SYSTEM_PROMPT + (FEW_SHOT if prompt_variant == "few-shot" else "")
    return {
        "schema_version": "2.1",
        "artifact_type": "openai_field_first_claim_verifier_benchmark",
        "model": model, "prompt_variant": prompt_variant,
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
        "aggregation": "conflict>all_matched>matched_plus_missing>all_missing",
        "fixed_fields": list(FIELD_NAMES), "workers": workers,
        "model_wire_contract": "guided_applicable_field_tuples_v2; null fields deterministically expanded to not_applicable",
        "request_granularity": "one_frozen_atomic_per_request",
        "timeout_seconds": timeout, "retries": retries, "max_tokens": max_tokens,
        "thinking_enabled": False,
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
        "metrics": metrics, "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--expected-decomposition-sha256", required=True)
    parser.add_argument("--diagnostic-scope", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-variant", choices=("zero-shot", "few-shot"), default="zero-shot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--confidence-cutoff", type=float, default=0.8)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows = _load_rows(dataset, args.expected_dataset_sha256)
    decomposition_path = args.decomposition.resolve()
    decomposition_artifact = json.loads(decomposition_path.read_text(encoding="utf-8"))
    if decomposition_artifact.get("dataset_sha256") != _sha256(dataset):
        raise ValueError("decomposition references a different dataset hash")
    decomposition = _load_decomposition(
        decomposition_path, args.expected_decomposition_sha256, rows,
    )
    diagnostic_scope = None
    if args.diagnostic_scope:
        rows, decomposition, diagnostic_scope = _apply_diagnostic_scope(
            args.diagnostic_scope.resolve(), dataset, decomposition_path, rows, decomposition,
        )
    artifact = run(
        rows, decomposition, base_url=args.base_url, api_key=args.api_key,
        model=args.model, prompt_variant=args.prompt_variant,
        workers=max(1, min(args.workers, 16)), timeout=args.timeout,
        retries=max(0, args.retries), max_tokens=args.max_tokens,
        confidence_cutoff=args.confidence_cutoff,
    )
    artifact.update({
        "dataset": str(dataset), "dataset_sha256": _sha256(dataset),
        "decomposition": str(decomposition_path),
        "decomposition_sha256": _sha256(decomposition_path),
        "evaluation_scope": "diagnostic" if diagnostic_scope else "full_development",
        "service_fingerprint": _service_fingerprint(args.base_url, args.api_key, args.timeout),
    })
    if diagnostic_scope:
        artifact["diagnostic_scope"] = {
            "path": str(args.diagnostic_scope.resolve()),
            "sha256": _sha256(args.diagnostic_scope.resolve()),
            "row_count": diagnostic_scope["row_count"],
            "atomic_count": diagnostic_scope["atomic_count"],
            "records_sha256": diagnostic_scope["records_sha256"],
        }
    _atomic(args.output.resolve(), artifact)
    print(json.dumps(artifact["metrics"], ensure_ascii=False, sort_keys=True))
    return 0 if artifact["metrics"]["coverage"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

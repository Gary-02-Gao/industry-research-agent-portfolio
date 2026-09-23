"""Benchmark an OpenAI-compatible four-class Claim-Evidence verifier."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from http.client import RemoteDisconnected
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


STATUSES = (
    "fully_supported",
    "partially_supported",
    "contradicted",
    "insufficient",
)
SYSTEM_PROMPT = """你是独立的 Claim-Evidence 四分类核验器，只能使用给定证据，不得使用常识补全。先在内部把 Claim 拆成主体、时间、指标、数值、单位、状态、范围、因果等原子字段，逐字段与证据核对，但不要输出分析过程。
分类定义：
- fully_supported：证据完整支持 Claim 的所有必要事实字段。
- partially_supported：证据支持一部分必要字段，其他字段缺失，且不存在直接冲突。
- contradicted：证据对至少一个必要字段给出明确不一致信息。
- insufficient：证据与 Claim 无关、为空，或无法据此判断 Claim。
硬性规则：
1. 优先级是明确冲突 > 部分支持 > 信息不足；只要一个必要字段明确冲突，整体就是 contradicted。
2. 数字必须逐位核对。除完全等价的单位换算外，不得把相近数字、百分比或预测值当作四舍五入；例如证据 11.6% 与 Claim 11.7% 是冲突。
3. 年、月、日和“已完成/即将/计划”等阶段必须严格一致；错误时间或阶段是冲突，Claim 中有精确日期而证据没有日期是部分支持。
4. 并列清单、比较、算式和多主体 Claim 必须逐项核对；部分项目有证据、其余缺失是部分支持。必须自行重算算式，不能沿用 Claim 的结论。
5. 不得从上下文猜测缺失字段，不得因大部分内容一致而忽略一个字符、数字、主体或状态差异。
输出严格 JSON，不要 Markdown，不要思维过程；理由不超过35个汉字：
{"status":"fully_supported|partially_supported|contradicted|insufficient","confidence":0到1,"rationale":"一句可核验理由"}。"""
FEW_SHOT = """

以下是与正式数据无关的合成示例：
示例1 Claim：甲公司2024年收入为10亿元。证据：甲公司2024年收入为10亿元。
输出：{"status":"fully_supported","confidence":0.99,"rationale":"主体、年份、指标、数值和单位一致。"}
示例2 Claim：甲公司2024年收入为12亿元。证据：甲公司2024年收入为10亿元。
输出：{"status":"contradicted","confidence":0.99,"rationale":"相同口径下收入数值冲突。"}
示例3 Claim：甲公司2024年收入为10亿元且利润为2亿元。证据：甲公司2024年收入为10亿元。
输出：{"status":"partially_supported","confidence":0.95,"rationale":"收入获支持，但利润缺失。"}
示例4 Claim：甲公司2024年收入为10亿元。证据：乙公司发布了新产品。
输出：{"status":"insufficient","confidence":0.99,"rationale":"证据未涉及该主体及财务指标。"}
示例5 Claim：甲公司利润同比增长11.7%。证据：甲公司利润同比增长11.6%。
输出：{"status":"contradicted","confidence":0.99,"rationale":"同比增速11.7%与11.6%冲突。"}
示例6 Claim：甲产品已批量生产。证据：甲产品即将进入批量生产阶段。
输出：{"status":"contradicted","confidence":0.99,"rationale":"已量产与即将量产的阶段冲突。"}
示例7 Claim：某决议于2025年3月2日通过且获全票支持。证据：某决议已通过，但未说明日期与票数。
输出：{"status":"partially_supported","confidence":0.98,"rationale":"通过获支持，日期和票数缺失。"}
示例8 Claim：三项合计为13万元。证据：三项分别为5万元、4万元、3万元。
输出：{"status":"contradicted","confidence":0.99,"rationale":"证据三项合计12万元，并非13万元。"}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: object) -> None:
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


def _load_rows(path: Path, expected_sha256: str | None) -> list[dict[str, Any]]:
    observed = _sha256(path)
    if expected_sha256 and observed != expected_sha256:
        raise ValueError(f"dataset hash mismatch: expected={expected_sha256} observed={observed}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must be a non-empty JSON list")
    identifiers = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("gold_status") not in STATUSES:
            raise ValueError(f"row {index} has no finalized four-class Gold")
        if row.get("four_class_review_required") is not False:
            raise ValueError(f"row {index} is still pending review")
        if "hard_negative" in row:
            raise ValueError(f"row {index} contains construction metadata")
        identifiers.append(row.get("example_id"))
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("missing example_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate example_id")
    return rows


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def _extract_json(text: str) -> Mapping[str, Any]:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, Mapping):
        raise ValueError("response is not a JSON object")
    return value


def _validate_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"status", "confidence", "rationale"}:
        raise ValueError("response keys must be status, confidence, rationale")
    status = value.get("status")
    confidence = value.get("confidence")
    rationale = value.get("rationale")
    if status not in STATUSES:
        raise ValueError("invalid status")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be within [0, 1]")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty")
    return {"status": status, "confidence": float(confidence), "rationale": rationale.strip()}


def _evidence_text(row: Mapping[str, Any]) -> str:
    group = row.get("evidence_group")
    if not isinstance(group, Mapping) or not isinstance(group.get("spans"), list):
        raise ValueError(f"invalid evidence group for {row.get('example_id')}")
    return "\n\n---\n\n".join(
        str(span.get("text", "")) for span in group["spans"] if isinstance(span, Mapping)
    ) or "（无证据）"


def _prompt(row: Mapping[str, Any], prompt_variant: str) -> str:
    prefix = SYSTEM_PROMPT + (FEW_SHOT if prompt_variant == "few-shot" else "")
    return (
        prefix
        + "\n\n待核验 Claim：\n"
        + str(row["claim_text"])
        + "\n\n证据：\n"
        + _evidence_text(row)
    )


def _post(
    *, base_url: str, api_key: str, model: str, prompt: str, timeout: float, max_tokens: int,
    response_format: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "response_format": response_format or {"type": "json_object"},
        # Qwen3's reasoning mode is unnecessary for this constrained classifier
        # and can spend most of the output budget before emitting the JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        _endpoint(base_url),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"]), payload.get("usage") or {}


def _service_fingerprint(base_url: str, api_key: str, timeout: float) -> Mapping[str, Any]:
    value = base_url.rstrip("/")
    endpoint = value + "/models" if not value.endswith("/models") else value
    request = Request(endpoint, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with build_opener(ProxyHandler({})).open(request, timeout=min(timeout, 10.0)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, Mapping) else None
    return {
        "models_response": models if isinstance(models, list) else [],
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


def _metrics(records: list[dict[str, Any]], confidence_cutoff: float) -> dict[str, Any]:
    total = len(records)
    completed = [item for item in records if item["predicted_status"] in STATUSES]
    class_metrics = {}
    confusion = {label: Counter() for label in STATUSES}
    for item in records:
        confusion[item["gold_status"]][item["predicted_status"] or "__error__"] += 1
    for label in STATUSES:
        tp = sum(item["gold_status"] == label and item["predicted_status"] == label for item in records)
        fp = sum(item["gold_status"] != label and item["predicted_status"] == label for item in records)
        fn = sum(item["gold_status"] == label and item["predicted_status"] != label for item in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(item["gold_status"] == label for item in records),
        }
    high_conf_false_accepts = sum(
        item["gold_status"] != "fully_supported"
        and item["predicted_status"] == "fully_supported"
        and float(item["confidence"] or 0.0) >= confidence_cutoff
        for item in records
    )
    negatives = sum(item["gold_status"] != "fully_supported" for item in records)
    durations = sorted(float(item["duration_ms"]) for item in records)
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else 0.0
    return {
        "count": total,
        "completed": len(completed),
        "coverage": len(completed) / total,
        "schema_valid_rate": sum(item["schema_valid"] for item in records) / total,
        "accuracy": sum(item["predicted_status"] == item["gold_status"] for item in records) / total,
        "macro_f1_all_four_classes": sum(value["f1"] for value in class_metrics.values()) / 4,
        "class_metrics": class_metrics,
        "confusion": {label: dict(confusion[label]) for label in STATUSES},
        "high_confidence_cutoff": confidence_cutoff,
        "high_confidence_false_accept_count": high_conf_false_accepts,
        "high_confidence_false_accept_rate": high_conf_false_accepts / negatives if negatives else 0.0,
        "p95_ms": p95,
        "usage_totals": {
            key: sum(
                int(item.get("usage", {}).get(key) or 0)
                for item in records
                if isinstance(item.get("usage"), Mapping)
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


def run(
    rows: list[dict[str, Any]], *, base_url: str, api_key: str, model: str,
    prompt_variant: str, workers: int, timeout: float, retries: int,
    max_tokens: int, confidence_cutoff: float,
) -> dict[str, Any]:
    def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        raw = ""
        prediction = None
        usage: Mapping[str, Any] = {}
        error = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                raw, usage = _post(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=_prompt(row, prompt_variant),
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
                prediction = _validate_prediction(_extract_json(raw))
                error = None
                break
            except (
                HTTPError,
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionError,
                KeyError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(0.5 * (2 ** attempt))
        return {
            "example_id": row["example_id"],
            "case_id": row.get("case_id"),
            "gold_status": row["gold_status"],
            "predicted_status": prediction["status"] if prediction else None,
            "confidence": prediction["confidence"] if prediction else None,
            "rationale": prediction["rationale"] if prediction else None,
            "schema_valid": prediction is not None,
            "attempts": attempts,
            "usage": dict(usage),
            "raw_content": raw,
            "error": error,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(evaluate, rows))
    prompt_payload = SYSTEM_PROMPT + (FEW_SHOT if prompt_variant == "few-shot" else "")
    return {
        "schema_version": "1.0",
        "artifact_type": "openai_four_class_claim_verifier_benchmark",
        "model": model,
        "prompt_variant": prompt_variant,
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
        "workers": workers,
        "timeout_seconds": timeout,
        "retries": retries,
        "max_tokens": max_tokens,
        "thinking_enabled": False,
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
        "metrics": _metrics(records, confidence_cutoff),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-variant", choices=("zero-shot", "few-shot"), default="zero-shot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--confidence-cutoff", type=float, default=0.8)
    args = parser.parse_args()
    rows = _load_rows(args.dataset.resolve(), args.expected_dataset_sha256)
    service_fingerprint = _service_fingerprint(args.base_url, args.api_key, args.timeout)
    artifact = run(
        rows,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt_variant=args.prompt_variant,
        workers=max(1, min(args.workers, 16)),
        timeout=args.timeout,
        retries=max(0, args.retries),
        max_tokens=args.max_tokens,
        confidence_cutoff=args.confidence_cutoff,
    )
    artifact.update({
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "service_fingerprint": service_fingerprint,
    })
    _atomic(args.output.resolve(), artifact)
    print(json.dumps(artifact["metrics"], ensure_ascii=False, sort_keys=True))
    return 0 if artifact["metrics"]["coverage"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

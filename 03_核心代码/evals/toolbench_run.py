"""Run ToolBench-200 as a gold-hidden OpenAI-compatible routing evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

BACKEND_APP = Path(__file__).resolve().parents[1] / "backend" / "app"
if str(BACKEND_APP) not in os.sys.path:
    os.sys.path.insert(0, str(BACKEND_APP))

from tool_routing_contract import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_routing_response,
)

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "web_search": {"query": "non-empty string", "count": "integer 1..10"},
    "knowledge_search": {"query": "non-empty string", "kb_name": "non-empty string", "top_k": "integer 1..30"},
    "text2sql": {"question": "non-empty string", "intent": "stats|trend|comparison|detail"},
    "data_analyzer": {"data": "array", "analysis_type": "auto|trend|distribution|comparison"},
    "chart_generator": {"data": "array", "chart_type": "line|bar|pie|scatter", "title": "string"},
    "stock_query": {"stock_code": "string", "keyword": "string"},
    "bidding_search": {"keyword": "non-empty string", "category": "招标|中标|采购", "region": "string", "page": "integer >=1"},
    "finish": {"summary": "non-empty string"},
}

TOOL_DESCRIPTIONS = {
    "web_search": "检索公开互联网的最新或外部信息，不用于内部知识库或结构化数据库。",
    "knowledge_search": "只检索当前用户已授权的文档知识库，不代表关系数据库查询。",
    "text2sql": "把问题转换为只读 SQL，查询已鉴权的结构化关系数据库；出现数据库、数据表、SQL 时优先使用。",
    "data_analyzer": "对用户已给数据或前一步工具结果做确定性趋势、分布、比较分析。",
    "chart_generator": "把用户已给数据或前一步工具结果转换为图表配置。",
    "stock_query": "查询指定股票代码或名称的实时价格、涨跌幅、成交量。",
    "bidding_search": "检索招标、中标或采购公告。",
    "finish": "用户明确要求停止工具调用、结束研究并产出总结时使用的终止动作。",
}

def build_model_prompt(case: Mapping[str, Any]) -> str:
    """Build a model prompt without embedding any gold routing labels."""

    return build_user_prompt(
        case["query"],
        case["allowed_tools"],
        TOOL_DESCRIPTIONS,
        TOOL_SCHEMAS,
    )


def _parse_prediction_with_diagnostics(content: str) -> tuple[Dict[str, Any], List[str]]:
    return parse_routing_response(content)


def parse_prediction(content: str) -> Dict[str, Any]:
    prediction, _ = _parse_prediction_with_diagnostics(content)
    return prediction


def _valid_arguments(name: str, arguments: Mapping[str, Any]) -> bool:
    allowed = set(TOOL_SCHEMAS.get(name, {}))
    if name not in TOOL_SCHEMAS or set(arguments) - allowed:
        return False
    string = lambda key, required=False: (
        key not in arguments and not required
    ) or (isinstance(arguments.get(key), str) and (bool(arguments[key].strip()) if required else True))
    integer = lambda key, low, high=None: (
        key not in arguments
        or isinstance(arguments[key], int) and not isinstance(arguments[key], bool)
        and arguments[key] >= low and (high is None or arguments[key] <= high)
    )
    if name == "web_search":
        return string("query", True) and integer("count", 1, 10)
    if name == "knowledge_search":
        return string("query", True) and string("kb_name", True) and integer("top_k", 1, 30)
    if name == "text2sql":
        return string("question", True) and arguments.get("intent", "stats") in {"stats", "trend", "comparison", "detail"}
    if name == "data_analyzer":
        return isinstance(arguments.get("data", []), list) and arguments.get("analysis_type", "auto") in {"auto", "trend", "distribution", "comparison"}
    if name == "chart_generator":
        return isinstance(arguments.get("data"), list) and arguments.get("chart_type") in {"line", "bar", "pie", "scatter"} and string("title", True)
    if name == "stock_query":
        return string("stock_code") and string("keyword") and bool(arguments.get("stock_code") or arguments.get("keyword"))
    if name == "bidding_search":
        return string("keyword", True) and arguments.get("category", "招标") in {"招标", "中标", "采购"} and string("region") and integer("page", 1)
    if name == "finish":
        return string("summary", True)
    return False


def score_prediction(case: Mapping[str, Any], prediction: Mapping[str, Any]) -> Dict[str, Any]:
    calls = prediction["tool_calls"]
    selected = [call["name"] for call in calls]
    allowed = set(case["allowed_tools"])
    forbidden = [name for name in selected if name not in allowed]
    schema_valid = not forbidden and all(_valid_arguments(call["name"], call["arguments"]) for call in calls)
    selection_exact = selected == case["expected_tools"]
    decision_correct = prediction["decision"] == case["expected_behavior"]
    return {
        "selected_tools": selected,
        "selection_exact": selection_exact,
        "schema_valid": schema_valid,
        "forbidden_tools": forbidden,
        "decision_correct": decision_correct,
        "routing_contract_success": selection_exact and schema_valid and decision_correct,
    }


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _request_prediction(
    case: Mapping[str, Any], *, base_url: str, api_key: str, model: str, timeout: float, retries: int
) -> Dict[str, Any]:
    endpoint = _endpoint(base_url)
    hostname = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    # Python's urllib may send loopback traffic through HTTP_PROXY even when
    # curl correctly bypasses it. A local SSH tunnel must be contacted
    # directly or a 200-case run can stall inside an unrelated desktop proxy.
    direct_opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if hostname in {"127.0.0.1", "localhost", "::1"}
        else None
    )
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_model_prompt(case)},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    last_content: str | None = None
    started = time.perf_counter()
    for attempt in range(retries + 1):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers=headers,
        )
        try:
            open_request = direct_opener.open if direct_opener is not None else urllib.request.urlopen
            with open_request(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            last_content = content
            prediction, normalizations = _parse_prediction_with_diagnostics(content)
            return {
                "prediction": prediction,
                "raw_content": content,
                "normalizations": normalizations,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "attempts": attempt + 1,
                "usage": body.get("usage", {}),
                "error": None,
            }
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    return {
        "prediction": None,
        "raw_content": last_content,
        "normalizations": [],
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "attempts": retries + 1,
        "usage": {},
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def run(
    cases: Sequence[Mapping[str, Any]], *, base_url: str, api_key: str, model: str,
    workers: int, timeout: float, retries: int,
) -> Dict[str, Any]:
    def evaluate(case: Mapping[str, Any]) -> Dict[str, Any]:
        response = _request_prediction(
            case, base_url=base_url, api_key=api_key, model=model, timeout=timeout, retries=retries
        )
        record: Dict[str, Any] = {
            "case_id": case["case_id"],
            "category": case["category"],
            "expected_tools": case["expected_tools"],
            "expected_behavior": case["expected_behavior"],
            **response,
        }
        if response["prediction"] is not None:
            record["score"] = score_prediction(case, response["prediction"])
        else:
            record["score"] = {
                "selected_tools": [], "selection_exact": False, "schema_valid": False,
                "forbidden_tools": [], "decision_correct": False, "routing_contract_success": False,
            }
        return record

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(evaluate, cases))
    n = len(records)
    successful_responses = sum(record["prediction"] is not None for record in records)
    metrics = {
        "case_count": n,
        "response_success_rate": successful_responses / n,
        "raw_contract_valid_rate": sum(
            record["prediction"] is not None and not record["normalizations"] for record in records
        ) / n,
        "normalized_response_count": sum(bool(record["normalizations"]) for record in records),
        "tool_selection_accuracy": sum(record["score"]["selection_exact"] for record in records) / n,
        "schema_valid_rate": sum(record["score"]["schema_valid"] for record in records) / n,
        "decision_accuracy": sum(record["score"]["decision_correct"] for record in records) / n,
        "routing_contract_success_rate": sum(record["score"]["routing_contract_success"] for record in records) / n,
        "forbidden_tool_execution_count": sum(bool(record["score"]["forbidden_tools"]) for record in records),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    return {"artifact_type": "toolbench_routing_run", "model": model, "workers": workers, "metrics": metrics, "cases": records}


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def load_toolbench_dataset(path: Path) -> tuple[List[Dict[str, Any]], str]:
    payload = path.read_bytes()
    cases: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        case = json.loads(raw)
        if not isinstance(case, dict):
            raise ValueError(f"dataset line {line_number} must be an object")
        case_id = case.get("case_id", case.get("id"))
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"dataset line {line_number} has a missing or duplicate ID")
        if case.get("suite") != "agent" or not isinstance(case.get("query"), str):
            raise ValueError(f"dataset case {case_id} has an invalid suite or query")
        for field in ("expected_tools", "allowed_tools", "unavailable_tools"):
            if not isinstance(case.get(field), list):
                raise ValueError(f"dataset case {case_id} has invalid {field}")
        seen.add(case_id)
        case["case_id"] = case_id
        cases.append(case)
    if not cases:
        raise ValueError("dataset contains no cases")
    return cases, hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/toolbench_v1.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", help="Override JOY_AGENT_BASE_URL for this process")
    parser.add_argument("--model", help="Override FAST_RESEARCH_MODEL/LLM_MODEL for this process")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.retries not in range(0, 6):
        parser.error("workers must be 1..32 and retries must be 0..5")
    _load_dotenv(args.env_file)
    base_url = args.base_url or os.getenv("JOY_AGENT_BASE_URL", "")
    api_key = os.getenv("JOY_AGENT_API_KEY", "")
    model = args.model or os.getenv("FAST_RESEARCH_MODEL") or os.getenv("LLM_MODEL", "")
    if not base_url or not model:
        parser.error("JOY_AGENT_BASE_URL/--base-url and FAST_RESEARCH_MODEL/LLM_MODEL/--model are required")
    cases, dataset_hash = load_toolbench_dataset(args.dataset)
    if args.limit is not None:
        cases = cases[:args.limit]
    artifact = run(
        cases, base_url=base_url, api_key=api_key, model=model,
        workers=args.workers, timeout=args.timeout, retries=args.retries,
    )
    artifact["dataset_hash"] = dataset_hash
    _atomic_write(args.output, artifact)
    print(json.dumps(artifact["metrics"], ensure_ascii=False, sort_keys=True))
    return 0 if artifact["metrics"]["response_success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

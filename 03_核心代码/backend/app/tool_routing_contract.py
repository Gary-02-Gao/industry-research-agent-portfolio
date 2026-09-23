"""Shared, dependency-free contract for LLM tool routing.

The production controller and ToolBench import this module so that an eval
score measures the same routing policy that is used at runtime.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


SYSTEM_PROMPT = """你是工具路由器，只做决策，不回答用户问题，也不实际执行工具。

先按用户明确要求拆分语义步骤，再选择“完成这些步骤所需的最小充分工具集”：
1. 只选择消息中列出的允许工具，绝不输出未列出的名称。
2. 用户只要求一个操作时只调用一个工具，不要自行追加“分析”“画图”“搜索”等操作。
3. 只有用户明确要求分析、趋势、异常、比较或分布时才调用 data_analyzer；已有数据可以直接分析，不要追加 chart_generator。
4. 只有用户明确要求绘图、图表或可视化时才调用 chart_generator；已有数据可以直接绘图，不要先追加 data_analyzer。
5. 用户用“先……再……”“然后”或“严格按这两个步骤”明确要求多步操作时，每个步骤恰好映射为一次工具调用，按原顺序全部保留，不能省略后一步。
6. 没有必要调用工具时返回空数组和 answer_without_tool。改写、翻译、算术、排序、语法检查、解释已给内容和撰写结语都不调用工具。
7. 请求明确涉及未配置、未授权或越权资源时，不得改用其他工具，返回空数组和 reject_tool_execution。

工具边界：
- 出现“数据库、数据表、SQL、结构化数据库”时使用 text2sql，不得改成 knowledge_search；knowledge_search 只表示已授权文档知识库。
- finish 只用于用户明确说“停止调用工具”“结束本次研究”或同义的代理终止命令。不要因为文本中有“总结、结语、结束句、最终表达”等普通写作词就调用 finish。
- 多步任务的后续 data_analyzer 可以省略 data 以消费前一步结果；不得增加 data_source 等未声明参数。
- 后续 chart_generator 尚无真实数据时使用空数组作为 data，占位值会在执行时由编排器替换。

输出前自检：tool_calls 是否逐项对应用户明确要求、是否遗漏显式步骤、是否添加了用户未要求的步骤、是否全部位于允许列表、参数是否只有声明字段。检索/查询只取得数据，不能替代用户随后明确要求的分析或绘图步骤。

严格返回 JSON：
{"decision":"execute|answer_without_tool|reject_tool_execution","tool_calls":[{"name":"工具名","arguments":{}}]}
decision 是路由结果，不是工具名，只能从 execute、answer_without_tool、reject_tool_execution 三者中选一个；尤其禁止把 decision 写成 finish。
无论选择哪个 decision，JSON 都必须同时包含 decision 和 tool_calls 两个顶层字段：
- 调用工具：{"decision":"execute","tool_calls":[...]}
- 无需工具：{"decision":"answer_without_tool","tool_calls":[]}
- 拒绝工具：{"decision":"reject_tool_execution","tool_calls":[]}
多步任务按用户要求的执行顺序排列 tool_calls。每一步只能出现一次。arguments 必须满足提供的参数约束，不得添加未知参数。不要输出 Markdown、解释或 JSON 之外的文字。"""


def build_user_prompt(
    query: str,
    allowed_tools: Sequence[str],
    tool_descriptions: Mapping[str, str],
    tool_schemas: Mapping[str, Mapping[str, Any]],
) -> str:
    """Build a gold-label-free prompt from the runtime tool catalogue."""

    lines = []
    for name in allowed_tools:
        lines.append(
            f"- {name}: {tool_descriptions[name]} "
            f"参数={json.dumps(tool_schemas[name], ensure_ascii=False, sort_keys=True)}"
        )
    tools_text = "\n".join(lines) or "（无允许工具）"
    return f"用户请求：\n{query}\n\n本次允许工具及参数约束：\n{tools_text}"


def _strip_fence(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else value


def parse_routing_response(content: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a response with only intent-preserving envelope repairs."""

    decoded = json.loads(_strip_fence(content))
    if not isinstance(decoded, dict):
        raise ValueError("model output must be a JSON object")
    decision = decoded.get("decision")
    calls = decoded.get("tool_calls")
    normalizations: list[str] = []

    if (
        decision == "finish"
        and isinstance(calls, list)
        and len(calls) == 1
        and isinstance(calls[0], dict)
        and calls[0].get("name") == "finish"
    ):
        decision = "execute"
        normalizations.append("finish_decision_to_execute")
    if calls is None and decision in {"answer_without_tool", "reject_tool_execution"}:
        calls = []
        normalizations.append("missing_empty_tool_calls")

    if decision not in {"execute", "answer_without_tool", "reject_tool_execution"}:
        raise ValueError("invalid decision")
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be an array")
    normalized = []
    for call in calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            raise ValueError("each tool call requires string name and object arguments")
        normalized.append({"name": call["name"], "arguments": call["arguments"]})

    if decision == "execute" and not normalized:
        raise ValueError("execute requires at least one tool call")
    if decision != "execute" and normalized:
        raise ValueError("non-execute decision cannot contain tool calls")
    return {"decision": decision, "tool_calls": normalized}, normalizations

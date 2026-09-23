"""
ReAct Controller - Reasoning + Acting 循环决策框架

实现了业界领先的 ReAct 范式，优化版流程：
1. Plan (规划) - LLM 分解问题，生成多个搜索子查询
2. Execute (执行) - 并行执行所有搜索任务
3. Reflect (反思) - 评估信息是否充足，决定是否补充搜索
4. Synthesize (综合) - 整合信息，生成最终报告

核心优化：
- 子查询由 LLM 智能生成，而非简单使用原始问题
- 支持并行执行多个搜索，大幅提升效率
- 迭代式深入，直到信息充足
"""

import json
import logging
import asyncio
import os
import re
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from service.joy_agent_client import create_joy_agent_client
from tool_routing_contract import SYSTEM_PROMPT as TOOL_ROUTING_SYSTEM_PROMPT
from tool_routing_contract import build_user_prompt, parse_routing_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ToolType(Enum):
    """工具类型枚举"""
    WEB_SEARCH = "web_search"
    KNOWLEDGE_SEARCH = "knowledge_search"
    TEXT2SQL = "text2sql"
    DATA_ANALYZER = "data_analyzer"
    CHART_GENERATOR = "chart_generator"
    STOCK_QUERY = "stock_query"
    BIDDING_SEARCH = "bidding_search"
    FINISH = "finish"


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Optional[Callable] = None
    enabled: bool = True
    required_credentials: List[str] = field(default_factory=list)
    side_effect: str = "none"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "required_credentials": self.required_credentials,
            "side_effect": self.side_effect,
            "input_schema": self.input_schema,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class Action:
    """动作定义"""
    tool: str
    params: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        return cls(
            tool=data.get('tool', ''),
            params=data.get('params', {})
        )


@dataclass
class Thought:
    """思考结果"""
    reasoning: str  # 推理过程
    should_finish: bool  # 是否应该结束
    next_action: Optional[Action] = None  # 下一步动作
    confidence: float = 0.0  # 置信度


@dataclass
class Observation:
    """观察结果"""
    tool: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuery:
    """子查询定义"""
    query: str  # 搜索关键词
    purpose: str  # 查询目的
    tool: str  # 使用的工具 (web_search / knowledge_search)
    priority: int = 1  # 优先级 1-3


@dataclass
class ResearchPlan:
    """研究计划"""
    understanding: str  # 对问题的理解
    sub_queries: List[SubQuery]  # 子查询列表
    strategy: str  # 研究策略说明
    expected_aspects: List[str]  # 预期覆盖的方面


@dataclass
class ReActStep:
    """ReAct 单步记录"""
    step: int
    thought: Thought
    action: Optional[Action]
    observation: Optional[Observation]


class ReActContext:
    """ReAct 上下文管理"""

    def __init__(self, query: str):
        self.query = query
        self.steps: List[ReActStep] = []
        self.observations: List[Observation] = []
        self.collected_data: List[Dict] = []  # 收集的数据
        self.insights: List[str] = []  # 发现的洞察
        self.charts: List[Dict] = []  # 生成的图表
        self.metadata: Dict[str, Any] = {}
        self.plan: Optional[ResearchPlan] = None  # 研究计划
        self.executed_queries: List[str] = []  # 已执行的查询
        self.iteration: int = 0  # 当前迭代轮次

    def add_step(self, step: ReActStep):
        self.steps.append(step)

    def add_observation(self, obs: Observation):
        self.observations.append(obs)

        # 如果是搜索结果，加入收集的数据
        if obs.tool in [ToolType.WEB_SEARCH.value, ToolType.KNOWLEDGE_SEARCH.value]:
            if obs.success and isinstance(obs.result, list):
                self.collected_data.extend(obs.result)

        # 如果是数据分析结果，记录洞察
        if obs.tool == ToolType.DATA_ANALYZER.value and obs.success:
            if isinstance(obs.result, dict) and 'insights' in obs.result:
                self.insights.extend(obs.result['insights'])

        # 如果是图表，记录
        if obs.tool == ToolType.CHART_GENERATOR.value and obs.success:
            self.charts.append(obs.result)

    def get_history_summary(self, max_items: int = 10) -> str:
        """获取历史摘要"""
        if not self.steps:
            return "尚未执行任何步骤。"

        summary_parts = []
        for step in self.steps[-max_items:]:
            step_summary = f"步骤 {step.step}:\n"
            step_summary += f"  思考: {step.thought.reasoning[:200]}...\n"
            if step.action:
                step_summary += f"  动作: {step.action.tool}({json.dumps(step.action.params, ensure_ascii=False)[:100]})\n"
            if step.observation:
                result_str = str(step.observation.result)[:200] if step.observation.result else "无结果"
                step_summary += f"  观察: {'成功' if step.observation.success else '失败'} - {result_str}\n"
            summary_parts.append(step_summary)

        return "\n".join(summary_parts)

    def get_collected_data_summary(self, max_items: int = 20) -> str:
        """获取收集数据摘要"""
        if not self.collected_data:
            return "尚未收集到数据。"

        summaries = []
        for i, item in enumerate(self.collected_data[:max_items]):
            if isinstance(item, dict):
                title = item.get('name', item.get('title', 'N/A'))
                content = item.get('summary', item.get('content', ''))[:150]
                source = item.get('source', 'unknown')
                summaries.append(f"[{i+1}] ({source}) {title}: {content}...")
            else:
                summaries.append(f"[{i+1}] {str(item)[:200]}...")

        return "\n".join(summaries)


class ReActController:
    """
    ReAct 控制器 - 核心推理引擎

    优化版流程：Plan -> Execute (并行) -> Reflect -> Synthesize
    """

    # ========== Plan 阶段 Prompt ==========
    PLAN_PROMPT = """你是一个专业的行业研究助手。请分析用户问题，制定研究计划并生成多个搜索子查询。

## 用户问题
{query}

## 本次请求允许的搜索工具
{available_search_tools}

## 任务要求
1. 深入理解用户问题的核心需求
2. 将问题分解为多个可搜索的子问题
3. 为每个子问题生成精准的搜索关键词
4. 确保子查询覆盖问题的各个方面

## 响应格式
请严格按照以下 JSON 格式响应：
```json
{{
    "understanding": "对用户问题的理解和分析",
    "sub_queries": [
        {{
            "query": "搜索关键词1（精准、具体）",
            "purpose": "这个查询的目的",
            "tool": "web_search",
            "priority": 1
        }},
        {{
            "query": "搜索关键词2",
            "purpose": "这个查询的目的",
            "tool": "web_search",
            "priority": 1
        }},
        {{
            "query": "搜索关键词3",
            "purpose": "这个查询的目的",
            "tool": "knowledge_search",
            "priority": 2
        }}
    ],
    "strategy": "整体研究策略说明",
    "expected_aspects": ["方面1", "方面2", "方面3"]
}}
```

## 子查询生成原则
1. 每个 query 必须是具体的搜索关键词，不能是问句
2. 生成 3-6 个不同角度的子查询
3. 关键词要精准，避免过于宽泛
4. 优先级：1=核心必需，2=重要补充，3=扩展了解
5. tool 只能从“本次请求允许的搜索工具”中选择，禁止输出未列出的工具

## 示例
用户问题："新能源汽车市场现状和发展趋势"
好的子查询：
- "2024年中国新能源汽车销量数据"
- "新能源汽车市场份额排名"
- "新能源汽车行业政策补贴"
- "电动汽车技术发展趋势"
- "新能源汽车企业竞争格局"

请开始分析："""

    # ========== Reflect 阶段 Prompt ==========
    REFLECT_PROMPT = """你是一个专业的研究助手。请评估当前收集的信息是否足够回答用户问题。

## 用户原始问题
{query}

## 研究计划预期覆盖的方面
{expected_aspects}

## 已收集的信息摘要
{collected_summary}

## 已执行的搜索查询
{executed_queries}

## 本次请求允许的搜索工具
{available_search_tools}

## 任务要求
评估当前信息的完整性，决定是否需要补充搜索。

## 响应格式
```json
{{
    "coverage_analysis": "对信息覆盖度的分析",
    "missing_aspects": ["缺失的方面1", "缺失的方面2"],
    "is_sufficient": true或false,
    "additional_queries": [
        {{
            "query": "补充搜索关键词",
            "purpose": "补充搜索的目的",
            "tool": "web_search"
        }}
    ],
    "confidence": 0.8
}}
```

注意：
- 如果信息已足够，设置 is_sufficient 为 true，additional_queries 为空数组
- 如果需要补充，生成 1-3 个精准的补充查询
- 最多进行 2 轮补充搜索，避免无限循环

请开始评估："""

    # ========== 传统 ReAct Prompt (备用) ==========
    REACT_PROMPT_TEMPLATE = """你是一个专业的行业研究助手，使用 ReAct 框架进行智能研究。

## 当前研究任务
{query}

## 可用工具
{tools_description}

## 执行历史
{history}

## 已收集的数据摘要
{data_summary}

## 响应格式
```json
{{
    "thought": "你的思考过程",
    "should_finish": false,
    "action": {{
        "tool": "工具名称",
        "params": {{
            "query": "具体参数值"
        }}
    }},
    "confidence": 0.8
}}
```

请开始推理："""

    TOOLS_DESCRIPTION_TEMPLATE = """{tools_list}"""

    def __init__(
        self,
        tools: List[Tool],
        llm_api_key: str,
        llm_base_url: str,
        max_steps: int = 10,
        max_parallel_tools: int = 3,
        model: str = "DeepSeek-V4-pro"
    ):
        """
        初始化 ReAct 控制器

        Args:
            tools: 可用工具列表
            llm_api_key: LLM API 密钥
            llm_base_url: LLM API 基础 URL
            max_steps: 最大执行步骤数
            model: 使用的模型名称
        """
        self.tool_catalog = {t.name: t for t in tools}
        # The planner must never see unavailable tools.
        self.tools = {t.name: t for t in tools if t.enabled}
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.max_steps = max_steps
        self.max_parallel_tools = max(1, min(int(max_parallel_tools), 3))
        self.model = model
        self.client = create_joy_agent_client(api_key=llm_api_key, base_url=llm_base_url)

    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        tools_list = []
        for name, tool in self.tools.items():
            params_str = ", ".join([f"{k}({v})" for k, v in tool.parameters.items()])
            schema = json.dumps(tool.input_schema, ensure_ascii=False, sort_keys=True)
            tools_list.append(
                f"- {name}: {tool.description}\n"
                f"  参数: {params_str}\n"
                f"  输入Schema: {schema}\n"
                f"  副作用: {tool.side_effect}"
            )

        return self.TOOLS_DESCRIPTION_TEMPLATE.format(
            tools_list="\n".join(tools_list)
        )

    def _available_routing_tools(self, context: ReActContext) -> List[str]:
        """Return the enabled tools that this individual request may invoke."""

        requested = context.metadata.get("allowed_tools")
        requested_set = set(requested) if isinstance(requested, list) else None
        available = []
        for name in self.tools:
            if requested_set is not None and name not in requested_set:
                continue
            if name == ToolType.WEB_SEARCH.value and not context.metadata.get("search_web", True):
                continue
            if name == ToolType.KNOWLEDGE_SEARCH.value and (
                not context.metadata.get("search_local", True)
                or not context.metadata.get("kb_name")
            ):
                continue
            available.append(name)
        return available

    async def _route_request(self, context: ReActContext) -> Dict[str, Any]:
        """Ask the model for a schema-bound route without executing anything."""

        allowed_tools = self._available_routing_tools(context)
        descriptions = {name: self.tools[name].description for name in allowed_tools}
        schemas = {name: self.tools[name].input_schema for name in allowed_tools}
        prompt = build_user_prompt(context.query, allowed_tools, descriptions, schemas)
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": TOOL_ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=512,
        )
        content = response.choices[0].message.content
        provider_usage = getattr(response, "usage", None)
        prompt_tokens = getattr(provider_usage, "prompt_tokens", None)
        completion_tokens = getattr(provider_usage, "completion_tokens", None)
        usage = {
            "model": self.model,
            "input_tokens": int(prompt_tokens or 0),
            "output_tokens": int(completion_tokens or 0),
            "total_tokens": int(getattr(provider_usage, "total_tokens", 0) or 0),
            "usage_observed": prompt_tokens is not None and completion_tokens is not None,
            "usage_source": "provider" if prompt_tokens is not None and completion_tokens is not None else "unavailable",
        }
        route, normalizations = parse_routing_response(content)

        if route["decision"] == "reject_tool_execution":
            return {**route, "normalizations": normalizations, "allowed_tools": allowed_tools, "_llm_usage": usage}
        for call in route["tool_calls"]:
            if call["name"] not in allowed_tools:
                raise ValueError(f"路由器选择了未授权或不可用工具: {call['name']}")
            tool = self.tools[call["name"]]
            params = self._materialize_routed_params(call["name"], call["arguments"], context)
            schema_error = self._validate_tool_input(tool, params)
            if schema_error:
                raise ValueError(f"工具 {call['name']} 参数无效: {schema_error}")
            call["arguments"] = params
        return {**route, "normalizations": normalizations, "allowed_tools": allowed_tools, "_llm_usage": usage}

    @staticmethod
    def _latest_observation_data(context: ReActContext) -> List[Any]:
        """Convert the latest successful tool result to downstream list input."""

        for observation in reversed(context.observations):
            if not observation.success:
                continue
            result = observation.result
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("data", "rows", "results"):
                    if isinstance(result.get(key), list):
                        return result[key]
                return [result]
        return list(context.collected_data)

    def _materialize_routed_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: ReActContext,
    ) -> Dict[str, Any]:
        """Bind trusted scope and pass prior results to deterministic tools."""

        materialized = dict(params)
        if tool_name == ToolType.KNOWLEDGE_SEARCH.value:
            # The model must never choose a collection. Always bind the KB that
            # the authenticated request resolver placed in the context.
            materialized["kb_name"] = context.metadata.get("kb_name", "")
        if tool_name in {ToolType.DATA_ANALYZER.value, ToolType.CHART_GENERATOR.value}:
            if not materialized.get("data"):
                materialized["data"] = self._latest_observation_data(context)
        return materialized

    async def _run_routed_tools(
        self,
        context: ReActContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single bounded model route through real tool handlers."""

        yield {"type": "react_start", "query": context.query, "mode": "tool_routing"}
        yield {"type": "status", "content": "正在生成最小工具执行计划..."}
        try:
            route = await self._route_request(context)
        except Exception as exc:
            logging.error("Tool routing failed closed: %s", exc)
            yield {
                "type": "routing_error",
                "content": str(exc),
                "executed_tool_count": 0,
            }
            yield {
                "type": "react_complete",
                "routing_decision": "reject_tool_execution",
                "total_steps": 0,
                "collected_data": context.collected_data,
                "insights": context.insights,
                "charts": context.charts,
            }
            return

        yield {"type": "llm_usage", **route["_llm_usage"]}
        yield {
            "type": "tool_plan",
            "decision": route["decision"],
            "tool_calls": route["tool_calls"],
            "allowed_tools": route["allowed_tools"],
            "normalizations": route["normalizations"],
        }
        if route["decision"] != "execute":
            yield {
                "type": "react_complete",
                "routing_decision": route["decision"],
                "total_steps": 0,
                "collected_data": context.collected_data,
                "insights": context.insights,
                "charts": context.charts,
            }
            return

        calls = route["tool_calls"]
        omitted = max(0, len(calls) - self.max_steps)
        calls = calls[: self.max_steps]
        executed = 0
        for step, call in enumerate(calls, 1):
            # Materialize again here because downstream input depends on the
            # observation produced by the immediately preceding call.
            params = self._materialize_routed_params(call["name"], call["arguments"], context)
            action = Action(tool=call["name"], params=params)
            yield {"type": "action", "step": step, "tool": action.tool, "params": action.params}
            observation = await self._execute_action(action, context)
            executed += 1
            context.add_observation(observation)
            yield {
                "type": "observation",
                "step": step,
                "tool": action.tool,
                "success": observation.success,
                "result": observation.result,
                "error": observation.error,
                "metadata": observation.metadata,
            }
            if not observation.success:
                break

        if omitted:
            yield {
                "type": "budget_exhausted",
                "budget": "max_tool_calls",
                "limit": self.max_steps,
                "omitted_tool_count": omitted,
            }
        yield {
            "type": "react_complete",
            "routing_decision": route["decision"],
            "total_steps": executed,
            "collected_data": context.collected_data,
            "insights": context.insights,
            "charts": context.charts,
        }

    def _build_prompt(self, context: ReActContext) -> str:
        """构建 ReAct 提示词"""
        return self.REACT_PROMPT_TEMPLATE.format(
            query=context.query,
            tools_description=self._format_tools_description(),
            history=context.get_history_summary(),
            data_summary=context.get_collected_data_summary()
        )

    async def _think(self, context: ReActContext) -> Thought:
        """
        执行思考步骤 - 调用 LLM 进行推理

        Args:
            context: 当前上下文

        Returns:
            Thought 对象，包含推理结果和下一步动作
        """
        prompt = self._build_prompt(context)

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的行业研究助手，擅长使用各种工具进行深度研究。请严格按照 JSON 格式响应，所有工具调用必须提供完整的params参数。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2  # 降低温度以获得更稳定的输出
            )

            content = response.choices[0].message.content
            logging.info(f"ReAct thinking response: {content[:500]}...")

            # 解析响应
            result = json.loads(content)

            action = None
            if result.get('action'):
                tool_name = result['action'].get('tool', '')
                params = result['action'].get('params', {})

                # 验证和修复 params
                params = self._validate_and_fix_params(tool_name, params, result.get('thought', ''), context)

                action = Action(
                    tool=tool_name,
                    params=params
                )

            return Thought(
                reasoning=result.get('thought', ''),
                should_finish=result.get('should_finish', False),
                next_action=action,
                confidence=result.get('confidence', 0.5)
            )

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            return Thought(
                reasoning=f"解析响应失败: {e}",
                should_finish=False,
                confidence=0.0
            )
        except Exception as e:
            logging.error(f"Error during thinking: {e}")
            return Thought(
                reasoning=f"思考过程出错: {e}",
                should_finish=True,
                confidence=0.0
            )

    def _validate_and_fix_params(self, tool_name: str, params: Dict, thought: str, context: ReActContext) -> Dict:
        """
        验证并修复工具参数

        如果 LLM 生成了空的 params，尝试从 thought 或 context 中提取参数
        """
        if tool_name == ToolType.WEB_SEARCH.value:
            if not params.get('query'):
                # 尝试从 thought 中提取搜索关键词
                extracted_query = self._extract_search_query_from_thought(thought, context)
                if extracted_query:
                    params['query'] = extracted_query
                    logging.info(f"Extracted query from thought: {extracted_query}")
                else:
                    # 使用原始问题作为备选
                    params['query'] = context.query
                    logging.warning(f"Using context query as fallback: {context.query}")
            if not params.get('count'):
                params['count'] = 5

        elif tool_name == ToolType.KNOWLEDGE_SEARCH.value:
            if not params.get('query'):
                params['query'] = context.query
                logging.warning(f"Using context query for knowledge_search: {context.query}")
            if not params.get('top_k'):
                params['top_k'] = 5

        elif tool_name == ToolType.FINISH.value:
            if not params.get('summary'):
                params['summary'] = f"完成对 '{context.query}' 的研究"

        return params

    def _extract_search_query_from_thought(self, thought: str, context: ReActContext) -> Optional[str]:
        """
        从思考内容中提取搜索关键词

        尝试从 thought 文本中识别用户意图的搜索词
        """
        # 常见的搜索意图表达模式
        patterns = [
            r'搜索[「"\'【](.+?)[」"\'】]',
            r'查找[「"\'【](.+?)[」"\'】]',
            r'搜索关于(.+?)的',
            r'查询(.+?)的信息',
            r'了解(.+?)的',
            r'获取(.+?)的',
        ]

        for pattern in patterns:
            match = re.search(pattern, thought)
            if match:
                return match.group(1).strip()

        # 如果没有匹配到，返回 None，让调用者使用备选方案
        return None

    async def _execute_action(self, action: Action, context: ReActContext) -> Observation:
        """
        执行动作 - 调用相应的工具

        Args:
            action: 要执行的动作
            context: 当前上下文

        Returns:
            Observation 对象，包含执行结果
        """
        started = time.perf_counter()
        span_id = f"tool-{uuid.uuid4().hex[:16]}"

        def span_metadata(**extra: Any) -> Dict[str, Any]:
            return {
                "span_id": span_id,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                **extra,
            }

        tool = self.tool_catalog.get(action.tool)
        if tool is None:
            return Observation(
                tool=action.tool,
                success=False,
                result=None,
                error=f"未知工具: {action.tool}",
                metadata=span_metadata(executed=False, schema_valid=False),
            )

        if not tool.enabled:
            return Observation(
                tool=action.tool,
                success=False,
                result=None,
                error=tool.unavailable_reason or f"工具 {action.tool} 不可用",
                metadata=span_metadata(executed=False, unavailable=True),
            )

        if not tool.handler:
            return Observation(
                tool=action.tool,
                success=False,
                result=None,
                error=f"工具 {action.tool} 未配置处理器",
                metadata=span_metadata(executed=False),
            )

        schema_error = self._validate_tool_input(tool, action.params)
        if schema_error:
            return Observation(
                tool=action.tool,
                success=False,
                result=None,
                error=schema_error,
                metadata=span_metadata(params=action.params, executed=False, schema_valid=False),
            )

        try:
            # 执行工具
            result = await tool.handler(action.params, context)

            if isinstance(result, dict) and result.get("success") is False:
                return Observation(
                    tool=action.tool,
                    success=False,
                    result=result,
                    error=str(result.get("error") or "工具返回失败"),
                    metadata=span_metadata(
                        params=action.params, executed=True, schema_valid=True,
                    ),
                )
            return Observation(
                tool=action.tool,
                success=True,
                result=result,
                metadata=span_metadata(
                    params=action.params, executed=True, schema_valid=True,
                ),
            )

        except Exception as e:
            logging.error(f"Error executing tool {action.tool}: {e}")
            return Observation(
                tool=action.tool,
                success=False,
                result=None,
                error=str(e),
                metadata=span_metadata(
                    params=action.params, executed=True, schema_valid=True,
                    exception_type=type(e).__name__,
                ),
            )

    @staticmethod
    def _validate_tool_input(tool: Tool, params: Dict[str, Any]) -> Optional[str]:
        """Validate the small JSON-Schema subset used by registered tools."""
        if not isinstance(params, dict):
            return "工具参数必须是对象"
        schema = tool.input_schema or {}
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in params or params[required] in (None, ""):
                return f"缺少必填参数: {required}"
        python_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "array": list,
            "object": dict,
            "boolean": bool,
        }
        for name, value in params.items():
            if name not in properties:
                if schema.get("additionalProperties") is False:
                    return f"未知参数: {name}"
                continue
            spec = properties[name]
            expected = python_types.get(spec.get("type"))
            if expected and (not isinstance(value, expected) or isinstance(value, bool) and spec.get("type") != "boolean"):
                return f"参数类型错误: {name}"
            if "enum" in spec and value not in spec["enum"]:
                return f"参数枚举值无效: {name}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in spec and value < spec["minimum"]:
                    return f"参数小于最小值: {name}"
                if "maximum" in spec and value > spec["maximum"]:
                    return f"参数大于最大值: {name}"
            if isinstance(value, str) and len(value) < int(spec.get("minLength", 0)):
                return f"参数字符串过短: {name}"
        if tool.name == ToolType.STOCK_QUERY.value and not (
            str(params.get("stock_code", "")).strip()
            or str(params.get("keyword", "")).strip()
        ):
            return "stock_code 和 keyword 至少一个非空"
        return None

    # ========== Plan 阶段：生成研究计划和子查询 ==========
    async def _generate_plan(self, context: ReActContext) -> ResearchPlan:
        """
        生成研究计划，包含多个子查询

        Args:
            context: ReAct 上下文

        Returns:
            ResearchPlan 对象
        """
        available_tools = self._available_search_tools(context)
        if not available_tools:
            return ResearchPlan(
                understanding=f"研究问题: {context.query}",
                sub_queries=[],
                strategy="当前请求没有可用且获授权的搜索工具",
                expected_aspects=[],
            )
        prompt = self.PLAN_PROMPT.format(
            query=context.query,
            available_search_tools=", ".join(available_tools),
        )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的研究规划师，擅长将复杂问题分解为可执行的搜索任务。请严格按照 JSON 格式响应。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            content = response.choices[0].message.content
            logging.info(f"Plan generation response: {content[:500]}...")

            result = json.loads(content)

            # 解析子查询
            sub_queries = []
            for sq in result.get('sub_queries', []):
                tool_name = sq.get('tool', available_tools[0])
                query = sq.get('query', '')
                if tool_name not in available_tools or not isinstance(query, str) or not query.strip():
                    continue
                sub_queries.append(SubQuery(
                    query=query.strip(),
                    purpose=sq.get('purpose', ''),
                    tool=tool_name,
                    priority=sq.get('priority', 1)
                ))

            # 如果没有生成子查询，使用原始问题创建默认子查询
            if not sub_queries:
                sub_queries = [
                    SubQuery(query=context.query, purpose="原始问题搜索", tool=available_tools[0], priority=1),
                ]

            return ResearchPlan(
                understanding=result.get('understanding', ''),
                sub_queries=sub_queries,
                strategy=result.get('strategy', ''),
                expected_aspects=result.get('expected_aspects', [])
            )

        except Exception as e:
            logging.error(f"Error generating plan: {e}")
            # 返回默认计划
            return ResearchPlan(
                understanding=f"研究问题: {context.query}",
                sub_queries=[
                    SubQuery(query=context.query, purpose="主要搜索", tool=available_tools[0], priority=1),
                ],
                strategy="直接搜索",
                expected_aspects=["基本信息"]
            )

    # ========== Execute 阶段：并行执行多个搜索 ==========
    async def _execute_queries_parallel(
        self,
        queries: List[SubQuery],
        context: ReActContext
    ) -> List[Tuple[SubQuery, Observation]]:
        """
        并行执行多个搜索查询

        Args:
            queries: 子查询列表
            context: ReAct 上下文

        Returns:
            (SubQuery, Observation) 元组列表
        """
        semaphore = asyncio.Semaphore(self.max_parallel_tools)

        async def execute_single_query(sq: SubQuery) -> Tuple[SubQuery, Observation]:
            async with semaphore:
                if sq.tool == ToolType.KNOWLEDGE_SEARCH.value:
                    params = {
                        "query": sq.query,
                        "kb_name": context.metadata.get("kb_name", ""),
                        "top_k": 5,
                    }
                else:
                    params = {"query": sq.query, "count": 5}
                action = Action(
                    tool=sq.tool,
                    params=params,
                )
                observation = await self._execute_action(action, context)
                return (sq, observation)

        # 并行执行所有查询
        tasks = [execute_single_query(sq) for sq in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Query execution error: {result}")
                continue
            valid_results.append(result)

        return valid_results

    # ========== Reflect 阶段：评估信息是否充足 ==========
    async def _reflect(self, context: ReActContext) -> Dict[str, Any]:
        """
        反思阶段：评估收集的信息是否足够

        Args:
            context: ReAct 上下文

        Returns:
            包含评估结果的字典
        """
        expected_aspects = context.plan.expected_aspects if context.plan else []

        available_tools = self._available_search_tools(context)
        prompt = self.REFLECT_PROMPT.format(
            query=context.query,
            expected_aspects=", ".join(expected_aspects) if expected_aspects else "未指定",
            collected_summary=context.get_collected_data_summary(),
            executed_queries=", ".join(context.executed_queries) if context.executed_queries else "无",
            available_search_tools=", ".join(available_tools) or "无",
        )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的研究评估师，擅长评估信息完整性。请严格按照 JSON 格式响应。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            content = response.choices[0].message.content
            logging.info(f"Reflect response: {content[:500]}...")

            result = json.loads(content)

            # 解析补充查询
            additional_queries = []
            for aq in result.get('additional_queries', []):
                tool_name = aq.get('tool', available_tools[0] if available_tools else '')
                query = aq.get('query', '')
                if tool_name not in available_tools or not isinstance(query, str) or not query.strip():
                    continue
                additional_queries.append(SubQuery(
                    query=query.strip(),
                    purpose=aq.get('purpose', ''),
                    tool=tool_name,
                    priority=2
                ))

            return {
                "coverage_analysis": result.get('coverage_analysis', ''),
                "missing_aspects": result.get('missing_aspects', []),
                "is_sufficient": result.get('is_sufficient', True),
                "additional_queries": additional_queries,
                "confidence": result.get('confidence', 0.5)
            }

        except Exception as e:
            logging.error(f"Error during reflection: {e}")
            return {
                "coverage_analysis": "评估出错",
                "missing_aspects": [],
                "is_sufficient": True,  # 出错时默认结束
                "additional_queries": [],
                "confidence": 0.0
            }

    def _available_search_tools(self, context: ReActContext) -> List[str]:
        """Return enabled search tools allowed by this request's data scope."""

        available: List[str] = []
        if context.metadata.get("search_web", True) and ToolType.WEB_SEARCH.value in self.tools:
            available.append(ToolType.WEB_SEARCH.value)
        if (
            context.metadata.get("search_local", True)
            and context.metadata.get("kb_name")
            and ToolType.KNOWLEDGE_SEARCH.value in self.tools
        ):
            available.append(ToolType.KNOWLEDGE_SEARCH.value)
        return available

    # ========== 主运行循环：优化版 ==========
    async def run(
        self,
        query: str,
        initial_context: Optional[Dict] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行优化版 ReAct 循环: Plan -> Execute (并行) -> Reflect -> Synthesize

        Args:
            query: 用户查询
            initial_context: 初始上下文数据

        Yields:
            包含事件类型和数据的字典
        """
        context = ReActContext(query)
        if initial_context:
            context.metadata.update(initial_context)

        if context.metadata.get("routing_mode") == "tools":
            async for event in self._run_routed_tools(context):
                yield event
            return

        requested_iterations = context.metadata.get("max_iterations", 3)
        try:
            max_iterations = max(1, min(int(requested_iterations), 3))
        except (TypeError, ValueError):
            max_iterations = 3
        tool_calls_used = 0
        step = 0

        yield {"type": "react_start", "query": query, "mode": "optimized"}

        # ========== Phase 1: Plan ==========
        yield {"type": "status", "content": "正在分析问题，生成研究计划..."}

        plan = await self._generate_plan(context)
        context.plan = plan

        yield {
            "type": "thought",
            "step": 1,
            "content": f"**问题理解**: {plan.understanding}\n\n**研究策略**: {plan.strategy}",
            "confidence": 0.9
        }

        yield {
            "type": "plan",
            "understanding": plan.understanding,
            "strategy": plan.strategy,
            "sub_queries": [{"query": sq.query, "purpose": sq.purpose, "tool": sq.tool} for sq in plan.sub_queries],
            "expected_aspects": plan.expected_aspects
        }

        # ========== Phase 2 & 3: Execute & Reflect Loop ==========
        while context.iteration < max_iterations:
            context.iteration += 1
            step += 1

            # 获取当前要执行的查询
            if context.iteration == 1:
                # 第一轮：执行计划中的所有查询
                queries_to_execute = [sq for sq in plan.sub_queries if sq.priority <= 2]
            else:
                # 后续轮次：执行反思阶段生成的补充查询
                queries_to_execute = context.metadata.get('additional_queries', [])

            if not queries_to_execute:
                break

            remaining_tool_calls = max(0, self.max_steps - tool_calls_used)
            if remaining_tool_calls == 0:
                yield {
                    "type": "budget_exhausted",
                    "budget": "max_tool_calls",
                    "limit": self.max_steps,
                }
                break
            omitted_query_count = max(0, len(queries_to_execute) - remaining_tool_calls)
            queries_to_execute = queries_to_execute[:remaining_tool_calls]

            # 显示即将执行的搜索
            yield {
                "type": "action",
                "step": step,
                "tool": "parallel_search",
                "params": {"queries": [sq.query for sq in queries_to_execute]}
            }

            yield {"type": "status", "content": f"正在并行执行 {len(queries_to_execute)} 个搜索..."}

            # 并行执行搜索
            results = await self._execute_queries_parallel(queries_to_execute, context)
            tool_calls_used += len(queries_to_execute)

            # 处理结果
            total_results = 0
            for sq, obs in results:
                context.executed_queries.append(sq.query)
                context.add_observation(obs)

                if obs.success and isinstance(obs.result, list):
                    total_results += len(obs.result)
                    # 流式返回搜索结果
                    for item in obs.result:
                        yield {"type": "search_result_item", "result": item}

            yield {
                "type": "observation",
                "step": step,
                "tool": "parallel_search",
                "success": bool(results) and all(obs.success for _, obs in results),
                "result": f"并行搜索完成，共获取 {total_results} 条结果",
                "queries_executed": [sq.query for sq, _ in results],
                "failed_queries": [sq.query for sq, obs in results if not obs.success],
                "tool_calls_used": tool_calls_used,
                "tool_call_limit": self.max_steps,
                "omitted_query_count": omitted_query_count,
                "tool_spans": [
                    {"tool": sq.tool, **obs.metadata}
                    for sq, obs in results
                ],
            }

            if omitted_query_count:
                yield {
                    "type": "budget_exhausted",
                    "budget": "max_tool_calls",
                    "limit": self.max_steps,
                    "omitted_query_count": omitted_query_count,
                }

            # 如果是最后一轮或没有收集到数据，跳过反思
            if context.iteration >= max_iterations or not context.collected_data:
                break

            # ========== Reflect ==========
            step += 1
            yield {"type": "status", "content": "正在评估信息完整性..."}

            reflect_result = await self._reflect(context)

            yield {
                "type": "thought",
                "step": step,
                "content": f"**信息评估**: {reflect_result['coverage_analysis']}",
                "confidence": reflect_result['confidence']
            }

            # 如果信息充足，结束循环
            if reflect_result['is_sufficient']:
                yield {"type": "status", "content": "信息收集完成"}
                break

            # 如果有缺失方面，准备补充搜索
            if reflect_result['additional_queries']:
                context.metadata['additional_queries'] = reflect_result['additional_queries']
                yield {
                    "type": "status",
                    "content": f"发现信息缺口，将补充搜索: {', '.join([q.query for q in reflect_result['additional_queries']])}"
                }
            else:
                break

        # ========== Phase 4: Complete ==========
        yield {"type": "status", "content": "研究完成，准备生成报告"}

        yield {
            "type": "react_complete",
            "total_steps": step,
            "total_iterations": context.iteration,
            "collected_data": context.collected_data,
            "executed_queries": context.executed_queries,
            "insights": context.insights,
            "charts": context.charts
        }

    def register_tool(self, tool: Tool):
        """注册新工具"""
        self.tool_catalog[tool.name] = tool
        if tool.enabled:
            self.tools[tool.name] = tool
        else:
            self.tools.pop(tool.name, None)

    def update_tool_handler(self, tool_name: str, handler: Callable):
        """更新工具处理器"""
        if tool_name in self.tool_catalog:
            self.tool_catalog[tool_name].handler = handler
            if self.tool_catalog[tool_name].enabled:
                self.tools[tool_name] = self.tool_catalog[tool_name]


def create_default_tools(
    *,
    search_api_key: Optional[str] = None,
    db_connection_string: Optional[str] = None,
) -> List[Tool]:
    """创建默认工具集"""
    tools = [
        Tool(
            name=ToolType.WEB_SEARCH.value,
            description="搜索互联网获取最新信息，适用于查找实时数据、新闻、市场信息等",
            parameters={
                "query": "搜索关键词",
                "count": "返回结果数量(默认5)"
            }
        ),
        Tool(
            name=ToolType.KNOWLEDGE_SEARCH.value,
            description="搜索本地知识库获取专业文档，适用于查找内部资料、专业报告等",
            parameters={
                "query": "搜索问题",
                "kb_name": "知识库名称",
                "top_k": "返回结果数量"
            }
        ),
        Tool(
            name=ToolType.TEXT2SQL.value,
            description="将自然语言转换为SQL查询数据库，获取结构化数据",
            parameters={
                "question": "自然语言问题",
                "intent": "查询意图(stats/trend/comparison/detail)"
            }
        ),
        Tool(
            name=ToolType.DATA_ANALYZER.value,
            description="分析数据并识别模式、趋势、异常等，自动推荐可视化方式",
            parameters={
                "data": "待分析数据(可选，默认使用已收集数据)",
                "analysis_type": "分析类型(auto/trend/distribution/comparison)"
            }
        ),
        Tool(
            name=ToolType.CHART_GENERATOR.value,
            description="根据数据生成可视化图表配置，支持折线图、柱状图、饼图等",
            parameters={
                "data": "图表数据",
                "chart_type": "图表类型(line/bar/pie/scatter)",
                "title": "图表标题"
            }
        ),
        Tool(
            name=ToolType.STOCK_QUERY.value,
            description="查询股票实时行情信息，获取股票价格、涨跌幅、成交量等数据",
            parameters={
                "stock_code": "股票代码(如sh601009/sz000001，6开头为上证，0/3开头为深证)",
                "keyword": "股票名称或代码关键词(当不确定完整代码时使用)"
            }
        ),
        Tool(
            name=ToolType.BIDDING_SEARCH.value,
            description="搜索招投标信息，获取招标公告、中标信息、采购公告等",
            parameters={
                "keyword": "搜索关键词(必填)",
                "category": "项目类别(招标/中标/采购)",
                "region": "地区筛选",
                "page": "页码(默认1)"
            }
        ),
        Tool(
            name=ToolType.FINISH.value,
            description="完成研究任务，开始生成最终研究报告",
            parameters={
                "summary": "研究总结"
            }
        )
    ]

    schemas = {
        "web_search": ({"query": {"type": "string", "minLength": 1}, "count": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
        "knowledge_search": ({"query": {"type": "string", "minLength": 1}, "kb_name": {"type": "string", "minLength": 1}, "top_k": {"type": "integer", "minimum": 1, "maximum": 30}}, ["query", "kb_name"]),
        "text2sql": ({"question": {"type": "string", "minLength": 1}, "intent": {"type": "string", "enum": ["stats", "trend", "comparison", "detail"]}}, ["question"]),
        "data_analyzer": ({"data": {"type": "array"}, "analysis_type": {"type": "string", "enum": ["auto", "trend", "distribution", "comparison"]}}, []),
        "chart_generator": ({"data": {"type": "array"}, "chart_type": {"type": "string", "enum": ["line", "bar", "pie", "scatter"]}, "title": {"type": "string", "minLength": 1}}, ["data", "chart_type", "title"]),
        "stock_query": ({"stock_code": {"type": "string"}, "keyword": {"type": "string"}}, []),
        "bidding_search": ({"keyword": {"type": "string", "minLength": 1}, "category": {"type": "string", "enum": ["招标", "中标", "采购"]}, "region": {"type": "string"}, "page": {"type": "integer", "minimum": 1}}, ["keyword"]),
        "finish": ({"summary": {"type": "string", "minLength": 1}}, ["summary"]),
    }
    credentials = {
        "web_search": ["SERPER_API_KEY"],
        "knowledge_search": ["authenticated_kb_scope"],
        "text2sql": ["authenticated_data_source"],
        "stock_query": ["JUHE_STOCK_API_KEY"],
        "bidding_search": ["BID_APP_CODE"],
    }
    availability = {
        "web_search": bool(search_api_key or os.getenv("SERPER_API_KEY")),
        "text2sql": bool(db_connection_string),
        "stock_query": bool(os.getenv("JUHE_STOCK_API_KEY")),
        "bidding_search": bool(os.getenv("BID_APP_CODE")),
    }
    for tool in tools:
        properties, required = schemas[tool.name]
        tool.input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        tool.required_credentials = credentials.get(tool.name, [])
        tool.side_effect = "read" if tool.name not in {"data_analyzer", "chart_generator", "finish"} else "none"
        tool.enabled = availability.get(tool.name, True)
        if not tool.enabled:
            tool.unavailable_reason = "required credential or authenticated data source is unavailable"
    return tools

from __future__ import annotations

import unittest

from evals.toolbench_run import build_model_prompt, parse_prediction, score_prediction


class ToolBenchRunnerTests(unittest.TestCase):
    def _case(self) -> dict:
        return {
            "case_id": "TBV1-test",
            "query": "查询最新公开行业信息",
            "allowed_tools": ["web_search"],
            "expected_tools": ["web_search"],
            "expected_behavior": "execute",
        }

    def test_prompt_is_independent_of_hidden_gold(self) -> None:
        first = self._case()
        second = {**first, "expected_tools": [], "expected_behavior": "answer_without_tool"}
        self.assertEqual(build_model_prompt(first), build_model_prompt(second))
        self.assertNotIn("expected_tools", build_model_prompt(first))
        self.assertNotIn("expected_behavior", build_model_prompt(first))

    def test_router_policy_requires_minimal_explicit_tool_set(self) -> None:
        from evals.toolbench_run import SYSTEM_PROMPT

        self.assertIn("最小充分工具集", SYSTEM_PROMPT)
        self.assertIn("每个步骤恰好映射为一次工具调用", SYSTEM_PROMPT)
        self.assertIn("不要先追加 data_analyzer", SYSTEM_PROMPT)
        self.assertIn("不要追加 chart_generator", SYSTEM_PROMPT)

    def test_json_fence_is_accepted_but_shape_is_strict(self) -> None:
        prediction = parse_prediction(
            '```json\n{"decision":"execute","tool_calls":[{"name":"web_search","arguments":{"query":"行业","count":5}}]}\n```'
        )
        self.assertEqual("web_search", prediction["tool_calls"][0]["name"])
        with self.assertRaises(ValueError):
            parse_prediction('{"decision":"execute","tool_calls":[{"name":"web_search"}]}')

    def test_safe_envelope_normalization_never_invents_tool_calls(self) -> None:
        finish = parse_prediction(
            '{"decision":"finish","tool_calls":[{"name":"finish","arguments":{"summary":"done"}}]}'
        )
        self.assertEqual("execute", finish["decision"])
        rejected = parse_prediction('{"decision":"reject_tool_execution"}')
        self.assertEqual([], rejected["tool_calls"])
        with self.assertRaises(ValueError):
            parse_prediction('{"decision":"execute"}')
        with self.assertRaises(ValueError):
            parse_prediction('{"decision":"web_search","tool_calls":[]}')

    def test_exact_selection_and_schema_are_scored_separately(self) -> None:
        case = self._case()
        valid = score_prediction(case, {
            "decision": "execute",
            "tool_calls": [{"name": "web_search", "arguments": {"query": "行业", "count": 5}}],
        })
        self.assertTrue(valid["routing_contract_success"])

        invalid_schema = score_prediction(case, {
            "decision": "execute",
            "tool_calls": [{"name": "web_search", "arguments": {"query": "行业", "count": 99}}],
        })
        self.assertTrue(invalid_schema["selection_exact"])
        self.assertFalse(invalid_schema["schema_valid"])

    def test_forbidden_tool_is_never_schema_valid(self) -> None:
        case = {**self._case(), "allowed_tools": []}
        score = score_prediction(case, {
            "decision": "execute",
            "tool_calls": [{"name": "web_search", "arguments": {"query": "行业"}}],
        })
        self.assertEqual(["web_search"], score["forbidden_tools"])
        self.assertFalse(score["schema_valid"])


if __name__ == "__main__":
    unittest.main()

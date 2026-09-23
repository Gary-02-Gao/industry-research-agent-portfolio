from __future__ import annotations

import random
import unittest

from evals.metrics import (
    answerable_refusal_metrics,
    citation_metrics,
    infer_refusal,
    mean_difference,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    retrieval_metrics_at_k,
    tool_routing_metrics,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_precision_and_recall_at_k_use_rank_slots_and_unique_hits(self) -> None:
        retrieved = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "b"}, {"doc_id": "c"}]
        relevant = ["b", "c", "d"]

        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 3), 1 / 3)
        self.assertEqual(
            retrieval_metrics_at_k(retrieved, relevant, 2),
            {"retrieval_precision@2": 0.5, "retrieval_recall@2": 1 / 3},
        )

    def test_empty_gold_and_invalid_k(self) -> None:
        self.assertEqual(precision_at_k(["a"], [], 1), 0.0)
        self.assertEqual(recall_at_k(["a"], [], 1), 0.0)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                precision_at_k([], [], invalid)  # type: ignore[arg-type]


class CitationMetricsTests(unittest.TestCase):
    def test_overlap_metrics(self) -> None:
        scores = citation_metrics(
            [{"citation_id": "a"}, {"citation_id": "b"}],
            [{"citation_id": "b"}, {"citation_id": "c"}],
        )
        self.assertEqual(scores["citation_precision"], 0.5)
        self.assertEqual(scores["citation_recall"], 0.5)
        self.assertEqual(scores["citation_f1"], 0.5)

    def test_empty_reference_semantics(self) -> None:
        self.assertEqual(
            citation_metrics([], []),
            {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0},
        )
        self.assertEqual(
            citation_metrics(["extra"], []),
            {"citation_precision": 0.0, "citation_recall": 1.0, "citation_f1": 0.0},
        )


class BehaviorMetricsTests(unittest.TestCase):
    def test_answerability_and_refusal_confusion_rates(self) -> None:
        scores = answerable_refusal_metrics(
            [True, True, False, False],
            [False, True, True, False],
        )
        self.assertEqual(scores["answerable_refusal_accuracy"], 0.5)
        self.assertEqual(scores["answer_rate_when_answerable"], 0.5)
        self.assertEqual(scores["correct_refusal_rate"], 0.5)
        self.assertEqual(scores["over_refusal_rate"], 0.5)
        self.assertEqual(scores["answer_when_unanswerable_rate"], 0.5)
        self.assertEqual(scores["refusal_f1"], 0.5)

    def test_refusal_inference_prefers_explicit_flag(self) -> None:
        self.assertTrue(infer_refusal("Insufficient information to answer."))
        self.assertTrue(infer_refusal("信息不足，无法确定。"))
        self.assertFalse(infer_refusal("The answer is 42."))
        self.assertFalse(infer_refusal({"refused": False, "answer": "cannot answer"}))
        self.assertFalse(
            infer_refusal(
                "已核实的主要结果如下：" + "有证据的结果与引用。" * 20
                + "另一个非必要拆分项根据当前知识库无法确定。"
            )
        )
        self.assertTrue(
            infer_refusal(
                "逐项核对如下：" + "现有证据仅涉及交易意向和业务背景。" * 12
                + "综上，三个核心字段均无法确定。"
            )
        )

    def test_tool_routing_supports_objects_and_multiple_cases(self) -> None:
        one = tool_routing_metrics(
            [{"tool_name": "search"}, {"tool_name": "write"}],
            ["search", "finish"],
        )
        self.assertEqual(one["tool_routing_precision"], 0.5)
        self.assertEqual(one["tool_routing_recall"], 0.5)
        self.assertEqual(one["tool_routing_f1"], 0.5)
        self.assertEqual(one["tool_routing_exact_match"], 0.0)

        many = tool_routing_metrics(
            [["search"], ["finish"]],
            [["search"], ["write"]],
        )
        self.assertEqual(many["tool_routing_precision"], 0.5)
        self.assertEqual(many["tool_routing_recall"], 0.5)
        self.assertEqual(many["tool_routing_exact_match"], 0.5)


class StatisticsTests(unittest.TestCase):
    def test_mean_difference_direction(self) -> None:
        self.assertEqual(mean_difference([2.0, 4.0], [1.0, 2.0]), 1.5)

    def test_paired_bootstrap_is_deterministic_and_local(self) -> None:
        random.seed(9182)
        first_global = random.random()
        first = paired_bootstrap(
            [2.0, 4.0, 8.0],
            [1.0, 2.0, 5.0],
            n_resamples=500,
            seed=17,
        )
        after_call = random.random()

        random.seed(9182)
        self.assertEqual(first_global, random.random())
        self.assertEqual(after_call, random.random())
        self.assertEqual(
            first,
            paired_bootstrap(
                [2.0, 4.0, 8.0],
                [1.0, 2.0, 5.0],
                n_resamples=500,
                seed=17,
            ),
        )
        self.assertEqual(first["mean_difference"], 2.0)
        self.assertLessEqual(first["ci_low"], 2.0)
        self.assertGreaterEqual(first["ci_high"], 2.0)

    def test_paired_statistics_validate_inputs(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap([], [])
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0], confidence=1.0)


if __name__ == "__main__":
    unittest.main()

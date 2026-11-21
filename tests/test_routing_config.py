"""Tests for the rule-based router and the configuration object.

Both are deliberately free of torch/transformers: `routing.py` was extracted out
of `gate.py` for exactly this reason, so the routing decision can be tested
without a multi-gigabyte install.

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dualmodel.routing import classify_by_rules  # noqa: E402
from dualmodel.config import DualModelConfig, MODEL_SIZE_MAP  # noqa: E402

LOCAL = ["你好", "嗨", "hello", "谢谢", "好的"]
CLOUD = ["代码", "算法", "python", "计算", "方程"]


def route(q, local=LOCAL, cloud=CLOUD):
    return classify_by_rules(q, local, cloud)


class RoutingBasicsTests(unittest.TestCase):
    def test_empty_input_defaults_to_local(self):
        r = route("   ")
        self.assertEqual(r["model"], "local")
        self.assertEqual(r["confidence"], 0.5)

    def test_a_greeting_goes_local(self):
        r = route("你好")
        self.assertEqual(r["model"], "local")
        self.assertGreaterEqual(r["confidence"], 0.65)

    def test_more_local_keywords_raise_confidence_but_cap_it(self):
        lo = route("你好")["confidence"]
        hi = route("你好，谢谢，好的，嗨")["confidence"]
        self.assertGreater(hi, lo)
        self.assertLessEqual(hi, 0.90)

    def test_the_reason_names_what_matched(self):
        self.assertIn("你好", route("你好")["reason"])

    def test_every_decision_carries_a_reason(self):
        for q in ["你好", "def f():", "1+1", "帮我解释一下这个复杂的算法问题好吗"]:
            self.assertTrue(route(q)["reason"])


class RoutingTechnicalTests(unittest.TestCase):
    def test_code_patterns_route_to_cloud(self):
        for q in ["def foo(x):", "class Bar:", "import os", "function f()", "const x = 1", "let y = 2"]:
            self.assertEqual(route(q)["model"], "cloud", q)

    def test_arithmetic_routes_to_cloud(self):
        self.assertEqual(route("帮我算 12 * 34")["model"], "cloud")

    def test_code_is_matched_case_insensitively(self):
        # The patterns are lowercase; matching the raw text meant IMPORT/DEF
        # slipped past them.
        for q in ["DEF foo(x):", "IMPORT os", "CLASS Bar"]:
            self.assertEqual(route(q)["model"], "cloud", q)

    def test_a_greeting_does_not_hijack_a_code_question(self):
        # Regression: the greeting was checked first, so this went to the small
        # local model even though it is a code question.
        r = route("你好，def foo() 这段怎么改")
        self.assertEqual(r["model"], "cloud")
        self.assertIn("代码", r["reason"])

    def test_a_greeting_still_wins_when_nothing_technical_is_present(self):
        self.assertEqual(route("你好，今天天气不错呀")["model"], "local")


class RoutingKeywordBoundaryTests(unittest.TestCase):
    def test_an_ascii_greeting_matches_as_a_whole_word(self):
        self.assertEqual(route("hi", ["hi"], [])["model"], "local")

    def test_an_ascii_greeting_does_not_match_inside_another_word(self):
        # "hi" 曾命中 "machine" 里的 hi，把技术问题误判给本地模型。
        r = classify_by_rules(
            "请解释 machine learning 的算法原理，并比较它和深度学习的区别",
            ["hi", "hello"],
            ["算法", "原理", "比较", "区别", "深度学习"],
        )
        self.assertEqual(r["model"], "cloud", r)

    def test_chinese_keywords_still_match_as_substrings(self):
        self.assertEqual(route("谢谢你的帮助")["model"], "local")


class RoutingComplexityTests(unittest.TestCase):
    def test_long_text_goes_to_cloud(self):
        self.assertEqual(route("这个" * 70)["model"], "cloud")

    def test_a_medium_sentence_with_a_cloud_keyword_goes_to_cloud(self):
        # >50 chars (+0.20) and a cloud keyword (+0.12) clears the 0.35 bar.
        q = "帮我看看这段" + "。" * 45 + "算法该怎么写"
        self.assertEqual(route(q)["model"], "cloud")

    def test_a_very_short_question_stays_local(self):
        self.assertEqual(route("嗯嗯")["model"], "local")

    def test_explanation_phrasing_adds_complexity(self):
        # Local-model confidence is *inverse* to complexity, so a more complex
        # question gets a lower local confidence, not a higher one.
        plain = route("这个东西")
        explained = route("这是什么东西，请解释")
        self.assertEqual(explained["model"], "local")
        self.assertLess(explained["confidence"], plain["confidence"])
        self.assertIn("疑问", explained["reason"])

    def test_explanation_plus_request_form_crosses_into_cloud(self):
        # 请.*解释 (+0.15) + 请 (+0.10) + 算法 (+0.12) = 0.37 >= 0.35
        r = route("请解释一下这个算法")
        self.assertEqual(r["model"], "cloud")
        self.assertIn("疑问", r["reason"])

    def test_cloud_confidence_never_exceeds_its_cap(self):
        r = route("请帮我解释这个算法为什么这么复杂，如何优化，怎么计算，方程" * 3)
        self.assertLessEqual(r["confidence"], 0.95)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_usable(self):
        c = DualModelConfig()
        self.assertEqual(c.local_model_size, "7B")
        self.assertEqual(c.local_model_name, MODEL_SIZE_MAP["7B"])
        self.assertEqual(c.cloud_model_name, "deepseek-chat")

    def test_size_selects_the_model(self):
        for size, name in MODEL_SIZE_MAP.items():
            self.assertEqual(DualModelConfig(local_model_size=size).local_model_name, name, size)

    def test_api_size_does_not_rewrite_the_model_name(self):
        c = DualModelConfig(local_model_size="api", local_model_name="custom/model")
        self.assertEqual(c.local_model_name, "custom/model")

    def test_api_key_falls_back_to_the_environment(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-from-env"}, clear=False):
            self.assertEqual(DualModelConfig().cloud_api_key, "sk-from-env")

    def test_an_explicit_key_is_not_overwritten_by_the_environment(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-from-env"}, clear=False):
            self.assertEqual(DualModelConfig(cloud_api_key="sk-explicit").cloud_api_key, "sk-explicit")

    def test_repr_hides_the_key(self):
        self.assertNotIn("sk-secret", repr(DualModelConfig(cloud_api_key="sk-secret")))


class ConfigFromEnvTests(unittest.TestCase):
    def test_integers_floats_and_booleans_are_parsed(self):
        env = {
            "LOCAL_MAX_LENGTH": "512",
            "CLOUD_TEMPERATURE": "0.25",
            "GATE_USE_EMBEDDING": "true",
            "API_PORT": "8080",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            c = DualModelConfig.from_env()
        self.assertEqual(c.local_max_length, 512)
        self.assertAlmostEqual(c.cloud_temperature, 0.25)
        self.assertTrue(c.gate_use_embedding)
        self.assertEqual(c.api_port, 8080)

    def test_false_and_zero_are_parsed_as_false(self):
        with mock.patch.dict(os.environ, {"GATE_USE_EMBEDDING": "0"}, clear=False):
            self.assertFalse(DualModelConfig.from_env().gate_use_embedding)

    def test_unparseable_values_leave_the_default_in_place(self):
        default = DualModelConfig().local_max_length
        with mock.patch.dict(os.environ, {"LOCAL_MAX_LENGTH": "not-a-number"}, clear=False):
            with self.assertLogs(level="WARNING"):
                c = DualModelConfig.from_env()
        self.assertEqual(c.local_max_length, default)

    def test_size_from_env_also_remaps_the_model_name(self):
        with mock.patch.dict(os.environ, {"LOCAL_MODEL_SIZE": "1.5B"}, clear=False):
            self.assertEqual(DualModelConfig.from_env().local_model_name, MODEL_SIZE_MAP["1.5B"])

    def test_a_model_name_from_env_is_not_overwritten_by_the_default_size(self):
        # 默认规格是 7B，之前 from_env 会把 LOCAL_MODEL_NAME 重新映射成 7B 的模型名。
        env = {"LOCAL_MODEL_NAME": "custom/model"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(DualModelConfig.from_env().local_model_name, "custom/model")


if __name__ == "__main__":
    unittest.main()

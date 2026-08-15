"""Unit tests for the A/B evaluator and local server manager."""

import pytest

from oneclick_distill.eval import Backend, exact_match, normalize, rouge_l_f1, tokenize

pytestmark = pytest.mark.smoke


def test_tokenize_cjk_chars_and_latin_words():
    assert tokenize("LoRA 和全量微调") == ["lora", "和", "全", "量", "微", "调"]
    assert tokenize("Q4_K_M") == ["q4", "k", "m"]


def test_rouge_l_identical_is_1():
    a = "知识蒸馏是一种将知识从一个模型转移到另一个更小模型的技术。"
    assert rouge_l_f1(a, a) == 1.0


def test_rouge_l_different_models_is_low():
    a = "知识蒸馏是一种将知识从一个模型转移到另一个更小模型的技术。"
    b = "今天天气很好，适合出去散步。"
    assert rouge_l_f1(a, b) < 0.3


def test_exact_match():
    assert exact_match("Hello, World!", "hello world")
    assert not exact_match("苹果", "香蕉")


def test_backend_parse():
    s = Backend.parse("transformers:runs/xxx/model")
    assert (s.kind, s.model) == ("transformers", "runs/xxx/model")
    o = Backend.parse("openai:http://127.0.0.1:8123#qwen-0.5b:sk-test")
    assert o.base_url == "http://127.0.0.1:8123"
    assert o.model == "qwen-0.5b"
    assert o.api_key == "sk-test"
    with pytest.raises(ValueError):
        Backend.parse("bogus")

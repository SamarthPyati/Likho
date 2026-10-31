import pytest

from likho.mocks import MockLLM
from likho.router import STAGES, Intent, route, rule_intent
from likho.schemas import Stage


class FixedLLM(MockLLM):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.reply


@pytest.mark.parametrize(
    "text, intent",
    [
        ("kal ke meeting ke liye Rohit ko bata dena", Intent.ROMAN_HINGLISH),
        ("मुझे कल जाना है", Intent.DEVANAGARI),
        ("मुझे kal जाना है", Intent.MIXED_SCRIPT),
        ("please call me when you reach", Intent.ENGLISH),
        ("  42 !! ", Intent.EMPTY),
        ("", Intent.EMPTY),
    ],
)
def test_rule_intent(text, intent):
    assert rule_intent(text) is intent


def test_devanagari_and_english_skip_transliteration():
    for intent in (Intent.DEVANAGARI, Intent.ENGLISH):
        assert Stage.TRANSLITERATE not in STAGES[intent]
        assert Stage.REWRITE in STAGES[intent]


def test_valid_llm_answer_is_used():
    llm = FixedLLM('{"intent": "english", "reason": "all English"}')
    r = route("please call me", llm=llm)
    assert r.source == "llm"
    assert r.stages == STAGES[Intent.ENGLISH]
    assert r.reason


def test_llm_answer_with_preamble_is_parsed():
    llm = FixedLLM('Sure! The answer is:\n```json\n{"intent": "Roman_Hinglish"}\n```')
    assert route("kal milte hain", llm=llm).source == "llm"


@pytest.mark.parametrize(
    "reply",
    [
        "I think this is Hinglish.",  # no JSON
        '{"intent": "hinglish"}',  # unknown intent
        '{"intent": ',  # truncated
        '["roman_hinglish"]',  # not an object
        "",
    ],
)
def test_malformed_llm_reply_falls_back(reply):
    r = route("kal milte hain", llm=FixedLLM(reply))
    assert r.source == "rule_fallback"
    assert r.stages == STAGES[Intent.ROMAN_HINGLISH]


def test_llm_answer_contradicting_script_falls_back():
    # Claims Devanagari for Roman text: impossible, so the rules decide.
    r = route("kal milte hain", llm=FixedLLM('{"intent": "devanagari"}'))
    assert r.source == "rule_fallback"
    assert Stage.TRANSLITERATE in r.stages


def test_llm_exception_falls_back():
    class Broken(MockLLM):
        def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("boom")

    assert route("मुझे कल जाना है", llm=Broken()).source == "rule_fallback"


def test_use_llm_false_never_calls_model():
    llm = FixedLLM('{"intent": "english"}')
    assert route("kal milte hain", llm=llm, use_llm=False).source == "rule_fallback"
    assert llm.calls == 0

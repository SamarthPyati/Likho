import pytest

from likho.errors import StageFailure
from likho.mocks import MockLLM
from likho.rewrite import REGISTER_GUIDE, clean_reply, rewrite, validate
from likho.schemas import Register, Stage

TEXT = "कल के meeting के लिए Rohit को बता देना"


class FixedLLM(MockLLM):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_mock_rewrite_is_identity_and_valid():
    assert rewrite(TEXT, Register.FORMAL, llm=MockLLM()) == TEXT


def test_prompt_carries_register_and_text():
    llm = FixedLLM(TEXT)
    rewrite(TEXT, Register.FORMAL, llm=llm)
    assert REGISTER_GUIDE[Register.FORMAL] in llm.prompts[0]
    assert f"<text>{TEXT}</text>" in llm.prompts[0]


@pytest.mark.parametrize(
    "reply",
    [
        f"Here is the rewritten text:\n{TEXT}",
        f"```\n{TEXT}\n```",
        f'"{TEXT}"',
        f"<text>{TEXT}</text>",
        f"Rewritten: {TEXT}",
    ],
)
def test_preamble_and_wrapping_are_stripped(reply):
    assert rewrite(TEXT, Register.NEUTRAL, llm=FixedLLM(reply)) == TEXT


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("", "empty"),
        ("ठीक है", "length ratio"),  # far too short: a dropped clause
        (TEXT + " और फिर हम सब शाम को पार्टी करेंगे और खाना खाएंगे", "length ratio"),
        ("Tell Rohit about the meeting tomorrow, please do", "script changed"),
    ],
)
def test_bad_rewrites_raise_stage_failure(reply, reason):
    with pytest.raises(StageFailure) as info:
        rewrite(TEXT, Register.FORMAL, llm=FixedLLM(reply))
    assert info.value.error.stage is Stage.REWRITE
    assert reason in info.value.error.message


def test_model_exception_becomes_stage_failure():
    class Broken(MockLLM):
        def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("MPS backend out of memory")

    with pytest.raises(StageFailure) as info:
        rewrite(TEXT, Register.CASUAL, llm=Broken())
    assert "RuntimeError" in info.value.error.message


def test_empty_input_skips_model():
    llm = FixedLLM("should not be used")
    assert rewrite("   ", Register.FORMAL, llm=llm) == "   "
    assert llm.prompts == []


def test_validate_bounds_are_inclusive():
    assert validate("a" * 10, "b" * 5) is None
    assert validate("a" * 10, "b" * 16) is None
    assert validate("a" * 10, "b" * 4) is not None


def test_clean_reply_keeps_inner_quotes():
    assert clean_reply('उसने कहा "हाँ"') == 'उसने कहा "हाँ"'


def test_english_rewritten_into_hindi_is_rejected():
    # The prompt is in Hindi; for all-English input Param-1 may answer in Hindi.
    english = "please send me the report by tomorrow"
    with pytest.raises(StageFailure) as info:
        rewrite(english, Register.FORMAL, llm=FixedLLM("कृपया कल तक मुझे report भेज दीजिए"))
    assert "script changed from Roman" in info.value.error.message

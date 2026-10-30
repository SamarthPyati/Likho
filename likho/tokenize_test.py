import pytest

from likho.mocks import MockLLM, MockXlitEngine
from likho.schemas import Token, TokenKind
from likho.tokenize import CONFIDENCE, SIGNAL_BY_CONFIDENCE, render, split, tag

H, E, NE = TokenKind.HINDI, TokenKind.ENGLISH, TokenKind.NAMED_ENTITY


class RecordingLLM(MockLLM):
    def __init__(self, reply: str | None = None) -> None:
        self.prompts: list[str] = []
        self.reply = reply

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.reply if self.reply is not None else super().generate(prompt, **kwargs)


def kinds(text: str, **kwargs) -> list[TokenKind]:
    return [t.kind for t in tag(text, engine=MockXlitEngine(), **kwargs)]


def signals(text: str, **kwargs) -> list[str]:
    tokens = tag(text, engine=MockXlitEngine(), **kwargs)
    return [SIGNAL_BY_CONFIDENCE[t.kind_confidence] for t in tokens]


# ---- split / render


def test_split_keeps_punctuation_numbers_and_devanagari():
    assert split("kal 10:30 baje, Rohit's ghar!") == [
        "kal",
        "10:30",
        "baje",
        ",",
        "Rohit's",
        "ghar",
        "!",
    ]
    assert split("आज गर्मी है। ok २०२६") == ["आज", "गर्मी", "है", "।", "ok", "२०२६"]


def test_split_empty():
    assert split("") == [] and split("   ") == []


def test_render_preserves_original_spacing():
    text = "kal  ke liye,\tRohit!"
    tokens = [Token(surface=s, kind=H, chosen=s.upper()) for s in split(text)]
    assert render(text, tokens) == "KAL  KE LIYE,\tROHIT!"


def test_render_rejects_mismatched_tokens():
    with pytest.raises(ValueError):
        render("kal ke", [Token(surface="kal", kind=H)])


# ---- non-Latin kinds


def test_fixed_kinds():
    assert kinds("मुझे 5 , है") == [
        TokenKind.DEVANAGARI,
        TokenKind.NUMERIC,
        TokenKind.PUNCT,
        TokenKind.DEVANAGARI,
    ]
    assert signals("5") == ["fixed"]


# ---- orthography


def test_mid_sentence_capital_is_named_entity():
    assert kinds("kal Rohit ko bata dena")[1] is NE


def test_sentence_initial_capital_is_not_assumed_a_name():
    assert kinds("Kal milte hain")[0] is H


def test_capital_after_sentence_end_is_sentence_initial():
    assert kinds("theek hai. Kal milte hain")[3] is H


def test_all_caps_is_english_acronym():
    assert kinds("mujhe PDF bhej do")[1] is E


# ---- lexicon signals


def test_hindi_only_and_english_only_words_settled_by_dictionary():
    tokens = tag("mujhe when", engine=MockXlitEngine())
    assert [t.kind for t in tokens] == [H, E]
    assert tokens[0].kind_confidence == CONFIDENCE["dictionary"]


def test_meeting_stays_english_and_kal_becomes_hindi():
    assert kinds("kal ke meeting ke liye") == [H, H, E, H, H]


# ---- homographs: context, LLM, fallback


def test_homograph_follows_hindi_context():
    out = tag("mujhe to pata hi nahi tha", engine=MockXlitEngine(), use_llm=False)
    assert out[1].kind is H
    assert SIGNAL_BY_CONFIDENCE[out[1].kind_confidence] == "context"


def test_homograph_follows_english_context():
    out = tag("please call me when you reach", engine=MockXlitEngine(), use_llm=False)
    assert out[2].kind is E
    assert SIGNAL_BY_CONFIDENCE[out[2].kind_confidence] == "context"


def test_llm_called_once_per_sentence_with_all_pending_tokens():
    llm = RecordingLLM(reply='{"0:to": "english", "1:me": "english"}')
    out = tag("to me", engine=MockXlitEngine(), llm=llm, use_llm=True)
    assert len(llm.prompts) == 1
    assert "<text>0:to 1:me</text>" in llm.prompts[0]
    assert [t.kind for t in out] == [E, E]
    assert all(SIGNAL_BY_CONFIDENCE[t.kind_confidence] == "llm" for t in out)


def test_llm_not_called_when_context_decides():
    llm = RecordingLLM()
    tag("mujhe to pata hi nahi tha", engine=MockXlitEngine(), llm=llm, use_llm=True)
    assert llm.prompts == []


@pytest.mark.parametrize(
    "reply",
    ["sure! here you go", '{"0:to": "maybe"}', "{not json}", '["hindi"]', ""],
)
def test_malformed_llm_reply_falls_back_without_crashing(reply):
    out = tag("to", engine=MockXlitEngine(), llm=RecordingLLM(reply=reply), use_llm=True)
    assert out[0].kind is H
    assert SIGNAL_BY_CONFIDENCE[out[0].kind_confidence] == "fallback"


def test_llm_exception_falls_back():
    class Broken(MockLLM):
        def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("CUDA out of memory")

    out = tag("to", engine=MockXlitEngine(), llm=Broken(), use_llm=True)
    assert SIGNAL_BY_CONFIDENCE[out[0].kind_confidence] == "fallback"


def test_llm_disabled_skips_call():
    llm = RecordingLLM()
    tag("to", engine=MockXlitEngine(), llm=llm, use_llm=False)
    assert llm.prompts == []


def test_confidences_are_distinct_per_signal():
    assert len(set(CONFIDENCE.values())) == len(CONFIDENCE)

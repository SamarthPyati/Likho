import pytest

from likho.errors import StageFailure
from likho.mocks import MockTranslator
from likho.schemas import Stage
from likho.translate import translate, translate_batch


class RecordingTranslator(MockTranslator):
    def __init__(self, reply: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], str, str]] = []
        self.reply = reply

    def translate_batch(self, texts, src_lang="", tgt_lang=""):
        self.calls.append((texts, src_lang, tgt_lang))
        return self.reply if self.reply is not None else super().translate_batch(texts)


def test_uses_flores_codes():
    tr = RecordingTranslator()
    assert translate("मुझे कल जाना है", translator=tr) == "[EN] मुझे कल जाना है"
    assert tr.calls[0][1:] == ("hin_Deva", "eng_Latn")


def test_batch_is_one_call_and_skips_blanks():
    tr = RecordingTranslator()
    assert translate_batch(["एक", "  ", "दो"], translator=tr) == ["[EN] एक", "", "[EN] दो"]
    assert tr.calls == [(["एक", "दो"], "hin_Deva", "eng_Latn")]


def test_blank_input_skips_model():
    tr = RecordingTranslator()
    assert translate("", translator=tr) == ""
    assert tr.calls == []


@pytest.mark.parametrize("reply, reason", [([""], "empty"), ([], "0 translations")])
def test_bad_output_raises(reply, reason):
    with pytest.raises(StageFailure) as info:
        translate("कल", translator=RecordingTranslator(reply))
    assert info.value.error.stage is Stage.TRANSLATE
    assert reason in info.value.error.message


def test_model_exception_becomes_stage_failure():
    class Broken(MockTranslator):
        def translate_batch(self, texts, src_lang="", tgt_lang=""):
            raise OSError("401 gated repo")

    with pytest.raises(StageFailure) as info:
        translate("कल", translator=Broken())
    assert "OSError" in info.value.error.message

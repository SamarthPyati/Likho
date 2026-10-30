import json

from likho.mocks import MockLLM, MockTranslator, MockXlitEngine, naive_transliterate


def test_xlit_known_word_ranked_candidates():
    engine = MockXlitEngine()
    assert engine.translit_word("kal", topk=5) == {"hi": ["कल", "काल"]}


def test_xlit_respects_topk():
    assert MockXlitEngine().translit_word("hai", topk=1) == {"hi": ["है"]}


def test_xlit_unknown_word_is_deterministic_devanagari():
    engine = MockXlitEngine()
    first = engine.translit_word("pyaar", topk=5)
    assert first == engine.translit_word("pyaar", topk=5)
    assert first["hi"][0] == naive_transliterate("pyaar")
    assert all("ऀ" <= ch <= "ॿ" for ch in first["hi"][0])


def test_xlit_sentence():
    out = MockXlitEngine().translit_sentence("mujhe kal jana hai")
    assert out == {"hi": "मुझे कल जाना है"}


def test_naive_transliterate_examples():
    assert naive_transliterate("kitab") == "कितब"
    assert naive_transliterate("aam") == "आम"


def test_llm_route_returns_parseable_json():
    out = json.loads(MockLLM().generate("TASK: route\nInput: kal milte hain"))
    assert out["stages"][0] == "route"


def test_llm_rewrite_is_identity():
    prompt = "TASK: rewrite\nRegister: formal\n<text>कल मिलते हैं</text>"
    assert MockLLM().generate(prompt) == "कल मिलते हैं"


def test_llm_tag_labels_every_token():
    out = json.loads(MockLLM().generate("TASK: tag\n<text>to me</text>"))
    assert out == {"to": "hindi", "me": "hindi"}


def test_translator_prefix():
    assert MockTranslator().translate_batch(["कल"]) == ["[EN] कल"]

"""Deterministic stubs for every model, selected by MOCK_MODELS=1.

These mirror the call signatures of the real models so stage code does not
branch on mock vs real. See docs/MODELS.md §6.

LLM prompt convention: every prompt sent to Param-1 starts with a line
`TASK: <name>` (route, tag, rewrite, judge). The real model ignores it as
context; `MockLLM` dispatches on it. Text the rewrite stage operates on goes
between `<text>` and `</text>`.
"""

import json
import re
from collections.abc import Sequence

# Roman → ranked Devanagari candidates. Extend as test fixtures grow (target ~200).
ROMAN_TO_DEVANAGARI: dict[str, list[str]] = {
    "aaj": ["आज"],
    "aap": ["आप", "आपकी"],
    "accha": ["अच्छा", "अछा"],
    "achha": ["अच्छा", "अछा"],
    "ap": ["आप", "अप"],
    "aur": ["और", "औऱ"],
    "bahut": ["बहुत", "बहूत"],
    "bata": ["बता", "बाता"],
    "bhai": ["भाई", "भई"],
    "chahiye": ["चाहिए", "चाहिये"],
    "dena": ["देना", "दीना"],
    "dost": ["दोस्त", "दोसत"],
    "do": ["दो", "डो"],
    "ek": ["एक", "इक"],
    "gaya": ["गया", "गाया"],
    "ghar": ["घर", "गर"],
    "hai": ["है", "हाई", "हैं"],
    "hain": ["हैं", "हैन"],
    "ham": ["हम", "हैम"],
    "ho": ["हो", "हौ"],
    "hum": ["हम", "हुम"],
    "is": ["इस", "इज़"],
    "jaana": ["जाना", "जना"],
    "jai": ["जय", "जई"],
    "jana": ["जाना", "जना"],
    "kab": ["कब", "काब"],
    "kaam": ["काम", "कम"],
    "kal": ["कल", "काल"],
    "kar": ["कर", "कार"],
    "karna": ["करना", "कारना"],
    "kaise": ["कैसे", "कैसी"],
    "kya": ["क्या", "कया"],
    "ke": ["के", "की"],
    "ki": ["की", "कि"],
    "ko": ["को", "कों"],
    "kuch": ["कुछ", "कुच"],
    "liye": ["लिए", "लिये"],
    "main": ["मैं", "मेन", "मैन"],
    "me": ["में", "मे", "मी"],
    "mein": ["में", "मेइन"],
    "mera": ["मेरा", "मीरा"],
    "meri": ["मेरी", "मेरि"],
    "mujhe": ["मुझे", "मुझें"],
    "na": ["ना", "न"],
    "nahi": ["नहीं", "नही"],
    "nahin": ["नहीं", "नहिन"],
    "pe": ["पे", "पी"],
    "par": ["पर", "पार"],
    "raha": ["रहा", "राहा"],
    "rahi": ["रही", "राही"],
    "se": ["से", "सी"],
    "tha": ["था", "ठा"],
    "thi": ["थी", "ठी"],
    "to": ["तो", "टू", "तू"],
    "tum": ["तुम", "टम"],
    "tumhe": ["तुम्हें", "तुम्हे"],
    "wala": ["वाला", "वला"],
    "yaar": ["यार", "यार्"],
    "yeh": ["यह", "ये"],
    "ye": ["ये", "यह"],
    # loanwords, for when the user opts in (U8)
    "bank": ["बैंक", "बंक"],
    "meeting": ["मीटिंग", "मीटिंग्स"],
    "office": ["ऑफिस", "ऑफ़िस", "ओफिस"],
    "train": ["ट्रेन", "ट्रैन"],
    "scooter": ["स्कूटर", "स्कुटर"],
}

_VOWELS = [  # (roman, independent, matra) — longest first
    ("aa", "आ", "ा"),
    ("ai", "ऐ", "ै"),
    ("au", "औ", "ौ"),
    ("ee", "ई", "ी"),
    ("oo", "ऊ", "ू"),
    ("a", "अ", ""),
    ("i", "इ", "ि"),
    ("u", "उ", "ु"),
    ("e", "ए", "े"),
    ("o", "ओ", "ो"),
]
_CONSONANTS = [  # longest first
    ("chh", "छ"),
    ("kh", "ख"),
    ("gh", "घ"),
    ("ch", "च"),
    ("jh", "झ"),
    ("th", "थ"),
    ("dh", "ध"),
    ("ph", "फ"),
    ("bh", "भ"),
    ("sh", "श"),
    ("k", "क"),
    ("g", "ग"),
    ("j", "ज"),
    ("t", "त"),
    ("d", "द"),
    ("n", "न"),
    ("p", "प"),
    ("f", "फ"),
    ("b", "ब"),
    ("m", "म"),
    ("y", "य"),
    ("r", "र"),
    ("l", "ल"),
    ("v", "व"),
    ("w", "व"),
    ("s", "स"),
    ("h", "ह"),
    ("z", "ज़"),
    ("q", "क"),
    ("x", "क्स"),
    ("c", "क"),
]


def _match(word: str, i: int, table: list[tuple]) -> tuple | None:
    for entry in table:
        if word.startswith(entry[0], i):
            return entry
    return None


def naive_transliterate(word: str) -> str:
    """Greedy letter-by-letter romanization → Devanagari. Deterministic, crude."""
    word = word.lower()
    out: list[str] = []
    i = 0
    after_consonant = False
    while i < len(word):
        if cons := _match(word, i, _CONSONANTS):
            out.append(cons[1])
            i += len(cons[0])
            after_consonant = True
        elif vowel := _match(word, i, _VOWELS):
            out.append(vowel[2] if after_consonant else vowel[1])
            i += len(vowel[0])
            after_consonant = False
        else:
            out.append(word[i])
            i += 1
            after_consonant = False
    return "".join(out)


class MockXlitEngine:
    """Stands in for `ai4bharat.transliteration.XlitEngine("hi", ...)`."""

    def __init__(self, lang: str = "hi") -> None:
        self.lang = lang

    def translit_word(self, word: str, topk: int = 5) -> dict[str, list[str]]:
        key = word.lower()
        if key in ROMAN_TO_DEVANAGARI:
            candidates = ROMAN_TO_DEVANAGARI[key]
        else:
            base = naive_transliterate(key)
            variants = [base, base.replace("ि", "ी"), base.replace("ु", "ू")]
            candidates = list(dict.fromkeys(variants))
        return {self.lang: candidates[:topk]}

    def translit_sentence(self, text: str) -> dict[str, str]:
        words = [self.translit_word(w, topk=1)[self.lang][0] for w in text.split()]
        return {self.lang: " ".join(words)}


_TASK_RE = re.compile(r"^TASK:\s*(\w+)", re.MULTILINE)
_TEXT_RE = re.compile(r"<text>([\s\S]*?)</text>")

_ROUTE_RESPONSE = {"intent": "roman_hinglish", "reason": "mock: fixed intent"}
_JUDGE_RESPONSE = {"meaning": 4, "register": 4, "fluency": 4}


class MockLLM:
    """Stands in for Param-1. Fixed JSON for route/judge, identity rewrite."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        examples: Sequence[tuple[str, str]] = (),
        prefix: str = "",
    ) -> str:
        reply = self._reply(prompt)
        return reply[len(prefix) :] if prefix and reply.startswith(prefix) else reply

    def _reply(self, prompt: str) -> str:
        task_match = _TASK_RE.search(prompt)
        task = task_match.group(1) if task_match else ""
        if task == "route":
            return json.dumps(_ROUTE_RESPONSE)
        if task == "judge":
            return json.dumps(_JUDGE_RESPONSE)
        if task == "tag":
            # Tiebreak: every ambiguous token resolved as Hindi.
            text_match = _TEXT_RE.search(prompt)
            tokens = text_match.group(1).split() if text_match else []
            return json.dumps({t: "hindi" for t in tokens})
        if task == "rewrite":
            text_match = _TEXT_RE.search(prompt)
            return text_match.group(1).strip() if text_match else ""
        return ""


class MockTranslator:
    """Stands in for IndicTrans2 (preprocess → generate → postprocess)."""

    def translate_batch(
        self, texts: list[str], src_lang: str = "hin_Deva", tgt_lang: str = "eng_Latn"
    ) -> list[str]:
        return ["[EN] " + t for t in texts]

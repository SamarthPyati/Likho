"""Cross-module data contracts. See docs/ARCHITECTURE.md §3.

Every structure that crosses a module boundary is defined here. Change these
only by agreement — every stage depends on them.
"""

import warnings
from enum import Enum

from pydantic import BaseModel, Field

# `register` is the API's field name for the tone (PRD U3). Pydantic warns that
# it shadows the `register` method BaseModel inherits from ABCMeta; nothing here
# calls that method, so the field is safe and renaming it would break clients.
warnings.filterwarnings("ignore", message='Field name "register"', category=UserWarning)


class TokenKind(str, Enum):
    HINDI = "hindi"  # Roman-script Hindi — transliterate
    ENGLISH = "english"  # English loanword — keep by default
    NAMED_ENTITY = "named_entity"  # person, place, brand — keep
    NUMERIC = "numeric"
    PUNCT = "punct"
    DEVANAGARI = "devanagari"  # already in native script


class Register(str, Enum):
    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"


class Stage(str, Enum):
    ROUTE = "route"
    TOKENIZE = "tokenize"
    TRANSLITERATE = "transliterate"
    RERANK = "rerank"
    REWRITE = "rewrite"
    TRANSLATE = "translate"


class Token(BaseModel):
    surface: str  # as the user typed it
    kind: TokenKind
    kind_confidence: float = 1.0
    candidates: list[str] = Field(default_factory=list)  # top-k, best first
    chosen: str | None = None  # after reranking
    chosen_rank: int | None = None  # which candidate won; 0 = model top-1


class Route(BaseModel):
    stages: list[Stage]
    reason: str  # shown in the UI (U6)
    source: str  # "llm" or "rule_fallback"


class StageError(BaseModel):
    stage: Stage
    message: str
    recoverable: bool = True


class AssistRequest(BaseModel):
    text: str
    register: Register = Register.NEUTRAL
    translate: bool = False
    transliterate_loanwords: bool = False  # U8
    top_k: int = 5


class AssistResponse(BaseModel):
    route: Route
    tokens: list[Token]
    native_script: str | None = None
    rewritten: str | None = None
    english: str | None = None
    errors: list[StageError] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

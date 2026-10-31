"""Loader for the hand-built YAML case files (codemixed_cases.yaml, dev_cases.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from likho.schemas import TokenKind

DATA = Path(__file__).parent / "data"
SPLITS = {"test": DATA / "codemixed_cases.yaml", "dev": DATA / "dev_cases.yaml"}

KIND_CODES = {
    "h": TokenKind.HINDI,
    "e": TokenKind.ENGLISH,
    "ne": TokenKind.NAMED_ENTITY,
    "n": TokenKind.NUMERIC,
    "p": TokenKind.PUNCT,
    "d": TokenKind.DEVANAGARI,
}


class ExpectedToken(BaseModel):
    surface: str
    kind: TokenKind


class Case(BaseModel):
    id: str
    input: str
    expected_tokens: list[ExpectedToken]
    expected_native: str
    expected_english: str
    difficulty: str
    tags: list[str]

    @property
    def category(self) -> str:
        """Primary tag: the one this case counts toward in the §1.2 distribution."""
        return self.tags[0]


def parse_compact_tokens(compact: str) -> list[ExpectedToken]:
    """`"kal/h Rohit/ne ,/p"` → [{surface: kal, kind: hindi}, ...]."""
    tokens = []
    for item in compact.split():
        surface, _, code = item.rpartition("/")
        if not surface or code not in KIND_CODES:
            raise ValueError(f"bad token annotation {item!r}")
        tokens.append(ExpectedToken(surface=surface, kind=KIND_CODES[code]))
    return tokens


def load_cases(split: str = "test") -> list[Case]:
    raw = yaml.safe_load(SPLITS[split].read_text(encoding="utf-8"))
    cases = []
    for entry in raw["cases"]:
        if isinstance(entry["expected_tokens"], str):
            entry = {**entry, "expected_tokens": parse_compact_tokens(entry["expected_tokens"])}
        cases.append(Case.model_validate(entry))
    return cases


class RouterCase(BaseModel):
    id: str
    input: str
    expected_intent: str


def load_router_cases() -> list[RouterCase]:
    raw = yaml.safe_load((DATA / "router_cases.yaml").read_text(encoding="utf-8"))
    return [RouterCase.model_validate(entry) for entry in raw["cases"]]

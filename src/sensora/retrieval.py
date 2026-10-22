"""A small BM25 index for retrieving knowledge passages without any model download."""
from __future__ import annotations

from collections import Counter
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been of in on at to for and or but what which who whom how why when where "
    "do does did can could should would will may might i me my we our you your it its this that these those "
    "with about as by from there their them then than so if not no yes tell explain describe give show please "
    "use using used also any some more most much many very".split()
)
_SYNONYMS = {
    "epsilon": ("permittivity", "dielectric"),
    "ε": ("permittivity", "dielectric"),
    "εr": ("permittivity", "dielectric"),
    "dielectric": ("permittivity",),
    "stiffness": ("modulus", "young"),
    "stiff": ("modulus", "young"),
    "soft": ("modulus", "young"),
    "polymer": ("matrix",),
    "particle": ("filler",),
    "particles": ("filler",),
    "ceramic": ("filler",),
    "optimizer": ("ga", "genetic", "optimization"),
    "optimization": ("ga", "genetic"),
    "recommend": ("best", "top", "rank"),
    "best": ("top", "rank"),
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    lowered = text.lower()
    words = _TOKEN.findall(lowered)
    if expand:
        extra = [alias for symbol in ("ε", "εr") if symbol in lowered for alias in _SYNONYMS[symbol]]
        words += extra
        for word in list(words):
            words += _SYNONYMS.get(word, ())
    return [_stem(word) for word in words if len(word) > 1 and word not in _STOP]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(
        self,
        documents: list[str],
        titles: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 2.0,
    ) -> None:
        self.k1, self.b, self.title_weight = k1, b, title_weight
        self.title_tokens = [set(tokenize(title)) for title in (titles or [""] * len(documents))]
        self.tokens = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average = sum(self.lengths) / max(len(self.lengths), 1) or 1.0
        self.frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query, expand=True)
        result = []
        for frequencies, length in zip(self.frequencies, self.lengths, strict=True):
            score = 0.0
            for term in set(terms):
                count = frequencies.get(term, 0)
                if count:
                    norm = count * (self.k1 + 1) / (
                        count + self.k1 * (1 - self.b + self.b * length / self.average)
                    )
                    score += self.idf[term] * norm
            for term in set(terms) & self.title_tokens[len(result)]:
                score += self.title_weight * self.idf.get(term, 0.0)
            result.append(score)
        return result

    def top(self, query: str, k: int = 6) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:k] if score > 0]

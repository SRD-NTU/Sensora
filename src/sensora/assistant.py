"""Retrieval-augmented assistant: retrieve passages, then let a small local model explain them."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from collections.abc import Iterator

from .config import ROOT
from .knowledge import Chunk, load_knowledge, material_chunks, session_chunks
from .database import MaterialDatabase
from .retrieval import BM25

MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    repo: str
    file: str
    size_gb: float
    note: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.file


MODELS = (
    ModelSpec("Qwen3 4B Instruct", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
              "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2.5, "Better answers, slower"),
    ModelSpec("Qwen2.5 1.5B Instruct", "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
              "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1.1, "Faster, weaker reasoning"),
)
TOP_K = 5
PROMPT_CHARS = 650
MIN_SCORE = 3.0
NOT_FOUND = "I don't have that in the Sensora knowledge base."

SYSTEM_PROMPT = (
    "You are the Sensora assistant. Sensora designs the dielectric layer of capacitive pressure sensors "
    "(a soft polymer matrix plus ceramic fillers) and ranks designs with a physics model and a genetic algorithm.\n"
    "Rules:\n"
    "1. Answer using only the numbered context passages. Cite the passages you used, like [1] or [2][3].\n"
    f"2. If the passages do not contain the answer, reply exactly: \"{NOT_FOUND}\" Do not guess.\n"
    "3. Never invent numbers, materials, papers, or results. Copy numbers exactly as written in the passages.\n"
    "4. Model results are predictions, not measurements; say so when you describe them.\n"
    "5. Start with the direct answer (Yes, No, or the value) and make sure it agrees with the passages. If a passage "
    "says something was not found or is not validated, do not answer Yes.\n"
    "6. Do not add claims that are not in the passages. Be concise: at most 100 words, plain language, equations in plain text."
)


@dataclass(frozen=True)
class Passage:
    number: int
    chunk: Chunk
    score: float


def installed_models() -> list[ModelSpec]:
    """Known models whose files are present, best first."""
    return [spec for spec in MODELS if spec.path.is_file()]


def find_model() -> Path | None:
    """The model to use by default: SENSORA_MODEL_PATH, else the best installed model."""
    configured = os.environ.get("SENSORA_MODEL_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    installed = installed_models()
    return installed[0].path if installed else None


def download_model(spec: ModelSpec) -> Path:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(spec.repo, spec.file, local_dir=MODEL_DIR))


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    values = []
    for token in _NUMBER.findall(text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def unsupported_numbers(answer: str, passages: list[Passage]) -> list[str]:
    """Numbers in the answer that appear in none of the retrieved passages (small integers ignored)."""
    known = _numbers(" ".join(p.chunk.text for p in passages))
    body = re.sub(r"\[\d+\]", " ", answer)
    missing = []
    for token in _NUMBER.findall(body):
        value = float(token.replace(",", "").rstrip("."))
        if value == int(value) and value < 10:
            continue
        if not any(abs(value - other) <= max(0.006 * abs(other), 1e-9) for other in known):
            missing.append(token)
    return missing


def parse_pressure_range(text: str) -> tuple[float, float] | None:
    """Find a 'a-b kPa' or 'a to b kPa' pressure range in a question."""
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kpa)?\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*kpa", text.lower()
    )
    if not match:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    return (low, high) if high > low else None


def retrieve(
    question: str,
    database: MaterialDatabase,
    result=None,
    *,
    k: int = TOP_K,
    static: list[Chunk] | None = None,
) -> list[Passage]:
    """Rank static documents, database materials, and the latest result against the question."""
    chunks = (load_knowledge() if static is None else static) + material_chunks(database)
    chunks += session_chunks(result, database)
    index = BM25([f"{chunk.title} {chunk.text}" for chunk in chunks], [chunk.title for chunk in chunks])
    hits = index.top(question, k)
    return [Passage(number, chunks[i], score) for number, (i, score) in enumerate(hits, start=1)]


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def build_messages(question: str, passages: list[Passage], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(
        f"[{p.number}] {p.chunk.title}\n{_shorten(p.chunk.text, PROMPT_CHARS)}" for p in passages
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"][:600]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


def retrieval_only_answer(
    passages: list[Passage],
    reason: str = "No language model is installed, so here are the most relevant passages:",
) -> str:
    if not passages:
        return NOT_FOUND
    lines = [reason]
    for p in passages[:3]:
        text = p.chunk.text if len(p.chunk.text) <= 500 else p.chunk.text[:500].rsplit(" ", 1)[0] + "…"
        lines.append(f"**[{p.number}] {p.chunk.title}** — {text}")
    return "\n\n".join(lines)


class LocalModel:
    """A llama.cpp model running on the CPU. One generation at a time."""

    _lock = threading.Lock()

    def __init__(self, path: Path, *, context: int = 4096) -> None:
        from llama_cpp import Llama

        self.path = Path(path)
        self.name = self.path.stem
        self.llm = Llama(
            model_path=str(self.path), n_ctx=context,
            n_threads=max(2, os.cpu_count() or 4), n_batch=512, verbose=False,
        )

    def stream(self, messages: list[dict], max_tokens: int = 300) -> Iterator[str]:
        with self._lock:
            for part in self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.1, top_p=0.9, stream=True
            ):
                delta = part["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


def answer_stream(question: str, passages: list[Passage], model, history: list[dict] | None = None) -> Iterator[str]:
    """Stream an answer; refuse without calling the model when nothing relevant was retrieved."""
    if not passages or passages[0].score < MIN_SCORE:
        yield NOT_FOUND
        return
    if model is None:
        yield retrieval_only_answer(passages)
        return
    yield from model.stream(build_messages(question, passages, history))


def is_empty_answer(text: str) -> bool:
    """True when a model produced nothing useful, for example only citation markers."""
    return not re.sub(r"[\[\]\d\s,.]", "", text)

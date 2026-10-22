"""The AI Assistant page: ask questions answered from Sensora's own knowledge."""
from __future__ import annotations

import sys

import streamlit as st

from .assistant import (
    NOT_FOUND,
    LocalModel,
    ModelSpec,
    MODELS,
    answer_stream,
    download_model,
    installed_models,
    is_empty_answer,
    parse_pressure_range,
    retrieve,
    retrieval_only_answer,
    unsupported_numbers,
)
from .composite_optimization import GeneticOptimizationSettings, optimize_composites
from .composite_physics import Targets
from .database import MaterialDatabase

SUGGESTIONS = [
    "What is relative permittivity?",
    "Which filler has the highest permittivity?",
    "Can I use three fillers together?",
    "What is the best design for 0-20 kPa?",
]


@st.cache_resource(show_spinner="Loading the language model…")
def _load(path: str):
    """Load a model once per server process. Failures raise, so they are never cached."""
    return LocalModel(path)


def get_model(path: str):
    """Return (model, None), or (None, a message that says why the model could not start)."""
    try:
        return _load(path), None
    except ImportError:
        return None, (
            f"The model runtime (llama-cpp-python) is not installed for the Python running this app "
            f"({sys.executable}). Start the app with the project's own environment, or run "
            f"`{sys.executable} -m pip install llama-cpp-python`."
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _result_for(question: str, database: MaterialDatabase):
    """Use the latest optimizer run, or run one for a pressure range named in the question."""
    latest = st.session_state.get("composite_result")
    wanted = parse_pressure_range(question)
    if wanted is None:
        return latest
    if latest is not None:
        targets = latest.report["targets"]
        if (targets["pressure_min_kpa"], targets["pressure_max_kpa"]) == wanted:
            return latest
    cached = st.session_state.setdefault("assistant_runs", {})
    if wanted not in cached:
        with st.status(f"Running the optimizer for {wanted[0]:g}–{wanted[1]:g} kPa…"):
            cached[wanted] = optimize_composites(
                database,
                [row["material_id"] for row in database.material_candidates],
                Targets(*wanted),
                settings=GeneticOptimizationSettings(population_size=20, generations=50),
            )
    return cached[wanted]


def _show_sources(sources: list[dict], warning: str | None) -> None:
    if warning:
        st.warning(warning)
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for item in sources:
                st.markdown(f"**[{item['number']}] {item['title']}** · {item['source']}")
                st.caption(item["text"])


def _model_controls() -> tuple[ModelSpec | None, object]:
    installed = installed_models()
    if not installed:
        st.info(
            "No language model is installed, so answers show the most relevant passages instead. "
            "Download one to get written explanations."
        )
        st.markdown(
            "<style>[class*='st-key-download_'] button{white-space:nowrap}</style>",
            unsafe_allow_html=True,
        )
        for column, spec in zip(st.columns(len(MODELS)), MODELS):
            if column.button(
                f"Download {spec.label} ({spec.size_gb:g} GB)", key=f"download_{spec.file}", width="stretch"
            ):
                with st.spinner(f"Downloading {spec.label}…"):
                    download_model(spec)
                st.rerun()
        return None, None
    labels = {spec.file: f"{spec.label} · {spec.note}" for spec in installed}
    chosen = st.selectbox(
        "Language Model", list(labels), format_func=labels.get, key="assistant_model"
    )
    spec = next(item for item in installed if item.file == chosen)
    model, error = get_model(str(spec.path))
    if model is None:
        st.warning(f"The language model could not be started, so answers show the relevant passages only. {error}")
    return spec, model


def assistant_page(database: MaterialDatabase) -> None:
    st.title("AI Assistant")
    st.markdown(
        "<div class='sf-tag'>Ask about capacitive pressure sensors, the materials, the model, or your "
        "optimizer results. Answers come from Sensora's own documents and data, with sources.</div>",
        unsafe_allow_html=True,
    )
    _, model = _model_controls()
    latest = st.session_state.get("composite_result")
    if latest is not None and not latest.table.empty:
        span = latest.report["targets"]
        st.caption(
            f"Using your latest optimizer run ({span['pressure_min_kpa']:g}–{span['pressure_max_kpa']:g} kPa)."
        )
    else:
        st.caption("No optimizer run yet. Name a pressure range, such as 0-20 kPa, and I will run one.")

    messages = st.session_state.setdefault("assistant_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_sources(message.get("sources", []), message.get("warning"))

    pending = None
    if not messages:
        columns = st.columns(len(SUGGESTIONS))
        for column, text in zip(columns, SUGGESTIONS):
            if column.button(text, key=f"suggest_{text}", width="stretch"):
                pending = text
    elif st.button("Clear Chat"):
        st.session_state.assistant_messages = []
        st.rerun()

    question = st.chat_input("Ask about sensors, materials or your results") or pending
    if not question:
        return
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result = _result_for(question, database)
        passages = retrieve(question, database, result)
        text = st.write_stream(answer_stream(question, passages, model, history))
        if model is not None and is_empty_answer(text):
            text = retrieval_only_answer(
                passages,
                "The model did not produce a usable answer, so here are the most relevant passages. "
                "Try the larger model or rephrase the question:",
            )
            st.markdown(text)
        warning = None
        if model is not None and text != NOT_FOUND:
            missing = unsupported_numbers(text, passages)
            if missing:
                warning = (
                    "Some figures in this answer were not found in the sources: "
                    + ", ".join(dict.fromkeys(missing)) + ". Check them before relying on them."
                )
        sources = [
            {"number": p.number, "title": p.chunk.title, "source": p.chunk.source,
             "text": p.chunk.text if len(p.chunk.text) <= 700 else p.chunk.text[:700].rsplit(" ", 1)[0] + "…"}
            for p in passages
        ]
        _show_sources(sources, warning)
    messages.append({"role": "assistant", "content": text, "sources": sources, "warning": warning})
    st.rerun()

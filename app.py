import streamlit as st
from utils import qa_pipeline

import os
import time


st.set_page_config(
    page_title="Indian Law Q&A Bot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
    st.markdown(
        """
### How this chatbot works (simple)
When you type a question, two things happen:
1) **Document search**: the app quickly searches its law PDFs to find the most relevant pages.
2) **AI answer**: an AI language model reads those pages + your question and writes a clear answer.

### How to use it
- Ask in plain language (like you’re asking a person).
- Add key details (state/city, your job type, dates, company type).
- If your question is long, break it into 2–3 smaller questions.

### Trust & sources
- Turn **Show sources** on to see which PDF/page the answer used.
- If the answer looks wrong, check the sources and ask a follow‑up like: “Show the exact clause/section for this.”
        """
    )


def _format_sources(source_documents, limit: int = 5):
    if not source_documents:
        return []

    sources = []
    for doc in source_documents[:limit]:
        md = getattr(doc, "metadata", {}) or {}
        src = md.get("source") or md.get("file_path") or "(unknown source)"
        src_name = os.path.basename(str(src))
        page = md.get("page")
        if page is not None:
            # Many loaders use 0-based page indexing; display 1-based for humans.
            try:
                page_display = int(page) + 1
            except Exception:
                page_display = page
            sources.append(f"{src_name} (page {page_display})")
        else:
            sources.append(str(src_name))

    # De-duplicate while keeping order
    seen = set()
    deduped = []
    for s in sources:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


def _reset_chat():
    st.session_state.messages = []
    st.session_state.pending_user_input = None

@st.cache_resource
def get_chain():
    return qa_pipeline()

def main():
    try:
        chain = get_chain()
    except Exception as exc:
        st.error(
            "Failed to initialize the model/QA pipeline. "
            "If you're using a gated Hugging Face model (e.g. Llama-2), you must log in and/or set an access token.\n\n"
            "Quick fixes:\n"
            "- Use a public model: set env var LLA_MODEL_ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)\n"
            "- Or authenticate: run `huggingface-cli login` or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN\n\n"
            f"Details: {exc}"
        )
        return

    # Small one-time startup animation (per browser session)
    if "did_intro" not in st.session_state:
        st.session_state.did_intro = False
    if not st.session_state.did_intro:
        intro = st.empty()
        with intro.container():
            st.caption("Starting Indian Law Q&A Bot...")
            bar = st.progress(0)
            for i in range(1, 101, 8):
                bar.progress(i)
                time.sleep(0.02)
        intro.empty()
        st.session_state.did_intro = True

    tab_chat, tab_how = st.tabs(["Chat", "How this works (tech)"])

    with tab_how:
        st.header("How this works (tech)")
        st.markdown(
            """
This app is a simple **RAG** (Retrieval‑Augmented Generation) chatbot.

**1) Ingestion (offline)**
- Your PDFs live in `dataset/`.
- `ingest.py` loads PDFs, splits them into chunks, generates embeddings for each chunk, and stores them in a FAISS index under `vectorstore/`.

**2) Retrieval (at question time)**
- When you ask a question, the app embeds your question using the same embeddings model.
- It performs a similarity search in FAISS (top‑k chunks).
- Those chunks become the `Context` for the model.

**3) Generation (LLM)**
- The model gets: your question + retrieved context + a strict prompt.
- The prompt forces India‑only answers, a step‑by‑step roadmap, and references when confident.

**4) Sources**
- The UI shows the PDF filename + page number from the retrieved chunks so you can verify grounding.

**Where to change behaviour**
- Model: environment variable `LLA_MODEL_ID`
- Answer length: `LLA_MAX_NEW_TOKENS`
- Time cap: `LLA_MAX_TIME`
- Device: `LLA_DEVICE_MAP` (`cpu` locally / set `auto` if GPU exists)
            """
        )

    with tab_chat:
        # Header
        left, right = st.columns([0.8, 0.2])
        with left:
            st.title('Indian Law Q&A Bot')
        with right:
            if st.button("Clear chat", use_container_width=True):
                _reset_chat()
                st.rerun()

        model_id = os.getenv('LLA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
        st.caption(f"Model: {model_id}")

        sample_prompts = [
            "Tell me about the Factories Act.",
            "What should I do if my employer is not paying salary on time?",
            "What is an FIR and how do I file one?",
            "What is IPC Section 420?",
        ]

        if "pending_user_input" not in st.session_state:
            st.session_state.pending_user_input = None

        if "show_sources" not in st.session_state:
            st.session_state.show_sources = True

        # Chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Render history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Quick sample prompts
        controls_left, controls_right = st.columns([0.75, 0.25])
        with controls_left:
            st.caption("Sample prompts")
        with controls_right:
            st.session_state.show_sources = st.toggle(
                "Show sources",
                value=st.session_state.show_sources,
            )
        cols = st.columns(len(sample_prompts))
        for idx, prompt in enumerate(sample_prompts):
            if cols[idx].button(prompt, use_container_width=True):
                st.session_state.pending_user_input = prompt
                st.rerun()

        # Input (always render chat_input; sample prompt acts as a fallback)
        typed_input = st.chat_input("Ask a question...")
        user_input = st.session_state.pending_user_input or typed_input
        st.session_state.pending_user_input = None

        if not user_input:
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = chain.invoke({"query": user_input})
                    bot_output = result.get("result", "")
                    sources = _format_sources(result.get("source_documents"))
                    if not bot_output.strip():
                        bot_output = "I couldn't generate an answer. Try a more specific question."
                except Exception as exc:
                    bot_output = f"Error while generating answer: {exc}"
                    sources = []
            st.markdown(bot_output)
            if st.session_state.show_sources and sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s}")

        st.session_state.messages.append({"role": "assistant", "content": bot_output, "sources": sources})

if __name__ == "__main__":
    main()
import streamlit as st
from utils import qa_pipeline

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

    # Set the title of the web application
    st.title('Indian Law Q&A Bot')

    sample_prompts = [
        "Tell me about the Factories Act.",
        "What should I do if my employer is not paying salary on time?",
        "What is an FIR and how do I file one?",
        "What is IPC Section 420?",
    ]

    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = None

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick sample prompts
    st.caption("Sample prompts")
    cols = st.columns(len(sample_prompts))
    for idx, prompt in enumerate(sample_prompts):
        if cols[idx].button(prompt, use_container_width=True):
            st.session_state.pending_user_input = prompt
            st.rerun()

    # Input (typed or from sample prompt)
    user_input = st.session_state.pending_user_input or st.chat_input("Ask a question...")
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
                if not bot_output.strip():
                    bot_output = "I couldn't generate an answer. Try a more specific question."
            except Exception as exc:
                bot_output = f"Error while generating answer: {exc}"
        st.markdown(bot_output)

    st.session_state.messages.append({"role": "assistant", "content": bot_output})

if __name__ == "__main__":
    main()
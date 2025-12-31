# Importing Dependencies
import os

import faiss
import google.generativeai as genai

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Faiss Index Path
FAISS_INDEX = "vectorstore/"


def _detect_faiss_index_dim() -> int | None:
    """Return the dimension `d` stored in vectorstore/index.faiss, if present."""
    try:
        index_path = os.path.join(FAISS_INDEX, "index.faiss")
        if not os.path.exists(index_path):
            return None
        idx = faiss.read_index(index_path)
        return int(getattr(idx, "d", 0) or 0) or None
    except Exception:
        return None


def _default_embeddings_model_for_dim(dim: int | None) -> str:
    # Known/common SentenceTransformer dims:
    # - all-MiniLM-L6-v2 => 384
    # - all-mpnet-base-v2 => 768
    if dim == 384:
        return "sentence-transformers/all-MiniLM-L6-v2"
    if dim == 768:
        return "sentence-transformers/all-mpnet-base-v2"
    # If we can't detect (e.g., first run), prefer the smaller model for Cloud stability.
    return "sentence-transformers/all-MiniLM-L6-v2"

# Custom prompt template (model-agnostic)
custom_prompt_template = """
You are an assistant focused on Indian law and Indian legal procedure.

India glossary (do not contradict):
- FIR = First Information Report (India) — NOT “Informal/International/Foreign”. It is generally recorded at a police station (not filed in courts).
- If police refuse to register, a common escalation is to approach the Superintendent of Police (SP) and/or the Magistrate.
- Do NOT expand FIR as anything else (no US FISA/FBI meanings).

Hard rules:
- Stay within INDIA. Do not mention non-Indian institutions (e.g., "Employment Tribunal").
- If the user question is outside India, ask which Indian state/central law they want, or say you can only answer for India.
- Give a practical roadmap (what to do first, next, where to complain, what documents to keep).
- Include relevant Indian law references (Act/Code name + section number) ONLY when you are confident.
    If you are not confident about a section number, say "common relevant laws include ..." without inventing numbers.
- Do NOT default to "consult a lawyer" / "seek legal advice". Only mention it if the user explicitly asks for professional help,
    or if the situation is high-risk (arrest, violence, imminent deadlines), and then keep it short.
- If the retrieved context conflicts with your knowledge, prefer the context.
- If you don't know, say so.

Extra strictness:
- Do NOT invent foreign procedures/courts (no “federal court”, no “Employment Tribunal”, no “High Court filing” for FIR).
- An FIR is generally recorded at a POLICE STATION in India (not filed in courts).
- If the provided context does not contain the answer, say: "Not found in the uploaded PDFs" and then give a short, careful general India-only explanation without citing section numbers.

Write answers in this structure:
1) Short answer (1–2 lines)
2) Roadmap (numbered steps)
3) Legal references (bullets)
4) What to share next (1 question to clarify: state, worker type, salary frequency)

Context:
{context}

User question: {question}

Answer:
"""

# Return the custom prompt template
def set_custom_prompt_template():
    """
    Set the custom prompt template for the LLMChain
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    return prompt

# Return the LLM
def load_llm():
    """
    Load the Gemini LLM via API
    """
    # Get API key from environment variable or Streamlit secrets
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    # Try to get from Streamlit secrets if not in environment
    if not gemini_api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                gemini_api_key = st.secrets['GEMINI_API_KEY']
        except:
            pass
    
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Local: Create a .env file with GEMINI_API_KEY=your-key. "
            "Streamlit Cloud: Add GEMINI_API_KEY to App Settings → Secrets."
        )
    
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Get model name from env or use default
    # Use actual model names that exist in the API
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    # Get temperature from env or use default
    temperature = float(os.getenv('LLA_TEMPERATURE', '0.6'))
    
    # Get max tokens from env or use default
    max_output_tokens = int(os.getenv('LLA_MAX_NEW_TOKENS', '512'))
    
    # Try different model configurations if the primary fails
    # These are actual model names from genai.list_models()
    model_fallback_list = [
        model_name,
        'gemini-2.5-flash',      # Newest, fast, good quality
        'gemini-flash-latest',   # Alias to latest flash model
        'gemini-2.5-pro',        # Highest quality
        'gemini-pro-latest',     # Alias to latest pro model
        'gemini-2.0-flash',      # Previous generation
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    model_fallback_list = [x for x in model_fallback_list if not (x in seen or seen.add(x))]
    
    last_error = None
    for model_to_try in model_fallback_list:
        try:
            # Create LangChain wrapper for Gemini
            llm = ChatGoogleGenerativeAI(
                model=model_to_try,
                google_api_key=gemini_api_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                convert_system_message_to_human=True
            )
            # Return immediately on success
            return llm
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            # Continue trying if model not found
            if 'not found' in error_msg or '404' in error_msg or 'not supported' in error_msg:
                continue
            else:
                # For other errors, raise immediately
                raise
    
    # If all models fail, raise the last error
    raise RuntimeError(
        f"Failed to load any Gemini model. Last error: {last_error}. "
        f"Tried models: {', '.join(model_fallback_list)}. "
        f"Verify your API key at https://makersuite.google.com/app/apikey"
    )
    
    return llm

# Return the chain
def retrieval_qa_chain(llm, prompt, db):
    """
    Create the Retrieval QA chain
    """
    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

# Return the chain
def qa_pipeline():
    """
    Create the QA pipeline
    """
    # Load embeddings.
    # IMPORTANT: embeddings vector dimension must match the FAISS index dimension.
    env_embeddings_model = os.getenv('LLA_EMBEDDINGS_MODEL', '').strip()
    if env_embeddings_model:
        embeddings_model = env_embeddings_model
    else:
        embeddings_model = _default_embeddings_model_for_dim(_detect_faiss_index_dim())
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    # Load the index
    db = FAISS.load_local("vectorstore/", embeddings, allow_dangerous_deserialization=True)

    # Load the LLM
    llm = load_llm()

    # Set the custom prompt template
    qa_prompt = set_custom_prompt_template()

    # Create the retrieval QA chain
    chain = retrieval_qa_chain(llm, qa_prompt, db)

    return chain
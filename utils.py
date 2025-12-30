# Importing Dependencies
import os

import faiss

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
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
    Load the LLM
    """
    # Model ID (can be overridden)
    # Streamlit Cloud machines are usually RAM-limited.
    # Prefer a small Meta Llama model if the user has HF access+token; otherwise fall back to public small models.
    env_model_id = os.getenv('LLA_MODEL_ID', '').strip()
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')

    prefer_meta = os.getenv('LLA_PREFER_META', '1').strip() in {'1', 'true', 'True', 'yes', 'YES'}
    meta_candidate = 'meta-llama/Llama-3.2-1B-Instruct'

    candidate_model_ids = [
        env_model_id,
        # Try Meta only if token is present (repo is typically gated).
        meta_candidate if (prefer_meta and hf_token) else None,
        # Prefer better instruction-following small models before ultra-tiny ones.
        'Qwen/Qwen2.5-0.5B-Instruct',
        'HuggingFaceTB/SmolLM2-135M-Instruct',
        'sshleifer/tiny-gpt2',
    ]
    candidate_model_ids = [m for m in candidate_model_ids if m]

    load_in_4bit = os.getenv('LLA_LOAD_IN_4BIT', '').strip() in {'1', 'true', 'True', 'yes', 'YES'}
    device_map = os.getenv('LLA_DEVICE_MAP', 'cpu')

    last_exc = None
    for repo_id in candidate_model_ids:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                token=hf_token,
                low_cpu_mem_usage=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_fast=True,
                token=hf_token,
            )
            break
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            if 'gated repo' in message.lower() or '401' in message or 'unauthorized' in message.lower():
                # If user explicitly chose a gated model, fail fast with instructions.
                if env_model_id and repo_id == env_model_id:
                    raise RuntimeError(
                        "Cannot download the Hugging Face model because it is gated or requires authentication. "
                        "Fix: request access on the model page and then run `huggingface-cli login` (or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN). "
                        f"Model: {repo_id}"
                    ) from exc
            continue
    else:
        raise RuntimeError(
            "Failed to load any Hugging Face model. "
            "Try setting LLA_MODEL_ID to a smaller public model, or provide HF_TOKEN if the model is gated. "
            f"Last error: {last_exc}"
        )

    # CPU defaults: keep generations short so responses don't take minutes.
    max_new_tokens = int(os.getenv('LLA_MAX_NEW_TOKENS', '96'))
    # On CPU, the first token can take a while; too-small max_time leads to empty output.
    max_time = float(os.getenv('LLA_MAX_TIME', '60'))
    do_sample = os.getenv('LLA_DO_SAMPLE', '').strip() in {'1', 'true', 'True', 'yes', 'YES'}

    repetition_penalty = float(os.getenv('LLA_REPETITION_PENALTY', '1.12'))
    no_repeat_ngram_size = int(os.getenv('LLA_NO_REPEAT_NGRAM', '4'))

    # Some tokenizers (e.g., GPT-2 style) don't define a pad token.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id

    # Create pipeline
    # IMPORTANT: use max_new_tokens (generated tokens) instead of max_length (prompt+generated)
    # to avoid crashes when the prompt itself reaches the max_length.
    generation_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'max_time': max_time,
        'truncation': True,
        'return_full_text': False,
        'repetition_penalty': repetition_penalty,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'pad_token_id': pad_token_id,
    }
    if do_sample:
        generation_kwargs.update({
            'temperature': float(os.getenv('LLA_TEMPERATURE', '0.6')),
            'top_p': float(os.getenv('LLA_TOP_P', '0.9')),
            'top_k': int(os.getenv('LLA_TOP_K', '40')),
        })

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        **generation_kwargs,
    )

    # Load the LLM
    llm = HuggingFacePipeline(pipeline=pipe)

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
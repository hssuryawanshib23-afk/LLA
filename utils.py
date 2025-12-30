# Importing Dependencies
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Faiss Index Path
FAISS_INDEX = "vectorstore/"

# Custom prompt template (model-agnostic)
custom_prompt_template = """
You are an assistant focused on Indian law and Indian legal procedure.

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
    # NOTE: Llama-2 is gated on Hugging Face and requires login + access.
    # Default to a small, public instruct model that runs on CPU.
    repo_id = os.getenv('LLA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')

    load_in_4bit = os.getenv('LLA_LOAD_IN_4BIT', '').strip() in {'1', 'true', 'True', 'yes', 'YES'}
    device_map = os.getenv('LLA_DEVICE_MAP', 'cpu')

    try:
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            token=hf_token,
        )
    except Exception as exc:
        message = str(exc)
        if 'gated repo' in message.lower() or '401' in message or 'unauthorized' in message.lower():
            raise RuntimeError(
                "Cannot download the Hugging Face model because it is gated or requires authentication. "
                "Fix: request access on the model page and then run `huggingface-cli login` (or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN). "
                f"Model: {repo_id}"
            ) from exc
        raise

    # CPU defaults: keep generations short so responses don't take minutes.
    max_new_tokens = int(os.getenv('LLA_MAX_NEW_TOKENS', '96'))
    # On CPU, the first token can take a while; too-small max_time leads to empty output.
    max_time = float(os.getenv('LLA_MAX_TIME', '60'))
    do_sample = os.getenv('LLA_DO_SAMPLE', '').strip() in {'1', 'true', 'True', 'yes', 'YES'}

    # Create pipeline
    # IMPORTANT: use max_new_tokens (generated tokens) instead of max_length (prompt+generated)
    # to avoid crashes when the prompt itself reaches the max_length.
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(os.getenv('LLA_TEMPERATURE', '0.7')),
        top_p=float(os.getenv('LLA_TOP_P', '0.95')),
        max_time=max_time,
        truncation=True,
        return_full_text=False,
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
    # Load the HuggingFace embeddings
    embeddings_model = os.getenv('LLA_EMBEDDINGS_MODEL', 'sentence-transformers/all-mpnet-base-v2')
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
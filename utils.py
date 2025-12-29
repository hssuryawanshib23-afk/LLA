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

# Custom prompt template (model-agnostic; works across non-Llama instruct models too)
custom_prompt_template = """
You are a helpful assistant that answers questions about Indian law.
Use the provided context when it is relevant. If the context does not contain the answer, say you are not sure.

Context:
{context}

Question: {question}

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
        retriever=db.as_retriever(search_kwargs={'k': 2}),
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
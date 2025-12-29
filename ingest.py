# Importing Dependencies
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Dataset Directory Path
DATASET = "dataset/"

# Faiss Index Path
FAISS_INDEX = "vectorstore/"

# Create Vector Store and Index
def embed_all():
    """
    Embed all files in the dataset directory
    """
    # Create the document loader
    loader = DirectoryLoader(DATASET, glob="*.pdf", loader_cls=PyPDFLoader)
    # Load the documents
    documents = loader.load()
    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    # Split the documents into chunks
    chunks = splitter.split_documents(documents)
    # Load the embeddings
    embeddings = HuggingFaceEmbeddings()
    # Create the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    # Save the vector store
    vector_store.save_local(FAISS_INDEX)

if __name__ == "__main__":
    embed_all()
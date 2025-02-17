import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL = "meta-llama/Llama-3.2-1B"  # Requires HF token and access
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_FILE_PATH = "output.txt"

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
)

# Load and process documents
loader = TextLoader(TEXT_FILE_PATH)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Initialize Llama model
qa_tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_MODEL,
    use_auth_token=True  # Requires HF_TOKEN environment variable
)
qa_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=True
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=qa_model,
    tokenizer=qa_tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Main execution
if __name__ == "__main__":
    query = "What is the main focus of haznain.com?"
    
    # Get response
    result = qa_chain({"query": query})
    
    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})")
import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAQ_URLS, PDF_DIR, FAISS_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

def build_vector_store():
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    docs = []
    
    # Load Web Content
    print("Loading web content...")
    for url in FAQ_URLS:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            print(f"Successfully loaded {url}")
        except Exception as e:
            print(f"Error loading {url}: {e}")
            
    # Load PDFs
    print("Loading PDFs...")
    if os.path.exists(PDF_DIR):
        try:
            pdf_loader = PyPDFDirectoryLoader(PDF_DIR)
            docs.extend(pdf_loader.load())
            print(f"Successfully loaded PDFs from {PDF_DIR}")
        except Exception as e:
            print(f"Error loading PDFs: {e}")
    else:
        print(f"PDF directory {PDF_DIR} does not exist. Skipping.")

    if not docs:
        print("No documents loaded. Please check URLs and PDF directory.")
        return

    # Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    
    # Build FAISS index
    print(f"Building FAISS index with {len(split_docs)} chunks...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # Save index
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index successfully saved to {FAISS_INDEX_DIR}")

if __name__ == "__main__":
    build_vector_store()

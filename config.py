import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdf_file")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
LOG_FILE = os.path.join(BASE_DIR, "chat_logs.csv")

# URLs to ingest
FAQ_URLS = [
    "https://jmi.ac.in/ACADEMICS/Examinations/Frequently-Asked-Questions-(Faqs)"
]

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b" # Groq model

# Chunking configs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

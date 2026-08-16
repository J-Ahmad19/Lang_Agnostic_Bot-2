# Language Agnostic Campus Bot 🏛️

A multilingual, conversational AI assistant designed for campus environments. Built using Retrieval-Augmented Generation (RAG), this bot answers repetitive queries regarding fee deadlines, scholarships, timetable changes, and campus FAQs in the student's preferred language (English, Hindi, regional languages). 

It uses HuggingFace for fast, local document embeddings, FAISS for efficient vector search, and Groq's high-speed LLaMA 3.1 model for intelligent, context-aware responses.

## 🚀 Key Features

* **Multilingual Support:** Automatically detects the user's language and replies in the exact same language.
* **Retrieval-Augmented Generation (RAG):** Sources answers strictly from provided campus PDFs and web FAQs.
* **Conversational Memory:** Remembers chat history for contextual follow-up questions.
* **Human Fallback:** Politely redirects to administration contact details if the answer is not found in the documents.
* **Interaction Logging:** Automatically logs all timestamps, queries, detected languages, and bot responses to `chat_logs.csv` for continuous improvement.
* **High Performance:** Uses Groq Cloud API for lightning-fast inference and HuggingFace for local embeddings to minimize overhead.

---

## 🛠️ Tech Stack

* **Framework:** [Streamlit](https://streamlit.io/) (Frontend UI)
* **LLM Orchestration:** [LangChain](https://www.langchain.com/) (using `langchain_classic` and `langchain_groq`)
* **LLM Engine:** [Groq Cloud API](https://groq.com/) (`llama-3.1-8b-instant`)
* **Embeddings:** [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`)
* **Vector Store:** [FAISS](https://faiss.ai/) (Facebook AI Similarity Search)

---

## 📂 Project Structure

```
Lang_Agnostic_Bot-2/
├── .env                  # Environment variables (Groq API Key)
├── .streamlit/           # Streamlit configurations (Disables file watcher)
├── app.py                # Streamlit Frontend application
├── bot.py                # Core RAG logic, LangChain setup, and LLM definitions
├── config.py             # Global configurations (Model names, chunk sizes, paths)
├── ingestion.py          # Script to process PDFs/Web URLs and build the FAISS index
├── logger.py             # Script handling chat logging to CSV
├── requirements.txt      # Python dependencies
├── pdf_file/             # Directory containing source PDFs
├── faiss_index/          # Directory containing the generated vector database
└── chat_logs.csv         # Auto-generated CSV containing user interactions
```

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone <repository-url>
cd Lang_Agnostic_Bot-2
```

**2. Create a Conda Environment**
```bash
conda create -n campus_bot python=3.10
conda activate campus_bot
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
pip install langchain langchain-community langchain-huggingface pypdf faiss-cpu
```

**4. Set up Environment Variables**
Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📖 Usage Instructions

**Step 1: Ingest Data**
Before running the bot, you need to process your PDFs and web content to build the vector database. Place your campus PDFs inside the `pdf_file/` directory, then run:
```bash
python ingestion.py
```
*(This will download the local embedding model on the first run and save the FAISS index).*

**Step 2: Run the Chatbot**
Once ingestion is complete, start the Streamlit UI:
```bash
streamlit run app.py
```

**Step 3: Access the App**
Open your browser and navigate to `http://localhost:8501`. Start chatting in English, Hindi, or any supported regional language!

---

## 📊 Viewing Logs
All user questions and bot responses are continuously logged for review. You can open `chat_logs.csv` in Excel or Numbers to monitor interactions, track language usage, and discover areas where campus documentation may need updating.
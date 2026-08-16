import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import FAISS_INDEX_DIR, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

def get_vector_store():
    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_DIR}. Please run ingestion.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_conversational_rag_chain(vector_store):
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set in the environment variables or .env file.")
        
    llm = ChatGroq(model=LLM_MODEL_NAME)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt with multilingual and fallback instructions
    qa_system_prompt = """You are a helpful and multilingual campus assistant. 
Use the following pieces of retrieved context to answer the user's question. 
Follow these STRICT rules:
1. Identify the language of the user's question (e.g., Hindi, English, Bengali, Tamil, Telugu, Marathi, etc.).
2. Always respond in the EXACT same language as the user's question. 
3. If the answer is found in the context, provide a clear and concise response.
4. If the answer is NOT found in the context or you are unsure, DO NOT make up an answer. Instead, politely state (in the user's language) that you don't have that information and provide the fallback human contact: "Please contact the administration office at admin@jmi.ac.in or call +91-1234567890 for further assistance."

Context:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

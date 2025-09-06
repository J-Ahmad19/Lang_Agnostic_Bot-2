import streamlit as st
import os
import getpass
import time
import ollama
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()


## LOADING GROQ API KEY FROM ENV VARIABLE   
groq_api_key=os.environ['GROQ_API_KEY']

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

import time


if "vector" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",transport="grpc")
    st.session_state.loader = WebBaseLoader("https://jmi.ac.in/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # Limiting to first 50 documents for demo purposes
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

   
   
    # Load the PDF file
    pdf_file="prospectus.pdf"
   
        
    st.session_state.loader = PyPDFLoader(pdf_file)
    st.session_state.pages = st.session_state.loader.load_and_split()
    


    # Split the pages into smaller chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    st.session_state.pdf = st.session_state.text_splitter.split_documents(st.session_state.pages)

    
    st.session_state.vector = FAISS.from_documents(st.session_state.pdf, st.session_state.embeddings)
    
    

st.title("Language Agnostic ChatBot")
llm=ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-120b")  

##PROMPT TEMPLATE
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

##document prompt chain
document_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {input}
""")




# Define the correct prompt template for the agent
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Use the available tools to answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
 
document_chain=create_stuff_documents_chain(llm,document_prompt)
retriever=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

#create a retriever tool
from langchain.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever,name="Jamia_prospectus_faqs_search",description="Search in Jamia prospectus and faq_jamia.pdf for relevant information using this tool")

#create agents
from langchain.agents import create_tool_calling_agent
agent=create_tool_calling_agent(llm,[retriever_tool],agent_prompt)

#create agent executor  
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

query=st.text_input("Enter your query here")

if query:
    start=time.process_time()
    response=agent_executor.invoke({"input":query})
    print("Response time : ",time.process_time()-start)
    st.write(response['output'])

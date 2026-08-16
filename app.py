import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from bot import get_vector_store, get_conversational_rag_chain
from logger import initialize_logger, log_interaction

st.set_page_config(page_title="Language Agnostic Campus Bot", layout="centered")

def init_app():
    initialize_logger()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "rag_chain" not in st.session_state:
        try:
            vector_store = get_vector_store()
            st.session_state.rag_chain = get_conversational_rag_chain(vector_store)
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.stop()

def main():
    st.title("🏛️ Language Agnostic Campus Bot")
    st.markdown("Ask any questions related to campus, fees, scholarships, or timetables in your preferred language!")
    
    init_app()

    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Chat Input
    query = st.chat_input("Enter your query here...")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
            
        # Invoke the chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.process_time()
                response = st.session_state.rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": query
                })
                end = time.process_time()
                
                answer = response["answer"]
                st.write(answer)
                st.caption(f"Response time: {end - start:.2f}s")
                
        # Update history
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=answer))
        
        # Log interaction
        log_interaction(query, "Auto-detected by LLM", answer)

if __name__ == "__main__":
    main()

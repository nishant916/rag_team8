import streamlit as st
from retrieval import RAGSystem
from config import settings
from logger_config import setup_app_logging

setup_app_logging(mode='a')

# Helper Function to extract text from LLM responses
def extract_llm_text(response):
    """
    Normalizes LangChain / Gemini responses into plain text.
    """
    if isinstance(response, str):               
        return response

    if isinstance(response, list):
        texts = []
        for item in response:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts)

    if isinstance(response, dict) and "text" in response:
        return response["text"]

    return str(response)

# 1. Page Configuration
st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🤖", layout="centered")

# 2. Initialize the RAG System (Cached to avoid reloading on every click)
@st.cache_resource
def load_rag_engine():
    return RAGSystem()

rag_engine = load_rag_engine()

# 3. Sidebar Configuration
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    # Toggle to switch between RAG and General Mode
    use_rag = st.toggle("Enable RAG Mode", value=True, help="When enabled, the bot answers using your local documents.")
    
    st.info(f"**Model:** {settings.models.LLM_MODEL}\n\n**Database:** ChromaDB")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 4. Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Input Logic
if prompt := st.chat_input("Ask me your question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if use_rag:
                # Use your custom RAG logic
                response = rag_engine.get_answer(prompt)
            else:
                # Use Gemini's general knowledge only
                from langchain_google_genai import ChatGoogleGenerativeAI
                general_llm = ChatGoogleGenerativeAI(
                    model=settings.models.LLM_MODEL,
                    google_api_key=settings.google.GOOGLE_API_KEY
                )
                raw_response = general_llm.invoke(prompt).content
                response = extract_llm_text(raw_response)
            
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
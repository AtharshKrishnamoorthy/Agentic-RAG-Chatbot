import streamlit as st
import os
import base64
import time
from typing import Iterator, Optional
import typer
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.models.google import Gemini
from agno.vectordb.lancedb import LanceDb, SearchType
from rich.prompt import Prompt
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


st.set_page_config(page_title="RAG Chatbot", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
    
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
    
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None


def create_kb(pdf_path):
    pdf_knowledge_base = PDFKnowledgeBase(
        path=pdf_path,
        vector_db=LanceDb(
            table_name="recipes",
            uri="tmp_rag/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(dimensions=1536),
        ),
        reader=PDFReader(chunk=True),
    )
    
    pdf_knowledge_base.load(recreate=True)
    return pdf_knowledge_base


def create_agent(knowledge_base, user="User"):
    agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="You are a friendly RAG Agent where you have to provide relevant answers for the question by aiding the Knowledge Base. If the question is asked out of the Knowledge Base just try to manage and answer that too. Use Web Search if needed.",
        user_id=user,
        knowledge=knowledge_base,
        tools=[DuckDuckGoTools()],
        add_references=True,
        search_knowledge=False,
        show_tool_calls=True,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=5,
    )
    return agent


def get_agent_response(agent, message):
    response = agent.run(message)
    return response


def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)


st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")


if not os.path.exists("data"):
    os.makedirs("data")


if uploaded_file is not None and (st.session_state.current_pdf != uploaded_file.name or not st.session_state.pdf_uploaded):
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    st.session_state.current_pdf = uploaded_file.name
    st.session_state.pdf_uploaded = True
    
    
    progress_text = "Creating knowledge base from document..."
    progress_bar = st.sidebar.progress(0)
    
    
    try:
        st.session_state.knowledge_base = create_kb(pdf_path)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        st.sidebar.success(f"Document '{uploaded_file.name}' loaded and processed successfully!")
        
        
        st.session_state.agent = create_agent(st.session_state.knowledge_base)
        
        
        st.sidebar.subheader("Document Preview")
        display_pdf(pdf_path)
        
    except Exception as e:
        st.sidebar.error(f"Error processing document: {e}")
        st.session_state.pdf_uploaded = False


st.title("Agentic RAG Chatbot")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    

    if st.session_state.agent is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_agent_response(st.session_state.agent, user_input)
                st.markdown(response.content)
        

        st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF document in the sidebar first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF document in the sidebar first."})
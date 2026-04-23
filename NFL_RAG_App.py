import asyncio
from typing import List

import chromadb
import streamlit as st
from ddgs import DDGS
from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# =========================
# Configuration
# =========================
CHROMA_PATH = "/Users/zacharysnow/Desktop/chroma_db2"
COLLECTION_NAME = "NFL_Data"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "BAAI/bge-small-en"
SIMILARITY_TOP_K = 5


# =========================
# LlamaIndex global settings
# =========================
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)


# =========================
# Helper functions
# =========================
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first and return the result."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers and return the result. Returns 0 if divisor is 0."""
    if b == 0:
        return 0
    return a / b


def web_search(query: str) -> str:
    """Search the web and return a short text summary of the top results."""
    context: List[str] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            href = result.get("href", "")
            context.append(f"Title: {title}\nSnippet: {body}\nURL: {href}")
    return "\n\n".join(context) if context else "No web results found."


@st.cache_resource
def build_query_engine():
    """Load existing Chroma vector store and create an NFL RAG query engine."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
        storage_context=storage_context,
    )

    custom_prompt = PromptTemplate(
        "You are an NFL stats expert. Use the retrieved context below to answer the question.\n"
        "If the answer is not in the context, clearly say that.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer clearly and concisely using only supported NFL facts from the context."
    )

    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": custom_prompt}
    )
    return query_engine


@st.cache_resource
def build_agent():
    """Create an agent with RAG, web search, and calculator tools."""
    query_engine = build_query_engine()

    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="nfl_rag_search",
            description=(
                "Use this tool for questions about NFL statistics stored in the Chroma database. "
                "This should be the first tool used for NFL stat questions."
            ),
        ),
    )

    search_tool = FunctionTool.from_defaults(fn=web_search)
    add_tool = FunctionTool.from_defaults(fn=add)
    subtract_tool = FunctionTool.from_defaults(fn=subtract)
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    divide_tool = FunctionTool.from_defaults(fn=divide)

    agent = ReActAgent(
        tools=[rag_tool, search_tool, add_tool, subtract_tool, multiply_tool, divide_tool],
        llm=Settings.llm,
        max_iterations=8,
        verbose=True,
        system_prompt=(
            "You are a helpful assistant with access to an NFL RAG database, web search, and calculator tools. "
            "For NFL stats questions, use nfl_rag_search first. "
            "Use web search only when the question is not covered by the NFL database or needs broader information. "
            "Use calculator tools only when math is required. "
            "Avoid repeating the same failed tool call. "
            "After gathering enough information, provide a final answer."
        ),
    )
    return agent


async def run_agent(user_query: str):
    """Run the ReAct agent asynchronously."""
    agent = build_agent()
    response = await agent.run(user_query)
    return response


def extract_response_text(response) -> str:
    """Safely extract text from LlamaIndex agent response."""
    try:
        if hasattr(response, "response") and hasattr(response.response, "blocks"):
            return response.response.blocks[0].text
        return str(response)
    except Exception:
        return str(response)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="NFL RAG + Agent App", layout="wide")

st.title("🏈 NFL RAG + Agent App")
st.caption("Ask NFL stat questions from your Chroma database, or use web search and math tools when needed.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Ollama model:** {OLLAMA_MODEL}")
    st.write(f"**Embedding model:** {EMBED_MODEL}")
    st.write(f"**Chroma collection:** {COLLECTION_NAME}")
    st.write(f"**Top K:** {SIMILARITY_TOP_K}")

    st.markdown("---")
    st.subheader("Example prompts")
    st.markdown(
        """
- Who led the NFL in field goal attempts?
- Which player had the longest reception?
- Compare the leader in passing yards and rushing yards.
- Search the web for the latest Super Bowl host city.
- What is 17 multiplied by 8?
        """
    )

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Ask a question:", placeholder="Example: Who led the NFL in field goal attempts?")

col1, col2 = st.columns([1, 1])
submit_clicked = col1.button("Submit", use_container_width=True)
clear_clicked = col2.button("Clear History", use_container_width=True)

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if submit_clicked:
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = asyncio.run(run_agent(user_query))
                final_answer = extract_response_text(response)
            except Exception as e:
                final_answer = f"Error: {e}"

        st.session_state.history.append(
            {"question": user_query, "answer": final_answer}
        )

        st.success("Answer generated")
        st.markdown("### Answer")
        st.write(final_answer)

if st.session_state.history:
    st.markdown("---")
    st.subheader("Previous Questions")
    for item in reversed(st.session_state.history):
        with st.expander(item["question"]):
            st.write(item["answer"])

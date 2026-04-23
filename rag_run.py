import chromadb
from chromadb import PersistentClient
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader,StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

llm = None

Settings.llm=Ollama(model="llama3.2", request_timeout=360.0)

chroma_client = chromadb.PersistentClient(path="/Users/zacharysnow/Desktop/chroma_db2")
chroma_collection = chroma_client.get_or_create_collection("NFL_Data")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

custom_prompt = PromptTemplate(
    "You are an NFL stats expert. Use the following context to answer the question.\n"
    "Context: {context_str}\n"
    "Question: {query_str}\n"
    "Answer in a clear and concise way, focusing only on NFL facts."
)

query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=5)
query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt})

response = query_engine.query("Who led the NFL in the Stat Rushing TDs?")
print(response)


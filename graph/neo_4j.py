from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from data.fetch import doc_splits
from langchain_core.documents import Document

_ = load_dotenv(find_dotenv())
api = os.getenv("GROQ_API_KEY")

LLM_NAME = "openai/gpt-oss-120b"
RANDOM_SEED = 415
FREQUENCY_PENALTY = 2
BATCH_SIZE = 3

graph_llm = ChatGroq(
    api_key = api,
    model = LLM_NAME,
    temperature=0.1
)
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="40d4e79a"
)
graph_transformer = LLMGraphTransformer(
    llm=graph_llm,
    allowed_nodes=["Paper", "Author", "Topic"],
    node_properties=["title", "summary", "url"],
    allowed_relationships=["AUTHORED", "DISCUSSES", "RELATED_TO"],
)

graph_transformer.prompt = """
Extract a knowledge graph.

IMPORTANT:
- Create a Paper node for each document
- Paper MUST include:
  - title
  - summary
  - url

Do NOT reduce papers to single keywords.
"""

def batch_iterable(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

graph_docs_input = [
    Document(
        page_content = f"""
        Title: {doc.metadata.get("title")}
        Summary: {doc.page_content}
        URL: {doc.metadata.get("url")}
        """,
        metadata=doc.metadata
    ) for doc in doc_splits
]

graph_documents = []

for batch in tqdm(batch_iterable(graph_docs_input, BATCH_SIZE)):
    try:
        docs = graph_transformer.convert_to_graph_documents(batch)
        graph_documents.extend(docs)

        time.sleep(2)
    except Exception as e:
        print("Batch Failed:", e)
        time.sleep(5)

graph.add_graph_documents(graph_documents)

print(f"Graph documents: {len(graph_documents)}")
print(f"Nodes from 1st graph doc:{graph_documents[0].nodes}")
print(f"Relationships from 1st graph doc:{graph_documents[0].relationships}")


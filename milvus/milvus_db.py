import os
import arxiv
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

# ── 1. Environment & Zilliz connection ───────────────────────────────────────
_ = load_dotenv(find_dotenv())

ZILLIZ_TOKEN     = os.environ["ZILLIZ_API_KEY"]
CLUSTER_ENDPOINT = "https://in03-5223ff782a72af1.serverless.aws-eu-central-1.cloud.zilliz.com"
COLLECTION_NAME  = "graph_rag_arxiv"

mc = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=ZILLIZ_TOKEN
)

schema = mc.create_schema()

schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=768  # for BGE-base
)

# ✅ Scalar fields for tracking
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=2000
)

schema.add_field(
    field_name="source",
    datatype=DataType.VARCHAR,
    max_length=500
)

schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=500
)

# ── 2. Arxiv search ───────────────────────────────────────────────────────────
search_query = "agent OR 'large language model' OR 'prompt engineering'"
max_results  = 20

client = arxiv.Client()
search = arxiv.Search(
    query=search_query,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.Relevance,
)

docs = []
for result in client.results(search):
    docs.append({
        "title":   result.title,
        "summary": result.summary,
        "url":     result.entry_id,
    })

print(f"Papers fetched: {len(docs)}")


# ── 3. Chunk documents ────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=50,
)
doc_splits = text_splitter.create_documents(
    [doc["summary"] for doc in docs],
    metadatas=docs,
)

print(f"Chunks created: {len(doc_splits)}")


#  -------------------------------------------------------

index_params = mc.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 32}
)
if mc.has_collection(COLLECTION_NAME):
    mc.drop_collection(COLLECTION_NAME)

mc.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
    consistency_level="Eventually"
)

# ── 4. BAAI BGE embedding model ───────────────────────────────────────────────
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

texts = [doc.page_content for doc in doc_splits]

vectors = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True
)

# ── 5. Ingest into Zilliz Cloud ───────────────────────────────────────────────
data = []

for i, doc in enumerate(doc_splits):
    data.append({
        "vector": vectors[i].tolist(),
        "text": doc.page_content,
        "source": doc.metadata.get("url", ""),
        "title": doc.metadata.get("title", "")
    })

mc.insert(
    collection_name=COLLECTION_NAME,
    data=data
)

mc.flush(COLLECTION_NAME)
print(f"Collection `{COLLECTION_NAME}` ready with {len(data)} vectors.")


# ── 6. Quick sanity-check query ───────────────────────────────────────────────
query = "How do LLM agents use tools?"

query_vector = model.encode(
    [query],
    normalize_embeddings=True
)[0].tolist()

results = mc.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=5,
    output_fields=["text", "source", "title"]
)

for i, hits in enumerate(results):
    for j, hit in enumerate(hits):
        print(f"\n[{j+1}] {hit['entity']['title']}")
        print(f"Source: {hit['entity']['source']}")
        print(f"Text: {hit['entity']['text'][:200]}...")
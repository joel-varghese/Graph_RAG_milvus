import arxiv
from langchain_text_splitters import RecursiveCharacterTextSplitter

search_query = "agent OR 'large language model' OR 'prompt engineering'"
max_results  = 20

client = arxiv.Client(page_size=20)
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
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from pymilvus import MilvusClient
import os
_ = load_dotenv(find_dotenv())


LLM_NAME = "llama-3.1-8b-instant"
ZILLIZ_TOKEN     = os.environ["ZILLIZ_API_KEY"]
CLUSTER_ENDPOINT = "https://in03-5223ff782a72af1.serverless.aws-eu-central-1.cloud.zilliz.com"
COLLECTION_NAME  = "graph_rag_arxiv"
api = os.getenv("GROQ_API_KEY")



llm = ChatGroq(
    api_key = api,
    model = LLM_NAME,
    temperature=0.1
)
mc = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=ZILLIZ_TOKEN
)


prompt = PromptTemplate(
    template="""You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     
    Here is the retrieved document: 
    {document}
    
    Here is the user question: 
    {question}
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

query = "Do we have articles that talk about Prompt Engineering?"

query_vector = model.encode(
    [query],
    normalize_embeddings=True
)[0].tolist()

retrieved_docs = []

results = mc.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=5,
    output_fields=["text", "source", "title"]
)

for hits in results:
    for hit in hits:
        retrieved_docs.append({
            "text": hit["entity"]["text"],
            "title": hit["entity"]["title"],
            "source": hit["entity"]["source"],
            "score": hit["distance"]
        })

graded_results = []

for doc in retrieved_docs:
    try:
        grade = retrieval_grader.invoke({
            "question": query,
            "document": doc["text"]
        })

        doc["grade"] = grade["score"]
        graded_results.append(doc)
    except Exception as e:
        print("Grading failed:", e)

relevant_docs = [doc for doc in graded_results if doc["grade"] == "yes"]
print("=== Relevant Documents ===")

for i, doc in enumerate(relevant_docs):
    print(f"[{i+1}] {doc['title']}")
    print(f"Score: {doc['score']}")
    print(f"Source: {doc['source']}")
    print(f"Text: {doc['text'][:200]}...\n")
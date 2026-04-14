from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_classic.chains import GraphCypherQAChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from pymilvus import MilvusClient
import os
_ = load_dotenv(find_dotenv())

LLM_NAME = "openai/gpt-oss-120b"

llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    model = LLM_NAME,
    temperature=0.1
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="40d4e79a"
)

cypher_prompt = PromptTemplate(
    template="""You are an expert at generating Cypher queries for Neo4j.
    Use the following schema to generate a Cypher query that answers the given question.
    Make the query flexible by using case-insensitive matching and partial string matching where appropriate.
    Focus on searching paper titles as they contain the most relevant information.
    
    Schema:
    {schema}
    
    Question: {question}
    
    Cypher Query:""",
    input_variables=["schema", "question"],
)

qa_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following Cypher query results to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise. If topic information is not available, focus on the paper titles.
    
    Question: {question} 
    Cypher Query: {query}
    Query Results: {context} 
    
    Answer:""",
    input_variables=["question", "query", "context"],
)


graph_rag_chain = GraphCypherQAChain.from_llm(
    cypher_llm = llm,
    qa_llm=llm,
    validate_cypher=True,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    return_direct=True,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    allow_dangerous_requests=True
)

question = "What paper talks about Multi-Agent?"
generation = graph_rag_chain.invoke({"query": question})
print(generation)
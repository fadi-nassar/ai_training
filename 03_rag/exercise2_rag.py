import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# STEP 1: our "documents" - normally these would come from files, a database, etc.
documents = [
    Document(page_content="LangGraph is a library for building stateful, multi-agent applications with LLMs. It represents workflows as graphs with nodes and edges."),
    Document(page_content="Groq provides ultra-low latency LLM inference using custom hardware called LPUs (Language Processing Units), making it much faster than typical GPU-based inference."),
    Document(page_content="A checkpointer in LangGraph, such as MemorySaver, saves the state of a graph at every step, enabling memory across multiple invocations using a thread_id."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It combines a retrieval step, pulling relevant documents from a knowledge base, with a generation step, where an LLM produces an answer using those documents as context."),
    Document(page_content="LangSmith is a platform for tracing, monitoring, and evaluating LLM applications. It shows the full execution path of a chain or graph, including token usage and latency."),
]

# STEP 2: embeddings - converts text into vectors representing meaning
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 3: vector store - indexes our documents so we can search by meaning
vectorstore = FAISS.from_documents(documents, embeddings)

# STEP 4: retriever - the interface for searching the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # return top 2 matches

# STEP 5: the generation chain - takes retrieved context + question, produces an answer
llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context:\n\n{context}"),
    ("human", "{question}"),
])

parser = StrOutputParser()
rag_chain = prompt | llm | parser

# STEP 6: tie it together - retrieve, then generate
def answer_question(question):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print(f"\n--- Retrieved context for: '{question}' ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i+1}] {doc.page_content}")

    answer = rag_chain.invoke({"context": context, "question": question})
    print(f"\n--- Answer ---\n{answer}")

# TEST WITH DIFFERENT QUERIES
test_questions = [
    "What makes Groq fast?",
    "How does memory work in LangGraph?",
    "What is RAG?",
]

for q in test_questions:
    answer_question(q)
    print("\n" + "="*60)
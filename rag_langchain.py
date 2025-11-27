import os
from pathlib import Path

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
OPENAI_MODEL = "gpt-4o-mini"

def load_documents():
    docs = []
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path)).load())
        elif path.suffix.lower() == ".md":
            docs.extend(TextLoader(str(path)).load())
        elif path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
    return docs

def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def rag_langchain(query):
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1
    )

    documents = load_documents()
    vector_store = build_vector_store(documents)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.invoke(query)

    context_text = "\n\n".join(d.page_content for d in context_docs)

    prompt = f"""
You are a helpful assistant. Answer ONLY using the context.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, context_docs



if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        answer, sources = rag_langchain(q)

        print("\n--- ANSWER (LANGCHAIN) ---")
        print(answer)
        print("\n--- SOURCES ---")
        for s in sources:
            print(f"- {s.metadata.get('source', 'unknown')}")

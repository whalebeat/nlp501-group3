import os
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# ==========================================
# EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device": "cuda"  # or "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# ==========================================
# LOAD VECTOR DB
# ==========================================

vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ==========================================
# LOCAL LLM (OLLAMA)
# ==========================================

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
Bạn là trợ lý học vụ HUST.

Quy tắc:

- Trả lời trực tiếp.
- Trả lời như đang tư vấn trực tiếp cho sinh viên.
- Không tự bịa ra thông tin. Nếu không biết, hãy nói "Tôi không tìm thấy thông tin!"
"""

# ==========================================
# ASK FUNCTION
# ==========================================

def ask(question: str):

    t0 = time.time()
    docs = retriever.invoke(question)

    print("Retrieval:", time.time() - t0)
    t1 = time.time()

    context_parts = []

    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"""
Nguồn: {source}
Trang: {page}

{doc.page_content}
"""
        )

        src = f"{source} - trang {page}"

        if src not in sources:
            sources.append(src)

    context = "\n\n".join(context_parts)

    prompt = f"""
{SYSTEM_PROMPT}

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{question}

======================
ANSWER
======================
"""

    response = llm.invoke(prompt)
    print("Context length:", len(response.content))
    print("LLM:", time.time() - t1)

    print("Total:", time.time() - t0)

    return {
        "answer": response.content,
        "sources": sources
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    while True:

        q = input("\nQuestion: ")

        if q.lower() in ["exit", "quit"]:
            break

        result = ask(q)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for s in result["sources"]:
            print("-", s)
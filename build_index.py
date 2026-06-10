import os
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

documents = []
total_pages = 0

# =====================
# Load PDF Documents
# =====================
for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, file)

        doc = fitz.open(pdf_path)

        total_pages += len(doc)

        for page_num in range(len(doc)):

            text = doc[page_num].get_text()

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "page": page_num + 1
                        }
                    )
                )

print(f"Total Pages: {total_pages}")
print(f"Document Pages Loaded: {len(documents)}")

# =====================
# Chunking
# =====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

# =====================
# Dataset Statistics
# =====================
total_chunks = len(chunks)

chunk_lengths = [
    len(chunk.page_content)
    for chunk in chunks
]

avg_chunk_length = (
    sum(chunk_lengths) / total_chunks
)

min_chunk_length = min(chunk_lengths)
max_chunk_length = max(chunk_lengths)

print("\n===== DATASET STATISTICS =====")
print(f"Total Pages          : {total_pages}")
print(f"Total Chunks         : {total_chunks}")
print(f"Average Chunk Length : {avg_chunk_length:.2f} chars")
print(f"Minimum Chunk Length : {min_chunk_length} chars")
print(f"Maximum Chunk Length : {max_chunk_length} chars")

# =====================
# Embedding + FAISS
# =====================
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)

vector_db.save_local(VECTOR_DB_PATH)

print("\nVector DB saved.")
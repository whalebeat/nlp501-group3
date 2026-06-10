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

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, file)

        doc = fitz.open(pdf_path)

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

print("Documents:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)

vector_db.save_local(VECTOR_DB_PATH)

print("Vector DB saved.")
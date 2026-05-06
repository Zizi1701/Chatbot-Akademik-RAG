import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def create_vector_db():
    print("1. Membaca dokumen PDF...")
    pdf_path = "data/BUKU PEDOMAN AKADEMIK.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File tidak ditemukan di '{pdf_path}'")
        return

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"❌ Error membaca PDF: {e}")
        return

    print("2. Chunking teks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"   Total: {len(chunks)} chunks")

    if len(chunks) == 0:
        print("❌ Tidak ada teks yang terbaca dari PDF.")
        return

    print("3. Membuat embeddings...")
    # Menggunakan model Multilingual yang lebih pintar bahasa Indonesia
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("4. Menyimpan ke FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local("vectorstore/faiss_index")
    print("✅ Selesai! Vectorstore berhasil disimpan.")

if __name__ == "__main__":
    create_vector_db()
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def create_vector_db():
    print("1. Membaca semua dokumen PDF di folder data/...")
    # Modifikasi 2: Arahkan ke folder, bukan ke satu file spesifik
    data_dir = "data/" 

    if not os.path.exists(data_dir):
        print(f"❌ Error: Folder tidak ditemukan di '{data_dir}'")
        return

    try:
        # Modifikasi 3: Membaca seluruh file .pdf yang ada di dalam folder
        loader = PyPDFDirectoryLoader(data_dir)
        documents = loader.load()
        print(f"   Berhasil memuat {len(documents)} halaman dari seluruh PDF.")
    except Exception as e:
        print(f"❌ Error membaca PDF: {e}")
        return

    if len(documents) == 0:
        print("❌ Tidak ada teks atau dokumen PDF yang terbaca di folder data/.")
        return

    print("2. Chunking teks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"   Total: {len(chunks)} chunks")

    if len(chunks) == 0:
        print("❌ Tidak ada teks yang terbaca untuk di-chunking.")
        return

    print("3. Membuat embeddings...")
    # Menggunakan model Multilingual yang lebih pintar bahasa Indonesia
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("4. Menyimpan ke FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local("vectorstore/faiss_index")
    print("✅ Selesai! Vectorstore berhasil menyimpan semua dokumen.")

if __name__ == "__main__":
    create_vector_db()
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Konfigurasi Halaman Tampilan Web ---
st.set_page_config(page_title="Chatbot Akademik RAG", page_icon="🤖", layout="centered")

st.title("🤖 Chatbot Akademik Fasilkom")
st.markdown("Tanyakan apa saja seputar pedoman akademik, jadwal kelas, panduan magang, atau informasi dosen.")
st.divider()

# --- 2. Load Model & Database (Di-cache agar web tidak lambat saat di-refresh) ---
@st.cache_resource
def load_knowledge_base():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY tidak ditemukan di .env!")
        st.stop()

    if not os.path.exists("vectorstore/faiss_index"):
        st.error("❌ Vectorstore belum ada! Jalankan dulu di terminal: `python ingest_data.py`")
        st.stop()

    # Load Embeddings & FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # MENGGUNAKAN MMR: Mengambil sampel topik yang bervariasi agar dokumen jadwal tidak tertutup dokumen besar
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

    # Load LLM Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Prompt Sistem Ketat Anti-Halusinasi
    system_prompt = (
        "Kamu adalah asisten akademik kampus yang sangat cerdas dan ramah. "
        "Gunakan HANYA potongan konteks dokumen berikut untuk menjawab pertanyaan mahasiswa. "
        "Jika jawabannya tidak ada di dalam konteks, katakan dengan jujur: "
        "'Maaf, saya tidak menemukan informasi tersebut di dalam buku pedoman kampus.' "
        "Jangan pernah mengarang jawaban dari luar dokumen.\n\n"
        "Konteks Dokumen:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    return rag_chain

# Panggil fungsi load database dan model
rag_chain = load_knowledge_base()

# --- 3. Memori Obrolan (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat obrolan sebelumnya di layar browser
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Lihat Sumber Sitasi"):
                for source in msg["sources"]:
                    st.markdown(source)

# --- 4. Kolom Input Pertanyaan Mahasiswa ---
if pertanyaan := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Tampilkan pertanyaan pengguna ke layar
    st.session_state.messages.append({"role": "user", "content": pertanyaan})
    with st.chat_message("user"):
        st.markdown(pertanyaan)

    # Proses pencarian dokumen (Retrieval) dan generasi jawaban oleh Gemini (Generation)
    with st.chat_message("assistant"):
        with st.spinner("⏳ Mencari di berbagai dokumen panduan..."):
            try:
                response = rag_chain.invoke({"input": pertanyaan})
                jawaban_ai = response["answer"]
                
                # Tampilkan jawaban utama chatbot
                st.markdown(jawaban_ai)

                # Ekstrak Informasi Metadata untuk Sitasi Multi-PDF
                daftar_sitasi = []
                if "context" in response and response["context"]:
                    with st.expander("📚 Lihat Sumber Sitasi"):
                        for doc in response["context"]:
                            halaman = doc.metadata.get('page', '?')
                            # Mengambil nama file PDF asli secara dinamis (bukan path folder lengkap)
                            nama_file = os.path.basename(doc.metadata.get('source', 'Dokumen Akademik'))
                            kutipan = doc.page_content[:120].replace('\n', ' ')
                            
                            # Format tampilan sitasi untuk web
                            teks_sitasi = f"- **{nama_file} (Halaman {halaman})**\n> *\"{kutipan}...\"*"
                            st.markdown(teks_sitasi)
                            daftar_sitasi.append(teks_sitasi)

                # Simpan jawaban beserta sitasinya ke memori session web
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": jawaban_ai,
                    "sources": daftar_sitasi
                })

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan sistem: {e}")
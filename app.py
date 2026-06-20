#!/usr/bin/env python3
"""
Kitap İçeriğini Pinecone Vektör Veritabanına Yükleme - Streamlit Arayüzü
=========================================================================
PDF/EPUB dosyalarını yükleyip Pinecone'a vektör olarak yükler.
Arama: RAG (Retrieval Augmented Generation) ile MiniMax LLM kullanır.

Yazar: Claude
Tarih: 2026-06-20
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# --- PDF İşleme ---
try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_LIBRARY = "pypdf"
    except ImportError:
        PDF_LIBRARY = None

# --- EPUB İşleme ---
try:
    import ebooklib
    from ebooklib import epub
    EPUB_LIBRARY = "ebooklib"
except ImportError:
    EPUB_LIBRARY = None

# --- LangChain ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Ollama / LangChain Embeddings ---
try:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --- Pinecone ---
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# LOGLAMA YAPILANDIRMASI
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SABİTLER
# =============================================================================

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_DIMENSION = 768  # nomic-embed-text = 768 boyut
DEFAULT_LLM_MODEL = "MiniMax-M2.7"  # MiniMax model

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def fix_reversed_text(text: str) -> str:
    """
    PDF'lerden çıkarılan metinlerde sütun sırası ters olabilir.
    Özellikle 2 sütunlu PDF'lerde satırdaki kelimeler ters sırada gelir.
    Bu fonksiyon satırlardaki kelime sırasını düzeltir.
    """
    if not text:
        return text

    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        words = line.split()
        if len(words) > 3:
            reversed_line = ' '.join(reversed(words))
            if len(reversed_line) <= len(line):
                line = reversed_line
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def clean_text(text: str) -> str:
    """Ham metinden gereksiz boşlukları ve gürültleri temizler."""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?:Page|Sayfa)\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*/\s*\d+', '', text)
    text = re.sub(r'(?:^|\n)\s*\d+\s*(?:\n|$)', '\n', text)
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # PDF sütun sırası düzeltmesi
    text = fix_reversed_text(text)

    return text


def extract_text_from_pdf(pdf_path: str, progress_callback=None) -> str:
    """PDF dosyasından metin çıkarır."""
    full_text = []

    if PDF_LIBRARY == "pdfplumber":
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    cleaned = clean_text(text)
                    if cleaned:
                        full_text.append(cleaned)
                if progress_callback:
                    progress_callback(page_num / total_pages)
    else:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                cleaned = clean_text(text)
                if cleaned:
                    full_text.append(cleaned)
            if progress_callback:
                progress_callback(page_num / total_pages)

    return "\n\n".join(full_text)


def extract_text_from_epub(epub_path: str, progress_callback=None) -> str:
    """EPUB dosyasından metin çıkarır."""
    full_text = []

    book = epub.read_epub(epub_path)
    items = list(book.get_items())
    total_items = len(items)

    for idx, item in enumerate(items):
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            text = re.sub(r'<[^>]+>', ' ', content)
            text = clean_text(text)
            if text:
                full_text.append(text)
        if progress_callback:
            progress_callback((idx + 1) / total_items)

    return "\n\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 750, chunk_overlap: int = 75) -> List[str]:
    """Metni parçalara ayırır."""
    if not LANGCHAIN_AVAILABLE:
        return _simple_chunk_text(text, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False
    )

    return splitter.split_text(text)


def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """LangChain olmadığında basit parçalama."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def initialize_pinecone(
    index_name: str,
    dimension: int = 1536,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
) -> Pinecone:
    """Pinecone istemcisini başlatır ve indeks yoksa oluşturur."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY bulunamadı")

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        # İndeks hazır olana kadar bekle
        import time
        while not pc.describe_index(index_name).status.ready:
            time.sleep(1)

    return pc


def create_embeddings(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    """
    Ollama API kullanarak embedding vektörleri oluşturur.
    LangChain'in OllamaEmbeddings sınıfını kullanır.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("OllamaEmbeddings mevcut değil. langchain-community kurulu olmalı.")

    embed = OllamaEmbeddings(
        model=model,
        base_url="http://localhost:11434"
    )

    embeddings = embed.embed_documents(texts)

    return embeddings


def generate_with_minimax(
    context: str,
    question: str,
    model: str = "MiniMax-M2.7"
) -> str:
    """
    MiniMax Anthropic-compatible API kullanarak RAG tabanlı yanıt üretir.
    """
    import anthropic

    api_key = os.getenv("MINIMAX_API_KEY")

    if not api_key:
        raise ValueError("MINIMAX_API_KEY bulunamadı")

    # Anthropic-compatible API
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url="https://api.minimax.io/anthropic"
    )

    # Sistem mesajı
    system_prompt = """Sen bir kitap içeriği hakkında soruları yanıtlayan asistansın.
Verilen bağlamdaki bilgilere göre Türkçe yanıt ver.
Sadece verilen bağlamdan bilgi kullan. Bağlamda cevap yoksa bilgi bulunamadığını söyle."""

    # Kullanıcı mesajı
    user_prompt = f"""Bağlam:
{context}

Soru: {question}

Türkçe yanıt:"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
        )

        # Yanıtı çıkar
        for block in message.content:
            if block.type == "text":
                return block.text

        return str(message.content)

    except Exception as e:
        logger.error(f"MiniMax API hatası: {e}")
        raise


def upsert_to_pinecone(index, vectors: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """Vektörleri Pinecone indeksine yükler."""
    total_vectors = len(vectors)
    uploaded_count = 0

    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        uploaded_count += len(batch)

    return uploaded_count


# =============================================================================
# STREAMLIT ARAYÜZÜ
# =============================================================================

def main():
    st.set_page_config(
        page_title="Kitap → Pinecone Vector Store",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Pinecone Vector Database - Kitap Arama")
    st.markdown("PDF veya EPUB dosyası yükleyin, vektörleştirin ve Pinecone'a aktarın.")

    # --- Sidebar Ayarları ---
    st.sidebar.header("⚙️ Ayarlar")

    # API Anahtarı Kontrolü
    pinecone_key = os.getenv("PINECONE_API_KEY")
    minimax_key = os.getenv("MINIMAX_API_KEY")
    minimax_group = os.getenv("MINIMAX_GROUP_ID")

    if not pinecone_key:
        st.sidebar.error("⚠️ PINECONE_API_KEY ayarlanmamış")
    else:
        st.sidebar.success("✓ Pinecone API Key OK")

    if not minimax_key or not minimax_group:
        st.sidebar.warning("⚠️ MiniMax ayarlanmamış (RAG çalışmaz)")
    else:
        st.sidebar.success("✓ MiniMax API OK")

    st.sidebar.info("✓ Ollama yerel embedding kullanılıyor")

    # Parametre Ayarları
    st.sidebar.subheader("📊 İşlem Parametreleri")

    chunk_size = st.sidebar.slider(
        "Parça Boyutu (karakter)",
        min_value=100,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50
    )

    chunk_overlap = st.sidebar.slider(
        "Parça Örtüşmesi (karakter)",
        min_value=0,
        max_value=200,
        value=DEFAULT_CHUNK_OVERLAP,
        step=5
    )

    embedding_model = st.sidebar.text_input(
        "Embedding Modeli",
        value=DEFAULT_EMBEDDING_MODEL
    )

    index_name = st.sidebar.text_input(
        "Pinecone İndeks Adı",
        value="book-vectors"
    )

    batch_size = st.sidebar.slider(
        "Yükleme Batch Boyutu",
        min_value=10,
        max_value=200,
        value=100,
        step=10
    )

    # --- Sekme Sistemi ---
    tab1, tab2 = st.tabs(["📤 Yükle", "🔍 Ara"])

    with tab1:
        st.subheader("📄 Dosya Yükle")
        uploaded_file = st.file_uploader(
            "PDF veya EPUB dosyası seçin",
            type=["pdf", "epub"],
            help="Desteklenen formatlar: PDF, EPUB"
        )

        if uploaded_file:
            st.success(f"✓ {uploaded_file.name} yüklendi ({uploaded_file.size / 1024:.1f} KB)")

            # Önizleme
            with st.expander("🔍 Dosya Önizleme"):
                st.write(f"**Dosya Adı:** {uploaded_file.name}")
                st.write(f"**Boyut:** {uploaded_file.size:,} bytes")
                st.write(f"**Tür:** {uploaded_file.type}")

        st.subheader("🚀 İşlem Durumu")

        if uploaded_file and pinecone_key:
            if st.button("▶️ İşlemi Başlat", type="primary", use_container_width=True):

                # Geçici dosyaya kaydet
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_file.name

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # --- Aşama 1: Metin Çıkarma ---
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("📖 Metin çıkarılıyor...")
                    progress_bar.progress(0.1)

                    def extract_progress(p):
                        progress_bar.progress(0.1 + p * 0.3)

                    if uploaded_file.type == "application/pdf":
                        raw_text = extract_text_from_pdf(str(temp_path), extract_progress)
                    else:
                        raw_text = extract_text_from_epub(str(temp_path), extract_progress)

                    progress_bar.progress(0.4)
                    status_text.text(f"✓ Metin çıkarıldı: {len(raw_text):,} karakter")

                    # --- Aşama 2: Parçalama ---
                    status_text.text("✂️ Metin parçalanıyor...")
                    progress_bar.progress(0.45)

                    chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    progress_bar.progress(0.55)
                    status_text.text(f"✓ {len(chunks)} parça oluşturuldu")

                    # --- Aşama 3: Embedding ---
                    status_text.text("🧠 Embedding oluşturuluyor...")
                    progress_bar.progress(0.6)

                    embeddings = create_embeddings(chunks, model=embedding_model)
                    progress_bar.progress(0.75)
                    status_text.text(f"✓ {len(embeddings)} embedding vektörü oluşturuldu")

                    # --- Aşama 4: Pinecone ---
                    status_text.text("🗄️ Pinecone başlatılıyor...")
                    progress_bar.progress(0.8)

                    pc = initialize_pinecone(index_name, dimension=DEFAULT_DIMENSION)
                    index = pc.Index(index_name)

                    status_text.text("⬆️ Pinecone'a yükleniyor...")
                    progress_bar.progress(0.85)

                    # Vektörleri hazırla
                    vectors = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        vector_id = f"{Path(uploaded_file.name).stem}_{i}"
                        vectors.append({
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "text": chunk,
                                "source": uploaded_file.name,
                                "chunk_index": i
                            }
                        })

                    uploaded = upsert_to_pinecone(index, vectors, batch_size=batch_size)
                    progress_bar.progress(1.0)
                    status_text.text("✅ İşlem tamamlandı!")

                    # --- Sonuç Özeti ---
                    st.balloons()

                    st.success(f"""
                    ### ✅ İşlem Başarıyla Tamamlandı!

                    | Bilgi | Değer |
                    |-------|-------|
                    | **Dosya** | {uploaded_file.name} |
                    | **Çıkarılan Karakter** | {len(raw_text):,} |
                    | **Oluşturulan Parça** | {len(chunks)} |
                    | **Yüklenen Vektör** | {uploaded} |
                    | **Hedef İndeks** | `{index_name}` |
                    """)

                    # Metadata örneği göster
                    with st.expander("📋 Örnek Vektör Metadata"):
                        st.json(vectors[0]["metadata"])

                except Exception as e:
                    st.error(f"❌ Hata oluştu: {str(e)}")
                    logger.error(f"İşlem hatası: {e}")

                finally:
                    # Geçici dosyayı temizle
                    if temp_path.exists():
                        temp_path.unlink()

        elif not uploaded_file:
            st.info("👆 Yukarıdan bir dosya yükleyin")
        else:
            st.warning("⚠️ Lütfen .env dosyasında API anahtarlarını ayarlayın")

    # =============================================================================
    # ARAMA / RAG SEKMESİ
    # =============================================================================
    with tab2:
        st.subheader("🔍 Kitap İçerisinde Ara (RAG)")

        if not pinecone_key:
            st.error("⚠️ Pinecone API Key ayarlanmamış. Lütfen .env dosyasını kontrol edin.")
        else:
            # Arama parametreleri
            col_search, col_top = st.columns([3, 1])

            with col_search:
                query_text = st.text_input(
                    "Sorunuzu Türkçe olarak sorun",
                    placeholder="Örn: Bu kitapta ana fikir nedir?",
                    help="Pinecone'dan ilgili içerikleri alır ve MiniMax ile yanıt üretir"
                )

            with col_top:
                top_k = st.number_input("İlgili parça sayısı", min_value=3, max_value=20, value=5)

            # Sorgu butonu
            search_button = st.button("🤖 Soru Sor", type="primary", use_container_width=True)

            if search_button and query_text:
                with st.spinner("🤔 Yanıt hazırlanıyor..."):
                    try:
                        # 1. Query'yi embedding'e çevir
                        query_embedding = create_embeddings([query_text], model=embedding_model)[0]

                        # 2. Pinecone'a bağlan ve ara
                        pc = initialize_pinecone(index_name, dimension=DEFAULT_DIMENSION)
                        index = pc.Index(index_name)

                        # 3. Similarity search - daha fazla parça al
                        search_results = index.query(
                            vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True
                        )

                        matches = search_results.get('matches', [])

                        if not matches:
                            st.warning("😕 İlgili içerik bulunamadı. Farklı kelimeler deneyin.")
                        else:
                            # 4. Bağlam metnini hazırla
                            context_parts = []
                            sources_info = []

                            for match in matches:
                                metadata = match.get('metadata', {})
                                text = metadata.get('text', '')
                                source = metadata.get('source', 'Bilinmeyen')
                                chunk_idx = metadata.get('chunk_index', 0)
                                score = match.get('score', 0)

                                context_parts.append(f"[Parça {chunk_idx} - %{score*100:.0f} benzerlik]\n{text}")
                                sources_info.append(f"{source} (Parça {chunk_idx})")

                            context_text = "\n\n---\n\n".join(context_parts)

                            # 5. MiniMax ile yanıt üret
                            if minimax_key and minimax_group:
                                with st.spinner("🧠 MiniMax yanıt üretiyor..."):
                                    answer = generate_with_minimax(
                                        context=context_text,
                                        question=query_text
                                    )

                                # Yanıtı göster
                                st.success("✅ Yanıt hazır!")

                                st.markdown("### 💬 Yanıt:")
                                st.markdown(f">{answer}")

                                # Kullanılan kaynaklar
                                with st.expander("📚 Kullanılan Kaynaklar"):
                                    unique_sources = list(dict.fromkeys(sources_info))
                                    for src in unique_sources:
                                        st.caption(f"• {src}")

                                # İstatistikler
                                with st.expander("📊 Arama Detayları"):
                                    st.json({
                                        "Sorgu": query_text,
                                        "Bulunan Parça": len(matches),
                                        "Kullanılan İndeks": index_name,
                                        "Embedding Modeli": embedding_model,
                                        "LLM": DEFAULT_LLM_MODEL
                                    })
                            else:
                                # MiniMax yoksa sadece arama sonuçlarını göster
                                st.warning("⚠️ MiniMax ayarlanmamış. Sadece arama sonuçları gösteriliyor.")

                                for i, match in enumerate(matches, 1):
                                    metadata = match.get('metadata', {})
                                    score = match.get('score', 0)
                                    source = metadata.get('source', 'Bilinmeyen')
                                    chunk_idx = metadata.get('chunk_index', 0)

                                    st.markdown(f"**📄 Sonuç {i} — %{score*100:.1f} benzerlik**")
                                    st.caption(f"📖 Kaynak: {source} (Parça: {chunk_idx})")
                                    st.markdown(f"> {metadata.get('text', '')}")
                                    st.divider()

                    except Exception as e:
                        st.error(f"❌ Hata oluştu: {str(e)}")
                        logger.error(f"RAG hatası: {e}")

            elif not search_button and not query_text:
                st.info("👆 Yukarıdan sorunuzu sorun")

            # RAG açıklaması
            with st.expander("ℹ️ RAG Nedir?"):
                st.markdown("""
                **RAG (Retrieval Augmented Generation):**

                1. **Retrieval (Alma):** Sorunuzla ilgili kitap parçalarını Pinecone'dan buluyoruz
                2. **Augmentation (Zenginleştirme):** Bulunan parçaları soruyla birlikte LLM'e gönderiyoruz
                3. **Generation (Üretme):** MiniMax LLM'i bu bilgilere dayanarak Türkçe yanıt üretiyor

                Bu sayede LLM sadece kitaptaki gerçek bilgilere dayanarak yanıt verir.
                """)

        # İndeks bilgisi
        st.divider()
        st.subheader("📊 İndeks Bilgisi")

        try:
            pc = initialize_pinecone(index_name, dimension=DEFAULT_DIMENSION)
            index = pc.Index(index_name)
            stats = index.describe_index_stats()

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Toplam Vektör", stats.get('total_vector_count', 0))
            with col_s2:
                st.metric("İndeks Adı", index_name)
            with col_s3:
                st.metric("Dimension", stats.get('dimension', DEFAULT_DIMENSION))

        except Exception as e:
            st.warning(f"İndeks bilgisi alınamadı: {e}")

    # --- Bilgi Paneli ---
    st.divider()
    st.markdown("""
    ### 📌 Nasıl Kullanılır?

    1. **📤 Yükle sekmesinden** PDF veya EPUB yükleyin
    2. **🔍 Ara sekmesine** geçin
    3. **Sorunuzu Türkçe olarak** yazın
    4. **🤖 Soru Sor** butonuna tıklayın
    5. MiniMax LLM'i kitaptaki bilgilere dayanarak **Türkçe yanıt** üretir

    ### 🔧 Gereksinimler

    - **Ollama** yerel olarak çalışıyor (http://localhost:11434)
    - **MiniMax API**: `.env` dosyasında `MINIMAX_API_KEY` ve `MINIMAX_GROUP_ID`
    - **Pinecone API**: `.env` dosyasında `PINECONE_API_KEY`

    ### 📦 Bağımlılıklar

    ```
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)


if __name__ == "__main__":
    main()

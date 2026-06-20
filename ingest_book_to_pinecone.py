#!/usr/bin/env python3
"""
Kitap İçeriğini Pinecone Vektör Veritabanına Yükleme Betiği
=============================================================
Bu betik, PDF/EPUB formatındaki bir kitabı akıllıca parçalara ayırır,
MiniMax embedding modeli kullanarak vektörleştirir ve Pinecone'a yükler.

Yazar: Claude
Tarih: 2026-06-20
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# --- MiniMax & Pinecone ---
try:
    from openai import OpenAI
    MINIMAX_CLIENT = None  # OpenAI uyumlu istemci
except ImportError:
    MINIMAX_CLIENT = None

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def validate_environment() -> bool:
    """
    Gerekli ortam değişkenlerini ve kütüphaneleri doğrular.
    Returns:
        bool: Tüm gerekli bağımlılıklar mevcutsa True döner.
    """
    missing_deps = []

    # API Anahtarları
    if not os.getenv("MINIMAX_API_KEY"):
        missing_deps.append("MINIMAX_API_KEY")
    if not os.getenv("PINECONE_API_KEY"):
        missing_deps.append("PINECONE_API_KEY")

    # Kütüphaneler
    if PDF_LIBRARY is None:
        missing_deps.append("pdfplumber veya pypdf")
    if EPUB_LIBRARY is None:
        missing_deps.append("ebooklib")
    if not LANGCHAIN_AVAILABLE:
        missing_deps.append("langchain")
    if not PINECONE_AVAILABLE:
        missing_deps.append("pinecone-client")

    if missing_deps:
        logger.error("Eksik bağımlılıklar: %s", ", ".join(missing_deps))
        logger.error("Şu komutu çalıştırın: pip install pdfplumber ebooklib langchain pinecone-client openai python-dotenv")
        return False

    logger.info("✓ Tüm bağımlılıklar doğrulandı")
    return True


def clean_text(text: str) -> str:
    """
    Ham metinden gereksiz boşlukları, sayfa numaralarını ve diğer gürültüleri temizler.

    Args:
        text: Ham metin

    Returns:
        Temizlenmiş metin
    """
    if not text:
        return ""

    # Çoklu boşlukları tek boşluğa indirgeme
    text = re.sub(r'\s+', ' ', text)

    # Sayfa numaralarını temizle (genellikle "Page X", "Sayfa X", "X / Y" formatlarında)
    text = re.sub(r'(?:Page|Sayfa)\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*/\s*\d+', '', text)

    # Sayfa başındaki/sonundaki numaraları temizle (tek başına sayılar)
    text = re.sub(r'(?:^|\n)\s*\d+\s*(?:\n|$)', '\n', text)

    # Kısa çizgilerle ayrılmış satır sonlarını düzelt
    text = re.sub(r'-\s*\n\s*', '', text)

    # Gereksiz yeni satırları temizle (3+ ardışık yeni satırı 2'ye indir)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Baştaki ve sondaki boşlukları temizle
    text = text.strip()

    return text


# =============================================================================
# METİN ÇIKARMA FONKSİYONLARI
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    PDF dosyasından metin çıkarır.

    Args:
        pdf_path: PDF dosyasının yolu

    Returns:
        Çıkarılan metin
    """
    logger.info("PDF dosyası işleniyor: %s", pdf_path)

    if PDF_LIBRARY == "pdfplumber":
        return _extract_with_pdfplumber(pdf_path)
    elif PDF_LIBRARY == "pypdf":
        return _extract_with_pypdf(pdf_path)
    else:
        raise RuntimeError("Hiçbir PDF kütüphanesi mevcut değil")


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """pdfplumber ile PDF'ten metin çıkarır."""
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info("Toplam sayfa sayısı: %d", total_pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                cleaned = clean_text(text)
                if cleaned:
                    full_text.append(cleaned)

            if page_num % 25 == 0:
                logger.debug("İşlenen sayfa: %d/%d", page_num, total_pages)

    result = "\n\n".join(full_text)
    logger.info("PDF'ten %d karakter çıkarıldı", len(result))
    return result


def _extract_with_pypdf(pdf_path: str) -> str:
    """pypdf ile PDF'ten metin çıkarır."""
    full_text = []

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    logger.info("Toplam sayfa sayısı: %d", total_pages)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            cleaned = clean_text(text)
            if cleaned:
                full_text.append(cleaned)

        if page_num % 25 == 0:
            logger.debug("İşlenen sayfa: %d/%d", page_num, total_pages)

    result = "\n\n".join(full_text)
    logger.info("PDF'ten %d karakter çıkarıldı", len(result))
    return result


def extract_text_from_epub(epub_path: str) -> str:
    """
    EPUB dosyasından metin çıkarır.

    Args:
        epub_path: EPUB dosyasının yolu

    Returns:
        Çıkarılan metin
    """
    logger.info("EPUB dosyası işleniyor: %s", epub_path)

    if EPUB_LIBRARY is None:
        raise RuntimeError("ebooklib kütüphanesi mevcut değil")

    full_text = []

    book = epub.read_epub(epub_path)

    # EPUB içindeki tüm dokümanları işle
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # HTML içeriğinden metin çıkar
            content = item.get_content().decode('utf-8', errors='ignore')
            # Basit HTML tag temizleme
            text = re.sub(r'<[^>]+>', ' ', content)
            text = clean_text(text)
            if text:
                full_text.append(text)

    result = "\n\n".join(full_text)
    logger.info("EPUB'ten %d karakter çıkarıldı", len(result))
    return result


def extract_text(file_path: str) -> str:
    """
    Dosya uzantısına göre uygun metin çıkarma fonksiyonunu çağırır.

    Args:
        file_path: Kitap dosyasının yolu

    Returns:
        Çıkarılan metin
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".epub":
        return extract_text_from_epub(file_path)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {extension}. Lütfen PDF veya EPUB kullanın.")


# =============================================================================
# METİN PARÇALAMA (CHUNKING)
# =============================================================================

def chunk_text(text: str, chunk_size: int = 750, chunk_overlap: int = 75) -> List[str]:
    """
    Metni LangChain'in RecursiveCharacterTextSplitter kullanarak parçalara ayırır.

    Args:
        text: Parçalara ayrılacak metin
        chunk_size: Her parçanın maksimum karakter sayısı
        chunk_overlap: Parçalar arası örtüşme karakter sayısı

    Returns:
        Parça metinler listesi
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain mevcut değil, basit parçalama kullanılıyor")
        return _simple_chunk_text(text, chunk_size, chunk_overlap)

    logger.info("Metin parçalanıyor (chunk_size=%d, chunk_overlap=%d)", chunk_size, chunk_overlap)

    # RecursiveCharacterTextSplitter: Önce paragraf, sonra cümle, sonra karakter düzeyinde böler
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False
    )

    chunks = splitter.split_text(text)

    logger.info("Metin %d parçaya ayrıldı", len(chunks))

    # Parça boyutlarını logla
    if chunks:
        avg_len = sum(len(c) for c in chunks) / len(chunks)
        logger.debug("Ortalama parça boyutu: %.1f karakter", avg_len)

    return chunks


def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    LangChain olmadığında kullanılacak basit parçalama fonksiyonu.

    Args:
        text: Parçalara ayrılacak metin
        chunk_size: Her parçanın maksimum karakter sayısı
        chunk_overlap: Parçalar arası örtüşme karakter sayısı

    Returns:
        Parça metinler listesi
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap  # Örtüşme ile kaydır

    return chunks


# =============================================================================
# PINECONE İŞLEMLERİ
# =============================================================================

def initialize_pinecone(
    index_name: str,
    dimension: int = 1536,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
) -> Pinecone:
    """
    Pinecone istemcisini başlatır ve belirtilen isimde indeks yoksa oluşturur.

    Args:
        index_name: Oluşturulacak veya kullanılacak indeks adı
        dimension: Vektör boyutu (MiniMax embedding için 1536)
        metric: Benzerlik metriği ("cosine", "euclidean", "dotproduct")
        cloud: Bulut sağlayıcısı
        region: Bölge

    Returns:
        Pinecone istemcisi
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY ortam değişkeni bulunamadı")

    logger.info("Pinecone istemcisi başlatılıyor...")

    pc = Pinecone(api_key=api_key)

    # Mevcut indeksleri kontrol et
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info("İndeks '%s' bulunamadı, oluşturuluyor...", index_name)

        # ServerlessSpec ile indeks oluştur
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )

        # İndeks hazır olana kadar bekle
        while not pc.describe_index(index_name).status.ready:
            logger.debug("İndeks hazırlanıyor...")
            import time
            time.sleep(1)

        logger.info("✓ İndeks '%s' başarıyla oluşturuldu (dimension=%d, metric=%s)",
                   index_name, dimension, metric)
    else:
        logger.info("Mevcut indeks '%s' kullanılıyor", index_name)

    return pc


def get_pinecone_index(pc: Pinecone, index_name: str):
    """
    Pinecone indeksini döndürür.

    Args:
        pc: Pinecone istemcisi
        index_name: İndeks adı

    Returns:
        Pinecone indeks nesnesi
    """
    return pc.Index(index_name)


# =============================================================================
# MINIMAX EMBEDDING & YÜKLEME
# =============================================================================

# MiniMax OpenAI-uyumlu API ayarları
MINIMAX_BASE_URL = "https://api.minimax.io"
MINIMAX_EMBEDDING_MODEL = "embedding-model"  # MiniMax embedding modeli


def create_embeddings(texts: List[str], model: str = "embedding-model") -> List[List[float]]:
    """
    MiniMax API kullanarak metinler için embedding vektörleri oluşturur.
    MiniMax, OpenAI-uyumlu bir API sağlar, bu nedenle openai kütüphanesi kullanılır.

    Args:
        texts: Embedding'i alınacak metinler listesi
        model: MiniMax embedding modeli

    Returns:
        Embedding vektörleri listesi
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY ortam değişkeni bulunamadı")

    logger.info("MiniMax embedding modeli başlatılıyor: %s", model)

    # MiniMax, OpenAI uyumlu API sağlar
    client = OpenAI(
        api_key=api_key,
        base_url=MINIMAX_BASE_URL
    )

    embeddings = []
    total = len(texts)

    for i in range(0, total, 100):  # 100'erli paketler halinde API çağrısı
        batch = texts[i:i + 100]
        batch_num = (i // 100) + 1
        total_batches = (total + 99) // 100

        logger.debug("Embedding batch %d/%d (%d metin)", batch_num, total_batches, len(batch))

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # Response yapısını kontrol et (openai >= 1.0.0)
        if hasattr(response, 'data'):
            batch_embeddings = [item.embedding for item in response.data]
        else:
            # Eski API yapısı için geriye dönük uyumluluk
            batch_embeddings = [item["embedding"] for item in response["data"]]

        embeddings.extend(batch_embeddings)

    logger.info("✓ %d metin için embedding oluşturuldu", len(embeddings))
    return embeddings


def upsert_to_pinecone(
    index,
    vectors: List[Dict[str, Any]],
    batch_size: int = 100
) -> int:
    """
    Vektörleri Pinecone indeksine yükler (upsert).

    Args:
        index: Pinecone indeks nesnesi
        vectors: Yüklenecek vektörler listesi (id, values, metadata içerir)
        batch_size: Her batch'teki vektör sayısı

    Returns:
        Yüklenen toplam vektör sayısı
    """
    total_vectors = len(vectors)
    total_batches = (total_vectors + batch_size - 1) // batch_size
    uploaded_count = 0

    logger.info("%d vektör Pinecone'a yükleniyor (batch_size=%d)...", total_vectors, batch_size)

    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        try:
            # Pinecone v5.x API: upsert metodu
            index.upsert(vectors=batch)
            uploaded_count += len(batch)
            logger.info("  Batch %d/%d tamamlandı (%d/%d vektör)",
                        batch_num, total_batches, uploaded_count, total_vectors)
        except Exception as e:
            logger.error("  Batch %d/%d hatası: %s", batch_num, total_batches, str(e))
            raise

    logger.info("✓ %d vektör başarıyla Pinecone'a yüklendi", uploaded_count)
    return uploaded_count


# =============================================================================
# ANA İŞLEM AKIŞI
# =============================================================================

def process_book(
    book_path: str,
    index_name: str,
    source_name: Optional[str] = None,
    chunk_size: int = 750,
    chunk_overlap: int = 75,
    embedding_model: str = "embedding-model",  # MiniMax embedding modeli
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Kitap dosyasını işleyerek Pinecone'a yükler.

    Args:
        book_path: Kitap dosyasının yolu (PDF veya EPUB)
        index_name: Pinecone indeks adı
        source_name: Kaynak adı (meta veri için, varsayılan: dosya adı)
        chunk_size: Parça boyutu (karakter)
        chunk_overlap: Parça örtüşmesi (karakter)
        embedding_model: MiniMax embedding modeli
        batch_size: Yükleme batch boyutu

    Returns:
        İşlem sonucu özeti
    """
    result = {
        "success": False,
        "book_path": book_path,
        "index_name": index_name,
        "chunks_created": 0,
        "vectors_uploaded": 0,
        "error": None
    }

    try:
        # 1. Ortam doğrulama
        logger.info("=" * 60)
        logger.info("KİTAP İŞLEME BAŞLATILDY")
        logger.info("=" * 60)

        if not validate_environment():
            raise RuntimeError("Ortam doğrulaması başarısız")

        # 2. Metin çıkarma
        logger.info("-" * 40)
        logger.info("AŞAMA 1: Metin Çıkarma")
        logger.info("-" * 40)

        raw_text = extract_text(book_path)
        result["characters_extracted"] = len(raw_text)
        logger.info("Çıkarılan metin uzunluğu: %d karakter", len(raw_text))

        if not raw_text.strip():
            raise ValueError("Çıkarılan metin boş")

        # 3. Metin parçalama
        logger.info("-" * 40)
        logger.info("AŞAMA 2: Metin Parçalama")
        logger.info("-" * 40)

        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result["chunks_created"] = len(chunks)

        if not chunks:
            raise ValueError("Hiç parça oluşturulamadı")

        # 4. Pinecone başlatma
        logger.info("-" * 40)
        logger.info("AŞAMA 3: Pinecone Başlatma")
        logger.info("-" * 40)

        pc = initialize_pinecone(
            index_name=index_name,
            dimension=1536,  # MiniMax embedding modeli boyutu
            metric="cosine"
        )
        index = get_pinecone_index(pc, index_name)

        # 5. Embedding oluşturma
        logger.info("-" * 40)
        logger.info("AŞAMA 4: Embedding Oluşturma")
        logger.info("-" * 40)

        embeddings = create_embeddings(chunks, model=embedding_model)

        # 6. Pinecone'a yükleme
        logger.info("-" * 40)
        logger.info("AŞAMA 5: Pinecone'a Yükleme")
        logger.info("-" * 40)

        # Kaynak adını belirle
        if source_name is None:
            source_name = Path(book_path).name

        # Vektörleri hazırla
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{Path(book_path).stem}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,           # Orijinal ham metin
                    "source": source_name,    # Kaynak dosya adı
                    "chunk_index": i          # Parça indeksi
                }
            })

        result["vectors_uploaded"] = upsert_to_pinecone(index, vectors, batch_size=batch_size)
        result["success"] = True

        logger.info("=" * 60)
        logger.info("✓ İŞLEM BAŞARIYLA TAMAMLANDI")
        logger.info("=" * 60)
        logger.info("Özet:")
        logger.info("  - Dosya: %s", book_path)
        logger.info("  - Çıkarılan karakter: %d", result["characters_extracted"])
        logger.info("  - Oluşturulan parça: %d", result["chunks_created"])
        logger.info("  - Yüklenen vektör: %d", result["vectors_uploaded"])
        logger.info("  - Hedef indeks: %s", index_name)

    except Exception as e:
        result["error"] = str(e)
        logger.error("İŞLEM HATASI: %s", str(e))
        raise

    return result


# =============================================================================
# ANA FONKSİYON (CLI)
# =============================================================================

def main():
    """
    Komut satırı arayüzü.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Kitap (PDF/EPUB) içeriğini Pinecone vektör veritabanına yükler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python ingest_book_to_pinecone.py --book ./kitap.pdf --index-name my-book-index
  python ingest_book_to_pinecone.py --book ./kitap.epub --index-name epub-index --chunk-size 1000
  python ingest_book_to_pinecone.py --book ./kitap.pdf --index-name my-index --source "Özel Kitap Adı"

Ortam değişkenleri (.env dosyasında veya sistemde):
  MINIMAX_API_KEY: MiniMax API anahtarı
  PINECONE_API_KEY: Pinecone API anahtarı
        """
    )

    parser.add_argument(
        "--book", "-b",
        required=True,
        help="İşlenecek kitap dosyasının yolu (PDF veya EPUB)"
    )

    parser.add_argument(
        "--index-name", "-i",
        required=True,
        help="Pinecone indeks adı"
    )

    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Kaynak adı (meta veri için, varsayılan: dosya adı)"
    )

    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=750,
        help="Parça boyutu karakter cinsinden (varsayılan: 750)"
    )

    parser.add_argument(
        "--chunk-overlap", "-o",
        type=int,
        default=75,
        help="Parça örtüşmesi karakter cinsinden (varsayılan: 75)"
    )

    parser.add_argument(
        "--embedding-model", "-e",
        default="embedding-model",
        help="MiniMax embedding modeli (varsayılan: embedding-model)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Pinecone yükleme batch boyutu (varsayılan: 100)"
    )

    args = parser.parse_args()

    # Kitap dosyasının varlığını kontrol et
    if not Path(args.book).exists():
        logger.error("Dosya bulunamadı: %s", args.book)
        return 1

    try:
        result = process_book(
            book_path=args.book,
            index_name=args.index_name,
            source_name=args.source,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )

        if result["success"]:
            return 0
        else:
            return 1

    except Exception as e:
        logger.error("Beklenmeyen hata: %s", str(e))
        return 1


if __name__ == "__main__":
    exit(main())

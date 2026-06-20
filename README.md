# 📚 Pinecone Vector Database - Kitap İndeksleme ve Arama (RAG) Uygulaması

Bu proje, PDF veya EPUB formatındaki kitap/doküman içeriklerini akıllıca parçalara ayırıp, vektör veri tabanı **Pinecone** üzerinde saklayan ve **RAG (Retrieval Augmented Generation - Bilgi Geri Çağırmayla Zenginleştirilmiş Nesil)** yaklaşımıyla kitap içeriği hakkında Türkçe soru-cevap yapılabilmesini sağlayan modern bir yapay zeka uygulamasıdır.

Proje, hem pratik ve şık bir **Streamlit Arayüzü** (`app.py`) hem de toplu işlemler için tasarlanmış bir **CLI Betiği** (`ingest_book_to_pinecone.py`) sunmaktadır.

---

## 🚀 Özellikler

* **Doküman Desteği:** PDF ve EPUB formatındaki kitapları otomatik olarak ayrıştırır ve temizler.
* **Akıllı Metin Parçalama (Chunking):** LangChain'in `RecursiveCharacterTextSplitter` bileşeni kullanılarak anlam bütünlüğünü koruyacak şekilde metni parçalar.
* **Gelişmiş Vektör Veritabanı:** Sunucusuz (Serverless) **Pinecone** yapısını destekler, indeksleri otomatik oluşturur ve yönetir.
* **İki Farklı Model Desteği:**
  * **Ollama (Yerel):** Streamlit arayüzünde varsayılan olarak yerel `nomic-embed-text` modeli kullanılarak yerel olarak embedding oluşturulur.
  * **MiniMax API:** CLI arayüzünde MiniMax `embedding-model` kullanılarak yüksek performanslı bulut tabanlı embedding üretilir.
* **MiniMax LLM (RAG):** Pinecone üzerinden benzerlik araması (similarity search) ile tespit edilen bağlamlar, Anthropic uyumlu MiniMax API (`MiniMax-M2.7`) modeline gönderilerek doğrudan kitaba sadık Türkçe yanıtlar üretilir.
* **Gelişmiş Streamlit Arayüzü:** 
  * Kolay sürükle-bırak dosya yükleyici.
  * Adım adım işlem durum çubukları (Progress bar).
  * Benzerlik oranları, kaynak parçalar ve RAG detaylarını içeren interaktif analiz paneli.
  * Pinecone indeks istatistikleri ve genel durum takibi.

---

## 🛠️ Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/hakanhnt/vector_database_createv1.git
cd vector_database_createv1
```

### 2. Bağımlılıkları Yükleyin
Gerekli Python paketlerini yüklemek için:
```bash
pip install -r requirements.txt
```

### 3. Ollama (Yerel Embedding için)
Arayüzde yerel embedding oluşturabilmek için sisteminizde **Ollama**'nın yüklü ve çalışır durumda olması gerekir.
* Ollama'yı [ollama.com](https://ollama.com) adresinden indirin.
* Terminalde modeli indirin ve servisi başlatın:
```bash
ollama pull nomic-embed-text
```

---

## ⚙️ Yapılandırma (`.env` Dosyası)

Proje dizininde bir `.env` dosyası oluşturun ve aşağıdaki değişkenleri tanımlayın:

```ini
# Pinecone API Ayarları (https://app.pinecone.io/)
PINECONE_API_KEY=your_pinecone_api_key_here

# MiniMax API Ayarları (RAG ve CLI Embedding için)
MINIMAX_API_KEY=your_minimax_api_key_here
MINIMAX_GROUP_ID=your_minimax_group_id_here
```

---

## 🖥️ Kullanım

### A. Streamlit Arayüzünü Başlatma (Önerilen)
Web arayüzünü çalıştırmak için terminalde şu komutu çalıştırın:
```bash
streamlit run app.py
```
Açılan tarayıcı ekranında:
1. **📤 Yükle** sekmesinden bir PDF veya EPUB dosyası seçip **▶️ İşlemi Başlat** butonuna tıklayın.
2. İşlem bittikten sonra **🔍 Ara** sekmesine geçerek kitabın içeriği ile ilgili sorularınızı sorabilirsiniz.

### B. CLI Üzerinden Kitap İndeksleme
Streamlit arayüzü yerine doğrudan terminalden bir kitabı indeksleyip Pinecone'a yüklemek için:
```bash
python ingest_book_to_pinecone.py --book "/yol/kitap.pdf" --index-name "kitap-indeksi"
```

**Kullanılabilir CLI Parametreleri:**
* `--book` / `-b`: İşlenecek kitap dosyasının yolu (PDF veya EPUB - Zorunlu)
* `--index-name` / `-i`: Pinecone indeks adı (Zorunlu)
* `--chunk-size` / `-c`: Parçalama boyutu (Varsayılan: 750 karakter)
* `--chunk-overlap` / `-o`: Parçalar arası örtüşme boyutu (Varsayılan: 75 karakter)
* `--embedding-model` / `-e`: Kullanılacak MiniMax embedding modeli (Varsayılan: `embedding-model`)

---

## 📋 Proje Yapısı

```
├── app.py                      # Streamlit web arayüzü ve RAG sorgu motoru
├── ingest_book_to_pinecone.py  # CLI tabanlı toplu kitap indeksleme betiği
├── requirements.txt            # Proje bağımlılık listesi
├── .gitignore                  # Git dışı bırakılacak dosyalar listesi
├── .env.example                # Çevre değişkenleri şablonu
└── README.md                   # Proje açıklama dokümanı (Bu dosya)
```

---

## ⚠️ Önemli Notlar

1. **Güvenlik:** `.env` dosyanız API anahtarlarını barındırır. `.gitignore` dosyasının bu dosyayı kapsadığından emin olun ve asla GitHub'a push etmeyin.
2. **Pinecone Ücretsiz Katmanı:** Pinecone'un ücretsiz katmanını kullanıyorsanız, tek seferde bir adet aktif indeks oluşturabileceğinizi unutmayın. Farklı kitaplar için aynı indeksi kullanabilir veya eski indeksi Pinecone konsolundan temizleyebilirsiniz.

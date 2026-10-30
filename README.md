# Generative AI

Materi pembelajaran Generative AI, dikelompokkan berdasarkan fase:

- **Phase 1 - Python Core for AI**: Modul 1-5, dasar Python, struktur data, OOP, file/API, dan Python untuk data.
- **Phase 2 - LLM APIs & Prompting**: Modul 6-8, penggunaan API LLM, prompt engineering, embeddings, dan semantic search.
- **Phase 3 - Gen AI Libraries**: Modul 9-14, LangChain, LlamaIndex, vector databases, Hugging Face Transformers, LangGraph, dan CrewAI.

## Instalasi dependensi

Gunakan Python virtual environment. Dependensi dipisah per fase atau modul karena beberapa library AI memiliki kebutuhan versi yang berbeda.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install hanya requirement yang dibutuhkan:

```bash
pip install -r "Phase 1 - Python Core for AI/requirements.txt"
pip install -r "Phase 2 - LLM APIs & Prompting/requirements.txt"
pip install -r "Phase 3 - Gen AI Libraries /Module 11 - Vector Databases/requirements.txt"
```

Untuk modul Phase 3 lainnya, pilih `requirements.txt` di folder modul terkait. Jika menemui konflik versi, buat virtual environment terpisah untuk modul tersebut.

## API keys dan `.gitignore`

File `.gitignore` di root mengabaikan `.env`, file `.env.*`, virtual environment, cache Python, file sistem seperti `.DS_Store`, dan output model `lora_rag_qa/`. **Jangan hapus aturan ini dan jangan commit file `.env` atau API keys.**

1. Salin `.env.example` menjadi `.env` di root repository.
2. Isi `.env` dengan API keys milikmu melalui editor lokal. Jangan tempel key ke chat atau commit.
3. Jika key pernah terekspos, revoke key tersebut di dashboard providernya dan gunakan key baru.

`.env.example` hanya berisi nama variabel dan placeholder kosong; file ini aman untuk di-commit sebagai panduan. Isi `.env` yang sebenarnya tetap lokal dan diabaikan Git.

Contoh menjalankan materi Modul 11:

```bash
python "Phase 3 - Gen AI Libraries /Module 11 - Vector Databases/11.3 Pinecone.py"
```

Script tersebut membutuhkan `OPENROUTER_API_KEY` untuk membuat embeddings. `PINECONE_API_KEY` diperlukan untuk mengakses Pinecone.

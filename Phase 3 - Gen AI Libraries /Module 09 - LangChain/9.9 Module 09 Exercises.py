# 9.9 Module 09 Exercises (Halaman 114-115)
import hashlib
import math
import os
import re
from collections.abc import Callable
from typing import Protocol

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


# 1. MultiQueryRetriever manual
def multi_query_retrieve(
    query: str,
    retriever: Retriever,
    llm_paraphrase_fn: Callable[[str], list[str]],
) -> list[Document]:
    paraphrases = llm_paraphrase_fn(query)
    seen_hashes: set[str] = set()
    unique_docs: list[Document] = []
    for q in [query] + paraphrases:
        docs = retriever.invoke(q)
        for d in docs:
            h = hashlib.sha256(d.page_content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_docs.append(d)
    return unique_docs

# 2. Map-Reduce Summarizer
def map_reduce_summarize(
    chunks: list[str],
    summarize_fn: Callable[[str], str],
    final_summarize_fn: Callable[[str], str],
) -> str:
    chunk_summaries = [summarize_fn(c) for c in chunks]
    combined = "\n".join(chunk_summaries)
    return final_summarize_fn(combined)

# 3. Groundedness scorer
def score_groundedness(
    answer: str,
    context: str,
    judge_llm_fn: Callable[[str], float],
) -> float:
    prompt = f"Context: {context}\nAnswer: {answer}\nBased on the context, score from 0 to 10 how accurate and grounded the answer is. Return ONLY the number without any other text or explanation."
    score = judge_llm_fn(prompt)
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise ValueError(f"Groundedness score must be a finite number from 0 to 10; got {score!r}.")
    return score


def parse_groundedness_score(response: str) -> float:
    match = re.fullmatch(r"\s*(10(?:\.0+)?|[0-9](?:\.\d+)?)\s*(?:/10)?\s*", response)
    if match is None:
        raise ValueError(f"Expected a groundedness score from 0 to 10, got {response!r}.")
    return float(match.group(1))

if __name__ == "__main__":
    print("=== MULAI IMPLEMENTASI LATIHAN MODUL 09 ===\n")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the environment or project .env file."
        )

    # Setup LLM menggunakan OpenRouter seperti file sebelumnya
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-super-vl:free",
        api_key=api_key,
    )
    parser = StrOutputParser()

    # ------------------------------------------------------------------
    # LATIHAN 1: Multi-Query Retriever
    # ------------------------------------------------------------------
    print("1. Menjalankan Multi-Query Retriever (Pencarian dari banyak sudut)...")
    def mock_paraphrase(query: str) -> list[str]:
        # Simulasi fungsi LLM untuk merangkai ulang pertanyaan
        return [f"Apa itu {query}?", f"Tolong jelaskan tentang {query}"]

    class MockRetriever:
        def invoke(self, query):
            # Simulasi database vector yang mengembalikan dokumen terkait
            return [Document(page_content=f"Informasi didapat untuk: {query}")]

    docs = multi_query_retrieve("Kecerdasan Buatan", MockRetriever(), mock_paraphrase)
    print("   Dokumen unik yang berhasil ditarik:")
    for d in docs:
        print(f"   - {d.page_content}")
    print()

    # ------------------------------------------------------------------
    # LATIHAN 2: Map-Reduce Summarizer
    # ------------------------------------------------------------------
    print("2. Menjalankan Map-Reduce Summarizer (Meringkas bertahap)...")
    def llm_summarize(text: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "Buat ringkasan pendek 1 kalimat dari teks ini: {text}"
        )
        chain = prompt | llm | parser
        return chain.invoke({"text": text})

    def llm_final_summarize(text: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "Gabungkan ringkasan berikut menjadi satu kalimat kesimpulan utama yang natural:\n{text}"
        )
        chain = prompt | llm | parser
        return chain.invoke({"text": text})

    # Misalkan ini adalah dokumen PDF panjang yang sudah dipotong-potong (chunk)
    chunks = [
        "Bagian 1: AI (Kecerdasan Buatan) diprediksi akan mengubah banyak sektor industri ke depannya.",
        "Bagian 2: Banyak perusahaan e-commerce kini beralih menggunakan AI untuk melayani pelanggan.",
        "Bagian 3: Namun ada juga tantangan etika dan privasi data yang perlu diawasi dengan ketat dalam AI."
    ]
    # Catatan: Ini memanggil API 4 kali (3x map, 1x reduce), mungkin butuh bbrp detik
    print("   [Menunggu respon dari LLM (Mungkin butuh beberapa detik)...]")
    final_summary = map_reduce_summarize(chunks, llm_summarize, llm_final_summarize)
    print("   Kesimpulan Akhir:")
    print(f"   => {final_summary.strip()}\n")

    # ------------------------------------------------------------------
    # LATIHAN 3: Groundedness Scorer (LLM sebagai Juri)
    # ------------------------------------------------------------------
    print("3. Menjalankan Groundedness Scorer (Menilai halusinasi LLM)...")
    def llm_judge(prompt_text: str) -> float:
        prompt = ChatPromptTemplate.from_template("{text}")
        chain = prompt | llm | parser
        response = chain.invoke({"text": prompt_text})
        return parse_groundedness_score(response)

    sumber_konteks = "LangChain adalah framework open-source yang dirilis oleh Harrison Chase pada Oktober 2022."

    # Skenario A: LLM menjawab dengan benar berdasarkan konteks
    jawaban_fakta = "Framework LangChain dibuat oleh Harrison Chase pada bulan Oktober tahun 2022."
    skor_a = score_groundedness(jawaban_fakta, sumber_konteks, llm_judge)
    print(f"   [Fakta] Jawaban: '{jawaban_fakta}'")
    print(f"           Skor Akurasi : {skor_a} / 10\n")

    # Skenario B: LLM halusinasi / mengarang cerita
    jawaban_halusinasi = "LangChain dikembangkan diam-diam oleh Elon Musk di markas SpaceX pada tahun 2010."
    skor_b = score_groundedness(jawaban_halusinasi, sumber_konteks, llm_judge)
    print(f"   [Ngawur] Jawaban: '{jawaban_halusinasi}'")
    print(f"            Skor Akurasi : {skor_b} / 10\n")

    print("Module 09 Bismillah Sukses TA.")

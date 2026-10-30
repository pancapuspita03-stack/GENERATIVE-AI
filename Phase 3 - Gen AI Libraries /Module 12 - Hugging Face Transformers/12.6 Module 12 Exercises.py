"""Module 12 exercises: generation benchmark, token budget, NER, and LoRA."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)


# ============================================================
# EXERCISE 1: Bandingkan dua model text-generation
# ============================================================

PROMPTS = [
    "Explain retrieval augmented generation in one sentence.",
    "What is a vector database?",
    "Define machine learning.",
    "Why are embeddings useful?",
    "What does fine tuning change?",
    "Explain a Python list.",
    "What is a tokenizer?",
    "Name one benefit of local models.",
    "What is an API?",
    "Explain semantic search.",
]


@dataclass(frozen=True)
class BenchmarkRow:
    model_name: str
    prompt_index: int
    latency_ms: float
    output_length: int


def run_generation_benchmark() -> list[BenchmarkRow]:
    model_names = (
        os.environ.get("HF_BENCHMARK_MODEL_A", "distilgpt2"),
        os.environ.get("HF_BENCHMARK_MODEL_B", "openai-community/gpt2"),
    )
    if model_names[0] == model_names[1]:
        raise ValueError("Pilih dua nama model yang berbeda.")

    rows: list[BenchmarkRow] = []

    for model_name in model_names:
        print(f"Memuat model: {model_name}")
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1,
            model_kwargs={"use_safetensors": True},
        )

        for prompt_index, prompt in enumerate(PROMPTS, start=1):
            started = time.perf_counter()
            result = generator(
                prompt,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            latency_ms = (time.perf_counter() - started) * 1000

            generated_text = result[0]["generated_text"]
            output_text = (
                generated_text[len(prompt):]
                if generated_text.startswith(prompt)
                else generated_text
            )
            rows.append(
                BenchmarkRow(
                    model_name=model_name,
                    prompt_index=prompt_index,
                    latency_ms=latency_ms,
                    output_length=len(output_text),
                )
            )

    return rows


def print_benchmark(rows: list[BenchmarkRow]) -> None:
    print(f"{'Model':30} {'Prompt':>6} {'Latency ms':>12} {'Output chars':>13}")
    print("-" * 67)
    for row in rows:
        print(
            f"{row.model_name:30} {row.prompt_index:>6} "
            f"{row.latency_ms:>12.2f} {row.output_length:>13}"
        )

    print("\nRata-rata latensi:")
    for model_name in dict.fromkeys(row.model_name for row in rows):
        latencies = [
            row.latency_ms for row in rows if row.model_name == model_name
        ]
        print(f"- {model_name}: {mean(latencies):.2f} ms")


# ============================================================
# EXERCISE 2: Filter teks berdasarkan batas token
# ============================================================

@dataclass(frozen=True)
class AcceptedText:
    text: str
    token_count: int


def filter_texts_by_token_budget(
    model_name: str,
    token_limit: int,
    texts: list[str],
) -> list[AcceptedText]:
    if token_limit < 1:
        raise ValueError("token_limit harus lebih besar dari 0.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    accepted: list[AcceptedText] = []

    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Semua item texts harus berupa string.")

        token_count = len(
            tokenizer.encode(text, add_special_tokens=True)
        )
        if token_count <= token_limit:
            accepted.append(AcceptedText(text, token_count))

    return accepted


# ============================================================
# EXERCISE 3: Ekstrak PERSON, ORG, dan LOC dari kumpulan teks
# ============================================================

ENTITY_TYPES = ("PERSON", "ORG", "LOC")


def extract_entities(
    texts: list[str],
    model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
) -> dict[str, list[str]]:
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts harus berisi string yang tidak kosong.")

    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=-1,
    )

    entities: dict[str, set[str]] = {entity_type: set() for entity_type in ENTITY_TYPES}

    for result in ner(texts):
        for entity in result:
            entity_type = str(entity.get("entity_group", "")).upper()
            entity_type = {"PER": "PERSON"}.get(entity_type, entity_type)
            value = str(entity.get("word", "")).strip()

            if entity_type in entities and value:
                entities[entity_type].add(value)

    return {
        entity_type: sorted(entities[entity_type])
        for entity_type in ENTITY_TYPES
    }


# ============================================================
# EXERCISE 4: Fine-tuning LoRA untuk 20 pasangan tanya-jawab
# ============================================================

QA_PAIRS = [
    ("What does RAG stand for?", "RAG stands for Retrieval-Augmented Generation."),
    ("What does RAG retrieve?", "RAG retrieves relevant external context."),
    ("When does RAG retrieve knowledge?", "RAG retrieves knowledge at inference time."),
    ("What do embeddings represent?", "Embeddings represent text as numerical vectors."),
    ("What is a vector database?", "A vector database stores embeddings for similarity search."),
    ("What is semantic search?", "Semantic search finds text by meaning rather than exact words."),
    ("What is chunking?", "Chunking splits a long document into smaller passages."),
    ("Why use chunking?", "Chunking makes relevant passages easier to retrieve."),
    ("What is a retriever?", "A retriever selects relevant documents for a query."),
    ("What is a generator in RAG?", "The generator writes an answer using retrieved context."),
    ("What is groundedness?", "Groundedness measures whether an answer is supported by context."),
    ("What is hallucination?", "A hallucination is an unsupported or fabricated answer."),
    ("What is fine-tuning?", "Fine-tuning trains a pretrained model on task-specific examples."),
    ("Does RAG change model weights?", "RAG does not change model weights."),
    ("What does an index organize?", "An index organizes data for efficient retrieval."),
    ("What is cosine similarity?", "Cosine similarity compares the direction of vectors."),
    ("Why cite sources?", "Citations show which passages support an answer."),
    ("What is a prompt?", "A prompt is input provided to a language model."),
    ("What does LoRA train?", "LoRA trains a small set of low-rank adapter parameters."),
    ("What is a QA pair?", "A QA pair contains one question and its answer."),
]


def build_training_dataset(tokenizer: Any) -> Dataset:
    prompts = [f"Question: {question}\nAnswer:" for question, _ in QA_PAIRS]
    answers = [f" {answer}" for _, answer in QA_PAIRS]
    dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        texts = [
            prompt + answer
            for prompt, answer in zip(batch["prompt"], batch["answer"])
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        prompt_lengths = [
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=128,
                )["input_ids"]
            )
            for prompt in batch["prompt"]
        ]
        encoded["labels"] = [
            [
                token_id
                if mask == 1 and position >= prompt_length
                else -100
                for position, (token_id, mask) in enumerate(
                    zip(input_ids, attention_mask)
                )
            ]
            for input_ids, attention_mask, prompt_length in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                prompt_lengths,
            )
        ]
        return encoded

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["prompt", "answer"],
    )


def fine_tune_and_verify(
    model_name: str = "distilgpt2",
    output_dir: str | Path = "lora_rag_qa",
    epochs: int = 3,
) -> list[dict[str, str | bool]]:
    if epochs < 1:
        raise ValueError("epochs harus lebih besar dari 0.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=build_training_dataset(tokenizer),
    )
    trainer.train()

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Muat ulang adapter untuk memastikan hasil pelatihan bisa digunakan.
    verification_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    verified_model = PeftModel.from_pretrained(base_model, output_dir)
    verified_model.eval()

    results: list[dict[str, str | bool]] = []
    for question, expected in QA_PAIRS:
        inputs = verification_tokenizer(
            f"Question: {question}\nAnswer:",
            return_tensors="pt",
        )
        with torch.no_grad():
            output = verified_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=verification_tokenizer.eos_token_id,
            )

        generated_tokens = output[0, inputs["input_ids"].shape[1] :]
        answer = verification_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        results.append(
            {
                "question": question,
                "expected": expected,
                "answer": answer,
                "verified": expected.casefold() in answer.casefold(),
            }
        )

    return results


# ============================================================
# Jalankan dan tampilkan empat output
# ============================================================

def main() -> None:
    print("\n========== OUTPUT 1: GENERATION BENCHMARK ==========")
    benchmark_rows = run_generation_benchmark()
    print_benchmark(benchmark_rows)

    print("\n========== OUTPUT 2: TOKEN BUDGET ==========")
    sample_texts = [
        "Short text.",
        "This is a longer example " * 20,
    ]
    accepted = filter_texts_by_token_budget(
        model_name="gpt2",
        token_limit=32,
        texts=sample_texts,
    )
    for item in accepted:
        print(f"{item.token_count} tokens: {item.text!r}")
    print(f"Accepted: {len(accepted)} of {len(sample_texts)} texts")

    print("\n========== OUTPUT 3: NAMED ENTITY RECOGNITION ==========")
    corpus = [
        "Anthropic was founded in San Francisco by Dario Amodei.",
        "Microsoft is headquartered in Redmond.",
    ]
    entities = extract_entities(corpus)
    for entity_type, values in entities.items():
        print(f"{entity_type}: {values}")

    print("\n========== OUTPUT 4: LoRA FINE-TUNING & VERIFICATION ==========")
    output_dir = Path("lora_rag_qa")
    results = fine_tune_and_verify(output_dir=output_dir)

    for item in results:
        status = "PASS" if item["verified"] else "FAIL"
        print(f"[{status}] {item['question']}\n  {item['answer']}")

    verified_count = sum(bool(item["verified"]) for item in results)
    print(f"Verified {verified_count}/{len(results)} answers.")


if __name__ == "__main__":
    main()
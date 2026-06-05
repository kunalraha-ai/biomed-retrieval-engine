# Biomed Retrieval Engine (BGE-M3)

High-precision semantic search microservice optimized for medical and regulatory text.

## 🚀 The Problem-
Standard dense retrievers (like OpenAI `text-embedding-3-small`) often fail on **high-overlap medical terminology**.
* **Example:** Confusing a prodrug (**Prednisone**) with its active metabolite (**Prednisolone**).
* **Consequence:** Inaccurate retrieval for clinical decision support or FDA regulatory submissions.

## ⚡ The Solution-
This microservice uses **BGE-M3** (BAAI General Embedding), utilizing a hybrid dense + sparse retrieval method to capture:
1.  **Semantic Meaning:** Understanding "hepatic failure" context.
2.  **Lexical Nuance:** Distinguishing "Predni-SONE" from "Predni-SOLONE".

## 🛠 Tech Stack-
* **Model:** BGE-M3 (Quantized for CPU Latency < 90ms)
* **Vector DB:** ChromaDB
* **API:** FastAPI
* **Container:** Docker

## 📌 Usage
The engine is designed to run locally (Air-gapped) for HIPAA compliance.

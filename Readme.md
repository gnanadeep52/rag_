# RAG Pipeline with Evaluation — LangChain + ChromaDB + Gemini

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with LangChain, HuggingFace embeddings, ChromaDB vector store, and Google Gemini — including a full **RAG evaluation suite** measuring context precision, recall, faithfulness, and answer relevancy.

> **Skills demonstrated:** RAG architecture · Vector stores · Embedding models · LLM orchestration · Evaluation pipelines · LangChain

---

## Problem

Standard LLMs hallucinate because they rely only on pre-trained knowledge. This pipeline solves that by retrieving relevant documents at query time and grounding every answer in real context — then **measures** how well it works with 4 evaluation metrics.

---

## Architecture

```
Document / URL Source
        ↓
PlaywrightURLLoader          ← Load live webpage content
        ↓
RecursiveCharacterTextSplitter  ← Chunk into 500-token segments (50-token overlap)
        ↓
HuggingFace Embeddings       ← sentence-transformers/all-MiniLM-L6-v2
        ↓
ChromaDB Vector Store        ← Cosine similarity index (persisted locally)
        ↓
Retriever (top-k = 3)        ← Semantic search over stored embeddings
        ↓
ChatPromptTemplate + Gemini  ← Context-grounded generation (gemma-3-1b-it)
        ↓
RAG Evaluation Pipeline      ← Scored against ground-truth JSONL dataset
```

---

## Evaluation Results

Evaluated against a domain-specific ground-truth JSONL dataset:

| Metric             | Score  | What it measures |
|--------------------|--------|------------------|
| Context Precision  | 0.8333 | How relevant retrieved chunks are to the question |
| Context Recall     | 0.6000 | How much of the needed information was retrieved |
| Faithfulness       | 0.8300 | Whether the answer is grounded in context (no hallucination) |
| Answer Relevancy   | 0.6970 | How on-topic the generated answer is |

---

## Tech Stack

| Layer             | Technology |
|-------------------|------------|
| Document Loading  | LangChain `PlaywrightURLLoader` |
| Text Splitting    | `RecursiveCharacterTextSplitter` (chunk=500, overlap=50) |
| Embeddings        | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store      | ChromaDB (local persistence, cosine similarity) |
| LLM               | Google Gemini `gemma-3-1b-it` via `langchain-google-genai` |
| Evaluation        | Custom RAGAS-style pipeline on JSONL ground truth |
| Language          | Python 3.10+ |

---

## Project Structure

```
rag_/
├── main.py               # Full RAG pipeline: load → chunk → embed → retrieve → generate
├── rag_evaluation.py     # Evaluation: context precision, recall, faithfulness, relevancy
├── Dataset.JSONL         # Ground-truth QA pairs for evaluation
├── requirements          # Python dependencies
├── chroma_langchain_db/  # Persisted ChromaDB vector store
└── Screenshots/          # Architecture and concept diagrams
```

---

## Setup & Run

```bash
# 1. Clone and install
git clone https://github.com/gnanadeep52/rag_.git
cd rag_
pip install -r requirements

# 2. Create .env file
echo "HUGGINGFACEHUB_API_TOKEN=your_hf_token" >> .env
echo "google_api_key=your_google_api_key" >> .env

# 3. Run
python main.py
```

The pipeline loads and chunks the document, builds the ChromaDB vector store, runs a sample RAG query, and prints all evaluation scores.

---

## Key Design Decisions

**ChromaDB over hosted vector store** — local persistence makes it fast to iterate without cloud infrastructure; same architecture ports directly to OpenSearch or Pinecone in production.

**HuggingFace embeddings over OpenAI** — demonstrates provider-agnostic design; any embedding model can be swapped via the LangChain interface.

**Evaluation-first approach** — RAG without measurement is incomplete. The 4-metric evaluation mirrors production workflows used with AWS SageMaker Model Monitor and CloudWatch.

---

## Related Production Work

This project reflects the same patterns used at scale:
- **AWS Bedrock + Amazon OpenSearch** (production equivalent of ChromaDB + HuggingFace)
- **SageMaker endpoints** (production equivalent of local Gemini calls)
- **CloudWatch + SageMaker Model Monitor** for real-time drift detection and alerting

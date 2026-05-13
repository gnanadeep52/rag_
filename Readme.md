# RAG Pipeline with Evaluation — LangChain + ChromaDB + Gemini

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with LangChain, HuggingFace embeddings, ChromaDB vector store, and Google Gemini as the LLM — including a full **RAG evaluation suite** measuring context precision, recall, faithfulness, and answer relevancy.

> Skills demonstrated: RAG architecture, vector stores, embedding models, LLM orchestration, evaluation pipelines.

## Problem

Standard LLMs hallucinate because they rely only on pre-trained knowledge. This pipeline fixes that by retrieving relevant documents at query time and grounding every answer in real context, then **measuring** quality with 4 RAG metrics.

## Architecture

```text
Document / URL Source
        ↓
PlaywrightURLLoader          ← Load webpage content
        ↓
RecursiveCharacterTextSplitter  ← Chunk into 500-token segments (50 overlap)
        ↓
HuggingFace Embeddings       ← sentence-transformers/all-MiniLM-L6-v2
        ↓
ChromaDB Vector Store        ← Cosine similarity index (persisted locally)
        ↓
Retriever (top-k = 3)        ← Semantic search over embeddings
        ↓
ChatPromptTemplate + Gemini  ← Context-grounded generation (gemma-3-1b-it)
        ↓
RAG Evaluation Pipeline      ← Scored against ground-truth JSONL dataset
```

## Evaluation Results

| Metric            | Score  |
|-------------------|--------|
| Context Precision | 0.8333 |
| Context Recall    | 0.6000 |
| Faithfulness      | 0.8300 |
| Answer Relevancy  | 0.6970 |

## Tech Stack

- LangChain (loader, splitter, prompt)
- ChromaDB vector store (cosine similarity)
- HuggingFace embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Google Gemini: `gemma-3-1b-it` via `langchain-google-genai`
- Python 3.10+

## Project Structure

```text
rag_/
├── main.py               # RAG pipeline: load → chunk → embed → retrieve → generate
├── rag_evaluation.py     # Evaluation: precision, recall, faithfulness, relevancy
├── Dataset.JSONL         # Ground-truth QA pairs
├── requirements          # Python dependencies
├── chroma_langchain_db/  # Persisted Chroma vector store
└── Screenshots/          # Diagrams and notes
```

## Setup & Run

```bash
git clone https://github.com/gnanadeep52/rag_.git
cd rag_
pip install -r requirements

# .env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
google_api_key=your_google_api_key

python main.py
```

The script builds the vector store, runs a sample question, then executes full RAG evaluation and prints the scores.

## Design Notes

- Local ChromaDB → easy to port to OpenSearch/Pinecone in production.
- Provider-agnostic embeddings via HuggingFace.
- Evaluation-first mindset, similar to RAGAS-style production setups.

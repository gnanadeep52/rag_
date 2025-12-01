import os
from dotenv import load_dotenv

# RAG imports
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Evaluation
from rag_evaluation import evaluate_rag
from pprint import pprint


# ======================================
# 0. Load Environment Variables
# ======================================

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("google_api_key")

if not hf_token:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

if not google_api_key:
    raise ValueError("Missing google_api_key in .env")



# ======================================
# 1. LOAD DOCUMENT FROM URL
# ======================================

loader = PlaywrightURLLoader(
    urls=["https://www.geeksforgeeks.org/deep-learning/what-is-fine-tuning/"]
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")


# ======================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# ======================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.create_documents([docs[0].page_content])

print("Total Chunks:", len(chunks))
print("Preview Chunk:\n", chunks[0].page_content[:200])


# ======================================
# 3. EMBEDDINGS
# ======================================



embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_token
)

# (Optional) embed once for debugging
sample_vector = embeddings.embed_query("test embedding")
print("Embedding vector length:", len(sample_vector))


# ======================================
# 4. VECTORSTORE (CHROMA)
# ======================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="example_collection",
    persist_directory="./chroma_langchain_db",
    collection_metadata={"hnsw:space": "cosine"},
)


retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ======================================
# 5. LLM (GOOGLE GENAI)
# ======================================

llm = (ChatGoogleGenerativeAI(
    model="gemma-3-1b-it",
    api_key=google_api_key,
    temperature=0.2
))


# ======================================
# 6. RAG CHAIN (Prompt + Retriever + LLM)
# ======================================

prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't know".

<context>
{context}
</context>

Question: {question}
Answer:
""")

rag_chain = (
    {
        "context": lambda x: [d.page_content for d in retriever.invoke(x["question"])],
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
)



# ======================================
# 7. TEST THE RAG PIPELINE
# ======================================

test_q = "What are the advantages of fine-tuning?"
response = rag_chain.invoke({"question": test_q})


print("\n===== SAMPLE RAG OUTPUT =====\n")
print(response.content)


# ======================================
# 8. RUN FULL RAG EVALUATION
# ======================================

print("\n\n===== RUNNING FULL RAG EVALUATION =====\n")

results = evaluate_rag(
    rag_chain=rag_chain,
    retriever=retriever,
    ground_truth_path="Dataset.jsonl",

)

pprint(results)

print("\n===== EVALUATION DONE =====")

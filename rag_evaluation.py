# rag_evaluation.py

import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


# ----------------------------------------------------
# Load Ground Truth JSONL
# ----------------------------------------------------
def load_ground_truth(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "question": item["question"],
                "answer": item["answer"],     # ground-truth
            })
    return data


# ----------------------------------------------------
# Build Ragas Dataset
# ----------------------------------------------------
def build_ragas_dataset(qa_list, rag_chain, retriever):

    questions = []
    ground_truth = []
    generated_answers = []
    contexts = []

    for item in qa_list:
        q = item["question"]
        gt = item["answer"]

        # 1. Retrieve context
        docs = retriever.invoke(q)
        ctx = [doc.page_content for doc in docs]

        # 2. RAG Chain answer
        response = rag_chain.invoke({"question": q})

        if hasattr(response, "content"):
            gen_ans = response.content
        elif isinstance(response, dict) and "answer" in response:
            gen_ans = response["answer"]
        else:
            gen_ans = str(response)

        questions.append(q)
        ground_truth.append(gt)
        generated_answers.append(gen_ans)
        contexts.append(ctx)

    # MATCHES RagAS Example Format
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": generated_answers,        # model output
        "contexts": contexts,
        "ground_truth": ground_truth        # true answer
    })

    return dataset


# ----------------------------------------------------
# Evaluation Function
# ----------------------------------------------------
def evaluate_rag(rag_chain, retriever, ground_truth_path):
    print("Loading dataset...")
    qa_list = load_ground_truth(ground_truth_path)

    print("Building Ragas dataset...")
    dataset = build_ragas_dataset(qa_list, rag_chain, retriever)

    print("Setting up evaluation LLM (OpenAI)...")
    evaluator_llm = LangchainLLMWrapper(
       ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0
    )
    )

    print("Running Ragas evaluation...")

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    results = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm    
    )
    return results

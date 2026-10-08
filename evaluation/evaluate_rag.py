import os
import uuid
import requests

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset
from langsmith import Client


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# Make sure Ragas evaluation traces are sent to LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "enterprise-rag-baseline-v1"


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "http://localhost:8000"

DATASET_NAME = "enterprise_ai_golden_dataset"


# ============================================================
# RAGAS JUDGE MODEL
# ============================================================

judge_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ============================================================
# RUN RAG
# ============================================================

def run_rag(question: str):

    params = {
        "query": question,
        "session_id": str(uuid.uuid4()),
        "retrieval_limit": 10,
        "top_k": 5
    }

    response = requests.get(
        f"{BACKEND_URL}/search/hybrid",
        params=params,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    contexts = []

    for result in data.get("results", []):

        if isinstance(result, dict):

            content = result.get("content", "")

            if content:
                contexts.append(content)

        elif isinstance(result, str):

            contexts.append(result)

    return {
        "question": question,
        "answer": data.get("answer", ""),
        "contexts": contexts
    }


# ============================================================
# LANGSMITH CLIENT
# ============================================================

client = Client()


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

examples = list(
    client.list_examples(
        dataset_name=DATASET_NAME
    )
)


# ============================================================
# PREPARE EVALUATION DATA
# ============================================================

questions = []
answers = []
contexts = []
ground_truths = []


print("\n========================================")
print("STARTING RAGAS + LANGSMITH EVALUATION")
print("========================================\n")


for index, example in enumerate(examples, start=1):

    question = example.inputs.get(
        "question",
        ""
    )

    ground_truth = example.outputs.get(
        "ground_truth",
        ""
    )

    print(
        f"[{index}/{len(examples)}] "
        f"{question}"
    )

    result = run_rag(question)

    questions.append(
        question
    )

    answers.append(
        result["answer"]
    )

    contexts.append(
        result["contexts"]
    )

    ground_truths.append(
        ground_truth
    )


# ============================================================
# CREATE RAGAS DATASET
# ============================================================

evaluation_dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)


# ============================================================
# RUN RAGAS
# ============================================================

print("\n========================================")
print("RUNNING RAGAS METRICS")
print("========================================\n")


result = evaluate(
    dataset=evaluation_dataset,

    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],

    llm=judge_model,

    embeddings=embedding_model
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("RAGAS BASELINE RESULTS")
print("========================================\n")

print(result)


# ============================================================
# DISPLAY INDIVIDUAL RESULTS
# ============================================================

print("\n========================================")
print("INDIVIDUAL RESULTS")
print("========================================\n")

try:

    results_df = result.to_pandas()

    print(
        results_df[
            [
                "question",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
        ].to_string(index=False)
    )

except Exception:

    print(
        "Individual results are not available."
    )


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "\nLangSmith Project:"
    "\nenterprise-rag-baseline-v1"
)

print(
    "\nOpen LangSmith to inspect the Ragas evaluator traces."
)
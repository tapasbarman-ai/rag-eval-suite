# eval_retriever.py
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import GeminiModel

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv(PROJECT_ROOT / ".env", override=True)

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retrivers_goldens.json"
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "2"))
JUDGE_MODEL_NAME = "gemini-3.1-flash-lite"
JUDGE_MODEL = GeminiModel(
    model=JUDGE_MODEL_NAME,
    api_key=os.getenv("GEMINI_API_KEY"),
)
THRESHOLD = 0.7


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    # Default to 1 example to avoid free-tier Gemini quota exhaustion while debugging.
    goldens = load_goldens(GOLDEN_PATH)[:EVAL_LIMIT]

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",          # vs "reranked" when you swap it in
            "embedding_model": "gemini-embedding-001",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": str(GOLDEN_PATH),
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
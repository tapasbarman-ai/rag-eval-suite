import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import GeminiModel

from src.generator import GeminiGenerator

load_dotenv(PROJECT_ROOT / ".env", override=True)

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "correctnes_goldens.json"
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "3"))
JUDGE_MODEL_NAME = "gemini-3.1-flash-lite"
JUDGE_MODEL = GeminiModel(
    model=JUDGE_MODEL_NAME,
    api_key=os.getenv("GEMINI_API_KEY"),
)
THRESHOLD = 0.7


# =====================================================================
# 1. HELPER FUNCTIONS (No need for external evals.harness)
# =====================================================================
def load_goldens(file_path: Path | str) -> list[dict]:
    """Loads query, ideal_context, and ideal_answer from JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Golden dataset not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(test_results):
    """Prints a clean summary of metric scores and pass/fail rates."""
    print("\n" + "=" * 60)
    print(" 📊 APPLICATION EVALUATION SUMMARY")
    print("=" * 60)

    for idx, test in enumerate(test_results.test_results):
        print(f"\n[Test Case #{idx + 1}] {test.input[:60]}...")
        for metric in test.metrics_data:
            status = " PASS" if metric.success else "❌ FAIL"
            print(f"  - {metric.name:15}: {metric.score:.2f} [{status}]")
            if metric.reason:
                print(f"    Reason: {metric.reason}")


# =====================================================================
# 2. DEFINE G-EVAL METRICS WITH RUBRICS
# =====================================================================

# 2a. CORRECTNESS --- Reference-based: judges TRUTH, not coverage or brevity
correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong ONLY if it contradicts the expected output or contains factually false information.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct points for brevity or omitted details — omissions are judged under Completeness, not Correctness.",
        "Additional correct details beyond the expected output must NEVER lower the score.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="All stated claims are factually correct and consistent with the expected output. No contradictions.",
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome="Mostly correct, but contains one minor inaccuracy or ambiguity.",
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome="Contains a clear factual error or directly contradicts the expected output.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# 2b. COMPLETENESS --- Reference-based: judges COVERAGE, not correctness
completeness_metric = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify all key points, facts, or numbers present in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially mentions.",
        "Judge coverage ONLY. Do NOT lower the score because a point is stated incorrectly (correctness is judged separately).",
        "Do NOT penalize the actual output for adding relevant extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Addresses essentially all key points, facts, and figures in the expected output.",
        ),
        Rubric(
            score_range=(5, 8),
            expected_outcome="Covers the core idea but misses one or more specific sub-points or figures.",
        ),
        Rubric(
            score_range=(0, 4),
            expected_outcome="Misses several essential points; only partially covers the question.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)


# =====================================================================
# 3. RUN EVALUATION PIPELINE
# =====================================================================
def run():
    # 1. Load test cases
    goldens = load_goldens(GOLDEN_PATH)[:EVAL_LIMIT]
    generator = GeminiGenerator(model=JUDGE_MODEL_NAME)

    test_cases = []
    print(f"Running pipeline for {len(goldens)} test cases...")

    # 2. Query your live RAG generator per query
    for item in goldens:
        query = item.get("query") or item.get("question")
        ideal = item.get("ideal_answer") or item.get("expected_output")

        # Let generator retrieve chunks and create live answer
        result = generator.generate(query=query, top_k=3)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=result["answer"],
                expected_output=ideal,
                retrieval_context=result["retrieved_contexts"],
            )
        )

    # 3. Run G-Eval
    print("Evaluating with DeepEval G-Eval (Correctness & Completeness)...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, completeness_metric],
    )

    # 4. Display results
    print_summary(results)
    return results


if __name__ == "__main__":
    run()
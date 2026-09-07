import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RagPipeline

load_dotenv(PROJECT_ROOT / ".env", override=True)

# ============================================================
# 1. CONFIGURATION & PRICING
# ============================================================
# Research paper questions to benchmark token load
QUESTIONS = [
    "What is the Transformer architecture and how does it differ from RNNs?",
    "Explain how Scaled Dot-Product Attention works and why scaling by sqrt(dk) is necessary.",
    "What optimizer, hyperparameters, and warmup steps were used during model training?",
    "What were the translation BLEU scores on English-to-German and English-to-French benchmarks?",
]

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
REPEATS = int(os.getenv("COST_REPEATS", "3"))

# --- Gemini 3.1 Flash  (USD per 1M tokens) ---
# Prompts <= 128k tokens:
PRICE_INPUT_PER_1M        = 0.10   # Uncached input: $0.10 / 1M
PRICE_CACHED_INPUT_PER_1M = 0.025  # Context caching: $0.025 / 1M (75% discount)
PRICE_OUTPUT_PER_1M       = 0.40   # Generated response: $0.40 / 1M

# --- Business Projection Knobs ---
QUERIES_PER_DAY = 2000             # Expected production traffic
USD_TO_INR      = 95.0             # Currency conversion rate

# --- Budget (SLO / Cost Ceiling per query) ---
COST_BUDGET_PER_QUERY_USD = 0.0005 # Target: < $0.0005 (~0.043 INR) per query


# ============================================================
# 2. TOKEN MEASUREMENT USING GEMINI SDK
# ============================================================
def measure_gemini_tokens(pipeline: RagPipeline, question: str) -> dict:
    """
    1. Retrieves real context chunks using ChromaDB.
    2. Constructs the exact grounded prompt.
    3. Calls Gemini and extracts usage_metadata from the response.
    """
    # Step 1: Retrieve context from ChromaDB
    docs = pipeline.retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    # Step 2: Build the exact prompt used by your generator
    generator = pipeline.generator
    prompt_text = generator._build_prompt(question, contexts)

    # Step 3: Call Gemini API directly to capture metadata
    client = generator.client
    response = client.models.generate_content(
        model=generator.model,
        contents=prompt_text
    )

    # Step 4: Extract token usage from Gemini metadata
    usage = response.usage_metadata
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

    return {
        "input": prompt_tokens,
        "output": candidates_tokens,
        "cached": cached_tokens,
    }


# ============================================================
# 3. COST CALCULATION FORMULA
# ============================================================
def calculate_cost_usd(input_tokens: int, output_tokens: int, cached_tokens: int) -> dict:
    """Calculates billing breakdown based on Gemini Flash pricing tiers."""
    uncached_input = max(input_tokens - cached_tokens, 0)

    cost_in     = (uncached_input / 1_000_000) * PRICE_INPUT_PER_1M
    cost_cached = (cached_tokens / 1_000_000) * PRICE_CACHED_INPUT_PER_1M
    cost_out    = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M

    return {
        "input": cost_in,
        "cached": cost_cached,
        "output": cost_out,
        "total": cost_in + cost_cached + cost_out,
    }


# ============================================================
# 4. BENCHMARK EXECUTION LOOP
# ============================================================
def benchmark_cost(pipeline: RagPipeline) -> list[dict]:
    rows = []
    total_runs = len(QUESTIONS) * REPEATS
    print(f"Measuring Gemini token usage across {total_runs} benchmark runs...\n")

    for q_idx, question in enumerate(QUESTIONS):
        print(f"[{q_idx + 1}/{len(QUESTIONS)}] Benchmarking: \"{question[:50]}...\"")
        for _ in range(REPEATS):
            tokens = measure_gemini_tokens(pipeline, question)
            costs = calculate_cost_usd(tokens["input"], tokens["output"], tokens["cached"])
            rows.append({**tokens, **{f"cost_{k}": v for k, v in costs.items()}})
            time.sleep(0.3)  # Small pacing to avoid rate limits

    return rows


# ============================================================
# 5. SUMMARY & REPORTING
# ============================================================
def avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def print_cost_report(rows: list[dict], model_name: str):
    n = len(rows)
    avg_in     = avg(rows, "input")
    avg_out    = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost   = avg(rows, "cost_total")
    min_cost   = min(r["cost_total"] for r in rows)
    max_cost   = max(r["cost_total"] for r in rows)

    avg_cost_in  = avg(rows, "cost_input") + avg(rows, "cost_cached")
    avg_cost_out = avg(rows, "cost_output")
    out_share    = (100 * avg_cost_out / avg_cost) if avg_cost else 0

    print("\n" + "=" * 76)
    print(f"💰 OPERATIONAL COST EVALUATION ({model_name})")
    print(f"   Rates: ${PRICE_INPUT_PER_1M}/1M input | ${PRICE_OUTPUT_PER_1M}/1M output | ${PRICE_CACHED_INPUT_PER_1M}/1M cached")
    print("=" * 76)
    print(f"Samples evaluated      : {n}")
    print(f"Avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"Avg output tokens      : {avg_out:8.0f}")
    print("-" * 76)
    print(f"Avg cost / query       : ${avg_cost:.6f}   (₹{avg_cost * USD_TO_INR:.4f})")
    print(f"Min / Max per query    : ${min_cost:.6f} / ${max_cost:.6f}")
    print(f"Cost split             : {100 - out_share:.0f}% input / {out_share:.0f}% output")
    print("-" * 76)

    # --- Business Projection ---
    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"Projection @ {QUERIES_PER_DAY:,} queries/day:")
    print(f"   Per day             : ${daily:8.3f}   (₹{daily * USD_TO_INR:8.2f})")
    print(f"   Per month (30 days) : ${monthly:8.3f}   (₹{monthly * USD_TO_INR:8.2f})")
    print("=" * 76)

    # --- Budget / SLO Verdict ---
    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET TARGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}")
    print(f"VERDICT      : ${avg_cost:.6f}  [{verdict}]")
    print("=" * 76 + "\n")


# ============================================================
# 6. ENTRY POINT
# ============================================================
def main():
    pipeline = RagPipeline(model=MODEL_NAME, top_k=3)
    results = benchmark_cost(pipeline)
    print_cost_report(results, model_name=pipeline.generator.model)


if __name__ == "__main__":
    main()
"""LLM-as-Judge evaluation harness for the credit-risk RAG pipeline.

Runs every case in eval_dataset.json through the RAG pipeline, then asks a
judge LLM (system prompt: banking_revised_judge_prompt.md) to score each
answer on faithfulness, helpfulness, citation accuracy, and efficiency —
using ONLY the query, the retrieved context, the ground truth (if any), and
the generated answer.

Usage:
    python evaluate.py                          # judge model = generator model
    python evaluate.py --judge-model qwen2.5    # use a different (ideally stronger) judge
    python evaluate.py --case ltv_math_exception

Note: judging a model with itself is a known bias (self-preference). For a
real evaluation, point --judge-model at a stronger model than the generator.
"""

import argparse
import json
import os
import re
import sys

from langchain_community.llms import Ollama

from credit_risk_rag import DEFAULT_MODEL, CreditRiskRAG, check_ollama

JUDGE_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banking_revised_judge_prompt.md")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.json")

METRICS = (
    ("metric_1_faithfulness", "Faithful"),
    ("metric_2_helpfulness", "Helpful"),
    ("metric_3_citation_accuracy", "Citations"),
    ("metric_4_efficiency", "BLUF"),
)


def build_judge_input(case: dict, context: str, answer: str) -> str:
    """Assemble the exact inputs the judge prompt declares it will receive."""
    ground_truth = case.get("ground_truth") or "(not provided)"
    return f"""
## Inputs

### original_query
{case["query"]}

### retrieved_context
{context}

### ground_truth
{ground_truth}

### generated_answer
{answer}
"""


def parse_judge_json(raw: str) -> dict:
    """Extract the first JSON object from the judge's output (tolerates fences)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in judge output: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def coerce_score(value):
    """Small local models sometimes emit scores as strings ('2') — normalize."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def verify_flags(flags: dict, answer: str) -> tuple[dict, list]:
    """Enforce the judge prompt's evidence rule programmatically.

    A critical flag only counts if its `evidence` is a verbatim quote from the
    generated answer — smaller judge models sometimes assert a flag while
    'quoting' the rubric instead of the answer. Returns (verified, discarded).
    """
    norm_answer = _norm(answer)
    verified, discarded = {}, []
    for name, f in flags.items():
        if isinstance(f, dict):
            value, evidence = bool(f.get("value")), f.get("evidence")
        else:  # tolerate legacy bare-boolean flags (no evidence to verify)
            value, evidence = bool(f), None
            verified[name] = value
            continue
        if value and (not evidence or _norm(str(evidence)) not in norm_answer):
            discarded.append(name)
            value = False
        verified[name] = value
    return verified, discarded


def compute_verdict(scores: dict, verified_flags: dict) -> str:
    """Recompute the deterministic PASS/FAIL rule from the judge prompt."""
    if any(verified_flags.values()):
        return "FAIL"
    gating = ("metric_1_faithfulness", "metric_2_helpfulness", "metric_3_citation_accuracy")
    if any(scores.get(k) == 0 for k in gating):
        return "FAIL"
    return "PASS"


def fmt_score(value) -> str:
    return "n/a" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge eval for the local RAG pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="generator model (default: %(default)s)")
    parser.add_argument("--judge-model", default=None, help="judge model (default: same as --model)")
    parser.add_argument("--data-dir", default="data", help="policy PDF directory")
    parser.add_argument("--case", default=None, help="run a single case by id")
    parser.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    args = parser.parse_args()
    judge_model = args.judge_model or args.model

    with open(JUDGE_PROMPT_FILE, encoding="utf-8") as f:
        judge_system_prompt = f.read()
    with open(DATASET_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            sys.exit(f"[!] No case with id '{args.case}' in {DATASET_FILE}")

    print("--- LLM-as-Judge Evaluation (100% Local) ---")
    rag = CreditRiskRAG(data_dir=args.data_dir, model=args.model, k=args.k)
    check_ollama(judge_model)
    judge = Ollama(model=judge_model, temperature=0.0, format="json")
    print(f"[*] Generator: {args.model} | Judge: {judge_model} | Cases: {len(cases)}\n")

    results = []
    for i, case in enumerate(cases, 1):
        # flush so progress is visible even when stdout is piped to a log
        print(f"[{i}/{len(cases)}] {case['id']}: generating answer...", flush=True)
        rag_result = rag.ask(case["query"])
        print(f"          intent={rag_result.intent}; judging...", flush=True)
        raw_verdict = judge.invoke(
            judge_system_prompt + build_judge_input(case, rag_result.context, rag_result.answer)
        )
        try:
            verdict = parse_judge_json(raw_verdict)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"          [!] Judge output unparseable: {exc}")
            verdict = None
        results.append({"case": case, "rag": rag_result, "verdict": verdict})

    # ---- Scorecard -------------------------------------------------------
    header = f"{'Case':<24} {'Faithful':>8} {'Helpful':>8} {'Citations':>9} {'BLUF':>5} {'Flags':>5} {'Verdict':>8}"
    print("\n" + "=" * len(header))
    print("SCORECARD")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    totals, counts = {key: 0 for key, _ in METRICS}, {key: 0 for key, _ in METRICS}
    failures = 0
    any_discarded = False
    for r in results:
        cid = r["case"]["id"]
        v = r["verdict"]
        if v is None:
            print(f"{cid:<24} {'JUDGE OUTPUT UNPARSEABLE':>45}")
            failures += 1
            continue
        scores = {}
        for key, _label in METRICS:
            score = coerce_score(v.get(key, {}).get("score"))
            scores[key] = score
            if isinstance(score, int):
                totals[key] += score
                counts[key] += 1
        verified, discarded = verify_flags(v.get("critical_flags", {}), r["rag"].answer)
        if discarded:
            any_discarded = True
        n_flags = sum(verified.values())
        # Don't trust the judge's own verdict: recompute the deterministic rule
        # from verified inputs (the judge may flag without valid evidence).
        overall = compute_verdict(scores, verified)
        if overall != "PASS":
            failures += 1
        print(
            f"{cid:<24} {fmt_score(scores['metric_1_faithfulness']):>8} "
            f"{fmt_score(scores['metric_2_helpfulness']):>8} "
            f"{fmt_score(scores['metric_3_citation_accuracy']):>9} "
            f"{fmt_score(scores['metric_4_efficiency']):>5} "
            f"{str(n_flags) + ('*' if discarded else ''):>5} {overall:>8}"
        )

    print("-" * len(header))
    averages = " ".join(
        f"{label}={totals[key] / counts[key]:.2f}" if counts[key] else f"{label}=n/a"
        for key, label in METRICS
    )
    print(f"Averages (max 2.00): {averages}")
    print(f"Result: {len(results) - failures}/{len(results)} PASS")
    if any_discarded:
        print("*  judge asserted a critical flag without verbatim evidence from the answer — flag discarded")

    # ---- Full rationales -------------------------------------------------
    print("\nDETAILED RATIONALES")
    for r in results:
        print(f"\n--- {r['case']['id']} (intent={r['rag'].intent}) ---")
        print(f"Q: {r['case']['query']}")
        print(f"A: {r['rag'].answer}\n")
        if r["verdict"]:
            print(json.dumps(r["verdict"], indent=2, ensure_ascii=False))

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

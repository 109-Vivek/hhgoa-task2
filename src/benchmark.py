import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Allow running directly as `python src\benchmark.py` from repo root on Windows
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.harness.orchestrator import VoiceRAGOrchestrator, PipelineResponse

BENCHMARK_QUERIES = {
    "gu": [
        "ભારતની રાજધાની કઈ છે?",
        "ભારતના વડાપ્રધાન કોણ છે?",
        "ગોવા ક્યાં આવેલું છે?",
        "રીટ્રીવલ-ઓગમેન્ટેડ જનરેશન શું છે?",
        "વિશ્વનો સૌથી મોટો મહાસાગર કયો છે?",
        "ભારતનું સૌથી નાનું રાજ્ય કયું છે?",
        "RAG પદ્ધતિ એઆઈ મોડેલોમાં ભ્રમણા કેવી રીતે ઘટાડે છે?",
        "પ્રશાંત મહાસાગરનો વિસ્તાર ક્યાં સુધી છે?",
        "૧૯૧૧ માં નવી દિલ્હીનો શિલાન્યાસ કોણે કર્યો હતો?",
        "ક્વોન્ટમ ફિઝિક્સના મૂળભૂત નિયમો શું છે?",  # Out of domain query to test abstention
    ],
    "hi": [
        "भारत की राजधानी क्या है?",
        "भारत के प्रधानमंत्री कौन हैं?",
        "गोवा कहाँ स्थित है?",
        "रिट्रीवल-ऑगमेंटेड जनरेशन क्या है?",
        "विश्व का सबसे बड़ा महासागर कौन सा है?",
        "भारत का सबसे छोटा राज्य कौन सा है?",
        "आरएजी तकनीक से एआई में क्या सुधार होता है?",
        "प्रशांत महासागर का विस्तार कहाँ तक है?",
        "1911 में नई दिल्ली की आधारशिला किसने रखी थी?",
        "क्वांटम भौतिकी के मूल सिद्धांत क्या हैं?",  # Out of domain query
    ],
    "te": [
        "భారతదేశ రాజధాని ఏది?",
        "భారత ప్రధానమంత్రి ఎవరు?",
        "గోవా ఎక్కడ ఉంది?",
        "రిట్రీవల్-ఆగ్మెంటెడ్ జనరేషన్ అంటే ఏమిటి?",
        "ప్రపంచంలో అతిపెద్ద మహాసముద్రం ఏది?",
        "భారతదేశంలో అతి చిన్న రాష్ట్రం ఏది?",
        "RAG విధానం ఏఐ మోడల్స్‌లో భ్రమలను ఎలా తగ్గిస్తుంది?",
        "పసిఫిక్ మహాసముద్రం ఏ ప్రాంతాలలో విస్తరించి ఉంది?",
        "1911 లో న్యూఢిల్లీకి శంకుస్థాపన చేసినది ఎవరు?",
        "క్వాంటం భౌతిక శాస్త్ర నియమాలు ఏమిటి?",  # Out of domain query
    ],
}


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p95": 0.0, "p100": 0.0, "mean": 0.0, "min": 0.0}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p100": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def run_benchmark(
    num_queries: int = 30,
    languages: List[str] = ["gu", "hi", "te"],
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
) -> Dict[str, Any]:
    print("\n=======================================================")
    print("[BENCHMARK] Initializing Indic Voice-RAG Benchmark Harness")
    print(f"[BENCHMARK] Languages: {languages} | Total Target Queries: {num_queries}")
    print("=======================================================\n")

    orchestrator = VoiceRAGOrchestrator()

    # Collect query list
    query_queue = []
    for lang in languages:
        lang_queries = BENCHMARK_QUERIES.get(lang, [])
        for q in lang_queries:
            query_queue.append((q, lang))

    # Expand or slice to match target num_queries
    if len(query_queue) < num_queries:
        multiplier = (num_queries // len(query_queue)) + 1
        query_queue = (query_queue * multiplier)[:num_queries]
    else:
        query_queue = query_queue[:num_queries]

    # Warmup run
    print("[Benchmark] Running warmup query...")
    _ = orchestrator.process_query("Warmup query", "en")

    # Metrics storage
    latencies = {
        "input_guard": [],
        "embedding": [],
        "dense_search": [],
        "lexical_search": [],
        "fusion": [],
        "retrieval_total": [],
        "llm_generation": [],
        "output_guard": [],
        "total_end_to_end": [],
    }

    results_log = []
    abstention_count = 0
    passed_guardrails_count = 0

    print(f"\n[Benchmark] Executing {len(query_queue)} benchmark test queries...")
    for idx, (query, lang) in enumerate(query_queue, start=1):
        res: PipelineResponse = orchestrator.process_query(query=query, language_code=lang)

        # Log latencies
        lat = res.latency
        latencies["input_guard"].append(lat.input_guardrail_ms)
        latencies["embedding"].append(lat.embedding_ms)
        latencies["dense_search"].append(lat.dense_search_ms)
        latencies["lexical_search"].append(lat.lexical_search_ms)
        latencies["fusion"].append(lat.fusion_ms)
        latencies["retrieval_total"].append(lat.total_retrieval_ms)
        latencies["llm_generation"].append(lat.llm_generation_ms)
        latencies["output_guard"].append(lat.output_guardrail_ms)
        latencies["total_end_to_end"].append(lat.total_end_to_end_ms)

        if res.is_abstention:
            abstention_count += 1
        if res.input_guard.get("is_safe", True) and res.output_guard.get("is_grounded", True):
            passed_guardrails_count += 1

        results_log.append({
            "index": idx,
            "query": query,
            "lang": lang,
            "answer": res.answer,
            "is_abstention": res.is_abstention,
            "provider": res.provider,
            "latency_ms": lat.total_end_to_end_ms,
        })

        if idx % 5 == 0 or idx == len(query_queue):
            print(f"  [{idx}/{len(query_queue)}] Query: '{query[:30]}...' | Total: {lat.total_end_to_end_ms:.1f}ms | Lang: {lang}")

    # Compute percentiles
    summary_stats = {
        component: calculate_percentiles(vals) for component, vals in latencies.items()
    }

    e2e = summary_stats["total_end_to_end"]
    retrieval = summary_stats["retrieval_total"]

    print("\n" + "=" * 70)
    print("[RESULTS] BENCHMARK RESULTS & LATENCY ANALYTICS SUMMARY")
    print("=" * 70)
    print(f"Total Queries Evaluated : {len(query_queue)}")
    print(f"Languages Tested        : {', '.join(languages)}")
    print(f"Guardrail Success Rate  : {(passed_guardrails_count/len(query_queue))*100:.1f}%")
    print(f"Abstentions Triggered   : {abstention_count} (out-of-domain safe refusals)")
    print("-" * 70)
    print(f"{'Component':<22} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P90 (ms)':<10} | {'P100 (ms)':<10}")
    print("-" * 70)
    for comp, stats in summary_stats.items():
        print(f"{comp:<22} | {stats['p50']:<10.2f} | {stats['p70']:<10.2f} | {stats['p90']:<10.2f} | {stats['p100']:<10.2f}")
    print("=" * 70)

    p50_e2e = e2e["p50"]
    p70_e2e = e2e["p70"]
    p100_e2e = e2e["p100"]

    print(f"\n[EVALUATION] Target Verification: <200ms Budget Target")
    print(f"   * P50 Latency: {p50_e2e:.2f} ms [{'PASS' if p50_e2e <= 200 else 'EXCEEDS'}]")
    print(f"   * P70 Latency: {p70_e2e:.2f} ms [{'PASS' if p70_e2e <= 200 else 'EXCEEDS'}]")
    print(f"   * P100 Latency: {p100_e2e:.2f} ms")
    print("=" * 70 + "\n")

    report = {
        "num_queries": len(query_queue),
        "languages": languages,
        "summary": summary_stats,
        "queries": results_log,
        "timestamp": time.time(),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Benchmark] Saved JSON results to: {out_path}")

    if output_md:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# HH Goa 2026 Voice Indic RAG - Latency Benchmark Report\n\n")
            f.write(f"- **Queries Tested**: {len(query_queue)}\n")
            f.write(f"- **Languages**: {', '.join(languages)}\n")
            f.write(f"- **P50 Latency**: {p50_e2e:.2f} ms\n")
            f.write(f"- **P70 Latency**: {p70_e2e:.2f} ms\n")
            f.write(f"- **P100 Latency**: {p100_e2e:.2f} ms\n\n")
            f.write("## Component Latency Breakdown (ms)\n\n")
            f.write("| Component | Min | P50 (Median) | P70 | P90 | P100 (Max) | Mean |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for comp, s in summary_stats.items():
                f.write(f"| `{comp}` | {s['min']:.2f} | **{s['p50']:.2f}** | {s['p70']:.2f} | {s['p90']:.2f} | {s['p100']:.2f} | {s['mean']:.2f} |\n")
        print(f"[Benchmark] Saved Markdown report to: {md_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Voice Indic RAG Latency Benchmark")
    parser.add_argument("--num-queries", type=int, default=30, help="Number of benchmark queries")
    parser.add_argument("--languages", nargs="+", default=["gu", "hi", "te"], help="Languages to benchmark")
    parser.add_argument("--output-json", type=str, default="data/benchmark_results.json", help="Path to save JSON results")
    parser.add_argument("--output-md", type=str, default="data/benchmark_report.md", help="Path to save Markdown report")
    args = parser.parse_args()

    run_benchmark(
        num_queries=args.num_queries,
        languages=args.languages,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()

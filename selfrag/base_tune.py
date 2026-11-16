"""Phase 2: the frozen base (Adaptive + CRAG 3f) on Self-RAG's tune set (78 = CRAG's 58 + 20 new), run like
Corrective's crag_eval and graded with the 1-vote and 3-vote graders. The 58 old questions replay from the cache
($0); only the 20 new ones are live calls. The saved answers (with the passages each one read) are what the hand
labels for the critic are written against.

Results: selfrag/results/phase2_base.json.

Run from self_rag_in_depth/ (Qdrant running; OpenAI key sourced):
    uv run selfrag/base_tune.py
"""

import json

import selfrag_bridge
from eval_harness import load_golden

import answer_tuning
import baseline
import crag
import crag_eval
import route_eval
import selfrag_golden as sg

RESULTS = selfrag_bridge.ROOT / "selfrag" / "results" / "phase2_base.json"
KINDS = ("old", "pair", "alike", "sure", "chain", "gap", "trap", "none")


def run(golden: dict, split: str = "tune") -> dict:
    crag.install(selfrag_bridge.BASE)
    picks = {}
    rows = route_eval.run_route("selfrag_base", golden, baseline.routed(picks), golden_set=sg.golden_name(split))
    for i, row in rows.items():
        q = golden[i].question
        row["route"], row["kind"] = picks[q], sg.kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
    return rows


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    rows = run(golden)
    s = crag_eval.summary(rows, None)
    s["by_kind"] = {k: f"{sum(r['correct'] for r in rows.values() if r['kind'] == k):g}/"
                       f"{sum(r['kind'] == k for r in rows.values())}" for k in KINDS}
    RESULTS.write_text(json.dumps({"summary": s, "questions": rows}, indent=2, ensure_ascii=False))
    print(f"\nright {s['correct']:.3f} (1 vote), {s['correct3']:.3f} (3 votes); by kind {s['by_kind']}")
    print(f"routes {s['routes']}, actions {s['actions']}, ${s['cost_per_question']:.5f}/question, "
          f"p50 {s['p50_ms'] / 1000:.1f} s")
    for i, r in rows.items():
        if r["kind"] in ("pair", "alike", "sure", "none") or r["correct"] < 1:
            print(f"  {i:9} {r['route']:6} {(r['crag'] or {}).get('action', '-'):9} {r['correct']:g}/{r['correct3']:g}  "
                  f"{r['answer'][:150]}")


if __name__ == "__main__":
    main()

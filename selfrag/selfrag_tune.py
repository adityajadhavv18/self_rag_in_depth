"""Phase 4: Self-RAG (sure check, candidates per passage + A_all, gpt-4o-mini critic v2, retry cap 1) on the full tune
set (78), run like base_tune.py and graded with the 1-vote and 3-vote graders, then compared question by question
with the base's saved run (phase2_base.json, 0.910).

Results: selfrag/results/phase4_tune.json.

Run from self_rag_in_depth/ (Qdrant running; OpenAI key sourced):
    uv run selfrag/selfrag_tune.py
"""

import json

import selfrag_bridge
from eval_harness import load_golden

import answer_tuning
import baseline
import crag
import crag_eval
import route_eval
import self_rag
import selfrag_golden as sg
from base_tune import KINDS

CONFIG = {"sure": True, "retries": 1}
RESULTS = selfrag_bridge.ROOT / "selfrag" / "results"


def run(golden: dict, split: str = "tune") -> dict:
    self_rag.install(CONFIG)
    picks = {}
    rows = route_eval.run_route("selfrag", golden, baseline.routed(picks), golden_set=sg.golden_name(split))
    for i, row in rows.items():
        q = golden[i].question
        log = self_rag.LOG.get(q, {})
        row["route"], row["kind"] = picks[q], sg.kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
        row["selfrag"] = {k: log.get(k) for k in ("sure", "picked", "good", "retries", "candidates")}
    return rows


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((RESULTS / "phase2_base.json").read_text())
    rows = run(golden)
    s = crag_eval.summary(rows, None)
    s["by_kind"] = {k: f"{sum(r['correct'] for r in rows.values() if r['kind'] == k):g}/"
                       f"{sum(r['kind'] == k for r in rows.values())}" for k in KINDS}
    s["sure"] = {v: sum((r["selfrag"]["sure"] == v) for r in rows.values()) for v in ("sure", "unsure")}
    s["retried"] = sum(bool(r["selfrag"]["retries"]) for r in rows.values())
    s["changed"] = {i: f"{base['questions'][i]['correct']:g} -> {r['correct']:g}" for i, r in rows.items()
                    if r["correct"] != base["questions"][i]["correct"]}
    (RESULTS / "phase4_tune.json").write_text(json.dumps({"config": CONFIG, "summary": s, "questions": rows},
                                                         indent=2, ensure_ascii=False))
    b = base["summary"]
    print(f"\nright: base {b['correct']:.3f} -> self-rag {s['correct']:.3f} (1 vote); "
          f"{b['correct3']:.3f} -> {s['correct3']:.3f} (3 votes)")
    print(f"by kind: base {b['by_kind']}\n         now  {s['by_kind']}")
    print(f"sure check {s['sure']}, retried {s['retried']}, ${b['cost_per_question']:.5f} -> "
          f"${s['cost_per_question']:.5f}/question, p50 {b['p50_ms'] / 1000:.1f} -> {s['p50_ms'] / 1000:.1f} s")
    print(f"changed: {s['changed'] or 'none'}")


if __name__ == "__main__":
    main()

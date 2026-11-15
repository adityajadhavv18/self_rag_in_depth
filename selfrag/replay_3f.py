"""Phase 1: the base Self-RAG sits on. Corrective's final system (Adaptive + CRAG 3f) on the CRAG tune set (58),
run live through the bridge:

    few-shot router -> none / single / multi -> CRAG 3f (grade, fallback) -> answer prompt v2

and graded with the 1-vote and 3-vote graders. Every LLM call should come from the caches; run it with a dummy
OpenAI key and any uncached call fails instead of spending. It must match Corrective's saved 3f run (0.931) question
by question: score, route, CRAG action and answer text, or the reuse is broken.

Results go to selfrag/results/phase1.json.

Run from self_rag_in_depth/ (Qdrant running):
    OPENAI_API_KEY=sk-dummy uv run selfrag/replay_3f.py
"""

import json

import selfrag_bridge
from eval_harness import load_golden

import crag_eval
import crag_golden_check as gc

RESULTS = selfrag_bridge.ROOT / "selfrag" / "results" / "phase1.json"
SAVED = selfrag_bridge.CORRECTIVE / "results" / "phase3.json"
FIELDS = ("correct", "correct3", "route", "answer")


def main():
    golden = {g.id: g for g in load_golden(gc.golden_name("tune"))}
    rows = crag_eval.run("3f", golden)
    old = json.loads(SAVED.read_text())["3f"]
    now, then = crag_eval.summary(rows, None), old["summary"]

    action = lambda row: row["crag"]["action"] if row["crag"] else None
    differs = {}
    for i in golden:
        then_row = old["questions"][i]
        d = [f for f in FIELDS if rows[i][f] != then_row[f]] + ["action"] * (action(rows[i]) != action(then_row))
        if d:
            differs[i] = d
    RESULTS.write_text(json.dumps({"summary": now, "saved_summary": then, "differs": differs, "questions": rows},
                                  indent=2, ensure_ascii=False))

    print("\n| | now | Corrective saved |\n|---|---|---|")
    for key in ("correct", "correct3"):
        print(f"| {key} | {now[key]:.3f} | {then[key]:.3f} |")
    for key in ("by_kind", "actions", "routes"):
        print(f"| {key} | {now[key]} | {then[key]} |")
    print(f"| $/question (original calls) | {now['cost_per_question']:.5f} | {then['cost_per_question']:.5f} |")
    print(f"| p50 s | {now['p50_ms'] / 1000:.1f} | {then['p50_ms'] / 1000:.1f} |")
    print(f"\nquestions that differ: {differs or 'none'}")


if __name__ == "__main__":
    main()

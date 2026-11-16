"""Phase 3: Self-RAG (retry cap 1, gpt-4o-mini critic v2) on 5 tune questions, one per new part, against the base's
saved answers (phase2_base.json):

    chain-02, trap-05   partly backed       -> the critic should flag them
    none-11             not backed (121)    -> the critic should prefer "I don't know"
    alike-04            refused             -> the retry should find the answer
    sure-01             routed to memory    -> the "are you sure?" check

Results: selfrag/results/phase3_smoke.json.

Run from self_rag_in_depth/ (Qdrant running; OpenAI key sourced):
    uv run selfrag/smoke.py
"""

import json

import selfrag_bridge
from eval_harness import load_golden, settings
from eval_harness.metrics import cost_usd, is_refusal

import baseline
import route_eval
import self_rag
import selfrag_golden as sg

IDS = ("chain-02", "trap-05", "none-11", "alike-04", "sure-01")
RESULTS = selfrag_bridge.ROOT / "selfrag" / "results"


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((RESULTS / "phase2_base.json").read_text())["questions"]
    self_rag.install({"sure": True, "retries": 1})
    pipeline, picks, rows = baseline.routed({}), {}, {}
    for i in IDS:
        g = golden[i]
        out = pipeline(g.question)
        if g.question_type == "no_answer":
            correct = float(is_refusal(out.answer))
        else:
            correct = 0.0 if is_refusal(out.answer) else route_eval.grade(g.question, g.ground_truth, out.answer)
        log = self_rag.LOG.get(g.question, {})
        rows[i] = {"base": base[i]["correct"], "now": correct, "answer": out.answer,
                   "cost": cost_usd(out.tokens_in, out.tokens_out, settings.llm_model), **log}
        print(f"\n{i}: base {base[i]['correct']:g} -> now {correct:g}  (sure check: {log.get('sure', '-')}, "
              f"picked {log.get('picked', '-')}, ${rows[i]['cost']:.4f})")
        print(f"  answer: {out.answer[:160]}")
        for c in log.get("candidates", []):
            print(f"  {c['name']:7} {c['supported']:6} use {c['useful']}  on {c.get('on_target')}  {c['answer'][:80]!r}")
        for r in log.get("retries", []):
            print(f"  retry: {r['query']!r} -> {r['found']} new, kept {r['kept']}")
    (RESULTS / "phase3_smoke.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nright: base {sum(r['base'] for r in rows.values()):g}/5 -> now {sum(r['now'] for r in rows.values()):g}/5, "
          f"${sum(r['cost'] for r in rows.values()):.4f} total")


if __name__ == "__main__":
    main()

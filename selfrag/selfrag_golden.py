"""Phase 2: build and check Self-RAG's golden sets before anything is run on them.

golden_cricket_selfrag_tune_v1.jsonl = CRAG's 58 tune questions (unchanged) + labels/new_tune.jsonl (20)
golden_cricket_selfrag_test_v1.jsonl = labels/new_test.jsonl (24, fresh: CRAG's test set is spent)

New kinds (from the id; written to labels/kind_labels_<split>.jsonl, outside the harness contract):
    pair    two facts asked together, in two different chunks         -> the critic's IsUse ("half answered")
    alike   a look-alike passage holds a tempting wrong answer         -> the critic's IsSup ("not what it says")
    sure    the router sends it to memory, but the answer is in the corpus -> the "are you sure?" check
    none    no answer in either library                                -> the critic must not push for one
Test also has solo / chain / gap (as in Corrective) to check that nothing breaks. CRAG's tune kinds keep their ids.

Checks, for every new question:
  1. the harness loads the file (every source_doc in the main corpus or the fallback library)
  2. every evidence part is in a chunk, word for word, and the last part is in the source_doc
  3. pair / chain: no single chunk holds every part
  4. alike: each decoy is in a main-index chunk that holds no evidence part
  5. gap: no main-index chunk holds any part; the fallback library holds them all
With --search (f6a's search, main index, top 20; rewrites and reranks are cached after the first run):
  alike: a decoy reaches the top 5 (the answer step would read it), else PROBLEM
  solo: the evidence is in the top 5; chain: the top 5 does not hold every step
  pair: how many parts the top 5 holds (reported); none: the top 5's headers, for a hand check
With --router: sure questions must be routed to none by the few-shot router
With --memory: the memory answer is graded (reported); a gap question answered right from memory is rejected
The search, router and memory checks run on tune only, unless --test is given.

Run from self_rag_in_depth/ (Qdrant running; OpenAI key sourced for --search / --router / --memory):
    uv run selfrag/selfrag_golden.py [--search] [--router] [--memory] [--test]
"""

import json
import sys

import selfrag_bridge
from chunks import included_docs, load_chunks
from eval_harness import load_golden
from eval_harness.metrics import is_refusal
from ingest import VARIANTS
from ladder import fragments, holds, norm

import crag_golden_check
import fallback_index

ROOT = selfrag_bridge.ROOT
LABELS = ROOT / "labels"
GOLDEN = selfrag_bridge.RAGS / "eval_harness" / "golden_sets"
CRAG_TUNE = GOLDEN / crag_golden_check.golden_name("tune")
RESULTS = ROOT / "selfrag" / "results" / "phase2_check.json"
KINDS = ("pair", "alike", "sure", "solo", "chain", "gap", "trap", "none")


def golden_name(split: str) -> str:
    return f"golden_cricket_selfrag_{split}_v1.jsonl"


def kind(gid: str) -> str:
    return next((k for k in KINDS if gid.startswith(k + "-")), "old")


def assemble() -> None:
    """Write the two golden files and the kind labels from CRAG's tune set and labels/new_*.jsonl."""
    for split in ("tune", "test"):
        new = (LABELS / f"new_{split}.jsonl").read_text().strip().splitlines()
        lines = (CRAG_TUNE.read_text().strip().splitlines() if split == "tune" else []) + new
        (GOLDEN / golden_name(split)).write_text("\n".join(lines) + "\n")
        ids = [json.loads(line)["id"] for line in lines]
        (LABELS / f"kind_labels_{split}.jsonl").write_text(
            "".join(json.dumps({"id": i, "kind": kind(i)}) + "\n" for i in ids))


def check(split: str, main, fallback, decoys, new_ids, search=None, route=None, memory=None):
    docs = set(included_docs()) | set(fallback_index.left_out())
    golden = load_golden(golden_name(split), known_docs=docs)
    text = lambda c: VARIANTS["header_match"](c)  # what the answer step reads
    problems, report = [], {}
    other = {g.id for g in load_golden(golden_name("test" if split == "tune" else "tune"))}
    problems += [f"{i}: id also in the other split" for i in {g.id for g in golden} & other]
    for g in (g for g in golden if g.id in new_ids):
        k, row = kind(g.id), {"kind": kind(g.id)}
        if k == "none":
            if g.question_type != "no_answer":
                problems.append(f"{g.id}: none-* must be no_answer")
            if search:
                row["top5"] = [" › ".join(h_path) for h_path in (_path(h) for h in search(g.question, 5))]
        else:
            parts = fragments(g)
            in_main = [[c for c in main if holds(p, text(c))] for p in parts]
            in_fall = [[c for c in fallback if holds(p, text(c))] for p in parts]
            row["chunks"] = [f"{len(m)} main + {len(f)} fallback" for m, f in zip(in_main, in_fall)]
            if not all(m or f for m, f in zip(in_main, in_fall)):
                problems.append(f"{g.id}: an evidence part is in no chunk")
            elif not any(c.doc_id == g.source_doc for c in in_main[-1] + in_fall[-1]):
                problems.append(f"{g.id}: the answer part is not in {g.source_doc}")
            if k in ("pair", "chain") and len(parts) < 2:
                problems.append(f"{g.id}: {k} needs two evidence parts")
            if k in ("pair", "chain") and any(all(holds(p, text(c)) for p in parts) for c in main + fallback):
                problems.append(f"{g.id}: one chunk holds every part")
            if k == "gap" and (any(in_main) or not all(in_fall)):
                problems.append(f"{g.id}: gap, but the main index holds a part or the fallback misses one")
            if k == "alike":
                for d in decoys.get(g.id, []):
                    where = [c for c in main if norm(d) in norm(text(c)) and not any(holds(p, text(c)) for p in parts)]
                    row.setdefault("decoy_chunks", []).append(len(where))
                    if not where:
                        problems.append(f"{g.id}: decoy {d[:40]!r} is in no chunk apart from the evidence")
                if not decoys.get(g.id):
                    problems.append(f"{g.id}: alike without a decoy")
            if search and k in ("pair", "alike", "solo", "chain"):
                texts = [text_of(h) for h in search(g.question, 20)]
                found5 = [any(holds(p, t) for t in texts[:5]) for p in parts]
                row["top5_parts"] = f"{sum(found5)}/{len(parts)}"
                row["evidence_rank"] = next((i for i, t in enumerate(texts, 1) if any(holds(p, t) for p in parts)), None)
                if k == "solo" and not all(found5):
                    problems.append(f"{g.id}: f6a's top 5 misses the evidence")
                if k == "chain" and all(found5):
                    problems.append(f"{g.id}: f6a's top 5 holds every step")
                if k == "alike":
                    row["decoy_ranks"] = [i for i, t in enumerate(texts, start=1)
                                          if any(norm(d) in norm(t) for d in decoys.get(g.id, []))
                                          and not any(holds(p, t) for p in parts)]
                    if not any(r <= 5 for r in row["decoy_ranks"]):
                        problems.append(f"{g.id}: no decoy in f6a's top 5")
        if route and k == "sure":
            row["route"] = route(g.question)
            if row["route"] != "none":
                problems.append(f"{g.id}: the router sends it to {row['route']}, not none")
        if memory and k in ("sure", "gap", "pair", "alike"):
            right, answer = memory(g)
            row["memory"] = {"correct": right, "answer": answer}
            if k == "gap" and right == 1.0:
                problems.append(f"{g.id}: the model answers it from memory")
        report[g.id] = row
        print(f"  {g.id:9} {k:5} " + "  ".join(f"{a}={b}" for a, b in row.items() if a not in ("kind", "memory", "top5"))
              + (f"  memory={row['memory']['correct']:g}" if "memory" in row else ""))
        for header in row.get("top5", []):
            print(f"              top5: {header}")
    return problems, report


def _path(hit) -> list[str]:
    from ladder import _chunks_by_id
    return _chunks_by_id()[hit.chunk_id].path


def text_of(hit) -> str:
    from ladder import labelled
    return labelled(hit)


def main():
    assemble()
    search = route = memory = None
    if "--search" in sys.argv:
        import fusion_ladder  # noqa: F401  (registers f5a, the search of f6a)
        from ladder import RUNGS
        search = RUNGS["f5a"][1]
    if "--router" in sys.argv:
        import router
        route = lambda q: router.route(q, "few")[0]
    if "--memory" in sys.argv:
        import route_eval
        import routes

        def memory(g):  # the harness's rule: a refusal on an answerable question is wrong, before any grading
            answer = routes.no_search(g.question).answer
            return 0.0 if is_refusal(answer) else route_eval.grade(g.question, g.ground_truth, answer), answer
    main_chunks = load_chunks()
    fallback = fallback_index.fallback_chunks(fallback_index.left_out())
    decoys = json.loads((LABELS / "decoys.json").read_text())
    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    bad = 0
    for split in ("tune", "test"):
        live = split == "tune" or "--test" in sys.argv
        new_ids = {json.loads(line)["id"] for line in (LABELS / f"new_{split}.jsonl").read_text().splitlines()}
        print(f"\n{split}{'' if live else ' (offline checks only)'}:")
        problems, rows = check(split, main_chunks, fallback, decoys, new_ids,
                               *((search, route, memory) if live else (None, None, None)))
        for p in problems:
            print(f"  PROBLEM {p}")
        out[split], out[split + "_problems"] = rows, problems
        bad += len(problems)
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{'all checks passed' if not bad else f'{bad} problem(s)'}")


if __name__ == "__main__":
    main()

"""Reuse Corrective RAG #5 as-is (and through it Adaptive #4, RAG Fusion #3 and Contextual RAG #2): router, routes,
f6a search, CRAG 3f with its fallback library, answer prompt v2, graders, chunks and Qdrant collections.

Corrective's folder goes on sys.path; its crag_bridge adds Adaptive's, which adds Fusion's fusion/ and Contextual's
cricket/. Every module there is imported by bare name, so our modules must not reuse any of their names:
    Contextual #2: chunks, ingest, retrieve, hyde, llm_rerank, notes, ladder, answer
    Fusion #3:     contextual, rewrite, rag_fusion, diagnose, fusion_ladder, fusion_answer
    Adaptive #4:   bridge, bar, golden_check, llm_cache, routes, route_eval, loop_knobs, router, router_eval,
                   end_to_end, stacked, answer_tuning, test_run
    Corrective #5: crag_bridge, crag, crag_eval, grader, grader_eval, baseline, chunk_labels, crag_golden_check,
                   fallback_index, laya_eval, model_check, test_once, test_relabel, compare_archs

Corrective's LLM cache is copied once into this build's data/cache/llm.json and llm_cache is pointed at the copy:
replays are free and new calls land here. The f6a search's rewrite and rerank caches stay in Fusion's and
Contextual's folders, as before. Importing this installs CRAG 3f, the frozen system Self-RAG sits on.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAGS = ROOT.parent
CORRECTIVE = RAGS / "corrective_rag_in_depth" / "corrective"
sys.path.append(str(CORRECTIVE))

import crag_bridge  # noqa: E402  (Corrective's bridge: adds Adaptive, Fusion and Contextual)
import crag  # noqa: E402
import crag_eval  # noqa: E402
import llm_cache  # noqa: E402

CACHE = ROOT / "data" / "cache" / "llm.json"
if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(crag_bridge.CACHE, CACHE)
llm_cache.CACHE = CACHE

BASE = crag_eval.STEPS["3f"][1]  # {"discard": True, "fallback": True}
crag.install(BASE)

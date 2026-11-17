"""Phase 3: Self-RAG (Asai et al., 2023) as a layer after CRAG 3f, with a prompted critic (gpt-4o-mini, no fine-tune).
Tried as the critic on the 5 smoke questions and not kept: Gemini (free tier: 20 requests a day) and gpt-5-mini.

CRAG judges the passages before the answer; Self-RAG judges the answer after:

    none route   "are you sure?" -> sure: the memory answer / unsure: f6a's single search
    after CRAG   candidates: one answer per kept passage (A1..An) + CRAG's own answer from all of them (A_all)
    critic       each candidate, against all the passages: supported? full / partly / no (IsSup) and
                 useful? 1-5 (IsUse)
    pick         most supported, then most useful (A_all first on a tie); "I don't know" beats only "no"
    retry        best not good enough (full and useful >= 4): one new query from what is missing -> f6a, skipping
                 chunks already seen -> CRAG's grader keeps 5s and 10s -> new candidates -> critic (cap: retries)
    still not    return the best answer seen

install() swaps routes.answer_from (after CRAG) and the none route of the router's table, as crag.install does.
LOG[question] holds what Self-RAG did.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import selfrag_bridge
from eval_harness import PipelineOutput, settings, traced_openai
from eval_harness.metrics import is_refusal
from ladder import labelled

import baseline
import crag
import grader
import hyde
import llm_cache
import routes
import stacked
from llm_cache import Reply, chat

CONFIG = {"sure": True, "retries": 1}
LOG: dict[str, dict] = {}
SUP = {"no": 0, "partly": 1, "full": 2}
GOOD_USE = 4
THREADS = 8

SURE = """You are about to answer this cricket question from memory, with no documents:
{question}

Are you sure you know the exact answer (names, numbers, dates), with no chance of mixing it up with a similar match,
season or player? Reply with one word: sure or unsure."""

# Critic v2 (Phase 3 smoke): v1 asked "is every fact in the excerpts?" and passed facts about the wrong match or year
# (none-11: Smith's 121 is from the 2023 final, asked about 2021). v2 first fixes the target, then checks the facts.
# On gpt-4o-mini v2 still passed all three bad answers, and so did v3 (one yes/no per passage plus a quote checked in
# code). Gemini 3.6 Flash caught 2 of the 3 with v2, gpt-5-mini 1; the critic stays on gpt-4o-mini with v2.
CRITIC = """You check an answer to a cricket question against the excerpts it was written from. Work in two steps.

Step 1, the target: write down exactly what the question asks about (the match, series or event with its year or
season, the team, the player, the ground). Then decide, excerpt by excerpt, which ones are about that exact target.
An excerpt about a different year, season, edition, match, ground or player does not count, however similar it looks
(the 2023 final is not the 2021 final; the record at one ground is not the overall record).

Step 2, the answer: check every fact in the answer against the excerpts that passed step 1 only. A claim that
something is the highest, first, only or a record needs an excerpt saying so; if an on-target excerpt shows a bigger
number, the claim is not supported.

Excerpts:
{context}

Question: {question}
Answer: {answer}

Reply in JSON with these keys:
"target": the exact match or event, year, team, player and ground asked about
"on_target": the numbers of the excerpts about that exact target
"supported": "full" if every fact in the answer is stated in on-target excerpts, "partly" if only some are,
             "no" if its main fact is not stated in any on-target excerpt
"useful": 1 to 5, how fully it answers the question (5 = every part answered, 1 = no answer at all)
"missing": in a few words, the fact still needed to answer the question fully ("" if nothing)"""

QUERY = """A search of a cricket Wikipedia library did not find everything needed to answer this question.

Question: {question}
Best answer so far: {answer}
Still missing: {missing}

Write one short search query for the missing fact. Reply with the query only."""


def numbered(texts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))


def prompt(question: str, texts: list[str]) -> str:
    return routes.ANSWERS["v2"].format(no_answer=routes.NO_ANSWER, context=numbered(texts), question=question, notes="")


def write(question: str, texts: list[str], replies: list) -> str:
    reply = chat(prompt(question, texts))
    replies.append(reply)
    return reply.text


def verdict(question: str, text: str) -> dict:
    try:
        v = json.loads(text)
        sup = str(v.get("supported", "no")).lower()
        return {"supported": sup if sup in SUP else "no", "useful": int(v.get("useful", 1)),
                "missing": str(v.get("missing", "")), "target": str(v.get("target", "")),
                "on_target": v.get("on_target", [])}
    except (ValueError, TypeError, AttributeError):
        return {"supported": "no", "useful": 1, "missing": question}


def warm(prompts: list[str], json_mode: bool = False) -> None:
    """Put every prompt chat() hasn't seen into llm_cache (same key as chat), THREADS calls at a time; the cache is
    written from this thread only. chat() then reads them back in order."""
    model, cache = settings.llm_model, llm_cache._load()
    key = lambda p: hashlib.sha1(json.dumps([model, 0.0, None, p] + ["json"] * json_mode).encode()).hexdigest()
    todo = list({key(p): p for p in prompts if key(p) not in cache}.items())
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

    def call(prompt: str) -> dict:
        start = time.perf_counter()
        resp = traced_openai().chat.completions.create(model=model, temperature=0.0, max_tokens=None,
                                                       messages=[{"role": "user", "content": prompt}], **extra)
        return Reply(resp.choices[0].message.content.strip(), resp.usage.prompt_tokens,
                     resp.usage.completion_tokens, (time.perf_counter() - start) * 1000).__dict__

    with ThreadPoolExecutor(THREADS) as pool:
        for (k, _), reply in zip(todo, pool.map(lambda kp: call(kp[1]), todo)):
            cache[k] = reply
    if todo:
        llm_cache.CACHE.write_text(json.dumps(cache, ensure_ascii=False))


def batch(prompts: list[str], replies: list, json_mode: bool = False) -> list[Reply]:
    """chat() on many prompts, run in parallel: the batch takes as long as its slowest call, not the sum."""
    warm(prompts, json_mode)
    out = [chat(p, json_mode=json_mode) for p in prompts]
    if out:
        llm_cache.CACHED_MS["ms"] -= sum(r.ms for r in out) - max(r.ms for r in out)
    replies += out
    return out


def critique(question: str, cands: list[dict], replies: list) -> list[dict]:
    """Every candidate at once. A refusal claims nothing: fully backed, no use."""
    ask = [c for c in cands if not is_refusal(c["answer"])]
    got = batch([CRITIC.format(context=numbered(c["texts"]), question=question, answer=c["answer"]) for c in ask],
                replies, json_mode=True)
    out = {id(c): c | verdict(question, r.text) for c, r in zip(ask, got)}
    return [out.get(id(c), c | {"supported": "full", "useful": 1, "missing": question}) for c in cands]


def candidates(question: str, texts: list[str], new: list[str], a_all: str, tag: str, replies: list) -> list[dict]:
    """A_all reads every passage; A1..An one new passage each (passages already turned into candidates are not redone).
    The critic checks every candidate against all the passages: checked against its own passage only, trap-05's
    "235 is the highest" passed, as the 303* was in another passage."""
    written = batch([prompt(question, [t]) for t in new], replies)
    cands = [{"name": f"A_all{tag}", "answer": a_all, "texts": texts}]
    cands += [{"name": f"A{i}{tag}", "answer": r.text, "texts": texts} for i, r in enumerate(written, start=1)]
    return critique(question, cands, replies)


def rank(c: dict) -> tuple:
    """Most supported, then most useful. A refusal ranks between "no" and "partly": it replaces an unsupported answer,
    never a partly supported one (Phase 4: with refusals ranked as "full", multi-02's right 328 lost to "I don't know")."""
    sup = 0.5 if is_refusal(c["answer"]) else SUP[c["supported"]]
    return sup, c["useful"], c["name"].startswith("A_all")


def good(c: dict) -> bool:
    return c["supported"] == "full" and c["useful"] >= GOOD_USE


def answer_from(question: str, hits: list, replies: list, answer: str = "v2", notes: str = "") -> PipelineOutput:
    base = crag.crag_answer_from(question, hits, replies, answer, notes)
    clog = crag.LOG[question]
    ids, texts, mine = list(clog["kept"]), list(base.contexts), []
    seen = {h.chunk_id for h in hits} | set(ids) | {c for x in ("fallback", "deep") for c in clog.get(x, {}).get("kept", [])}
    cands = candidates(question, texts, texts, base.answer, "", mine)
    log = {"retries": []}
    usage = dict(tokens_in=0, tokens_out=0)
    while not good(best := max(cands, key=rank)) and len(log["retries"]) < CONFIG["retries"]:
        q = chat(QUERY.format(question=question, answer=best["answer"], missing=best["missing"] or question))
        mine.append(q)
        before = dict(hyde.USAGE)
        found = [h for h in routes.f6a(q.text.strip().strip('"')) if h.chunk_id not in seen]
        usage = {k: usage[k] + hyde.USAGE[k] - before[k] for k in usage}
        seen |= {h.chunk_id for h in found}
        scores, r = grader.grade(question, [labelled(h) for h in found])
        mine += r
        new = [labelled(h) for h, s in zip(found, scores) if s >= grader.LOWER]
        ids += [h.chunk_id for h, s in zip(found, scores) if s >= grader.LOWER]
        log["retries"].append({"query": q.text, "found": len(found), "scores": scores, "kept": len(new)})
        if not new:
            continue
        texts += new
        cands += candidates(question, texts, new, write(question, texts, mine), f"r{len(log['retries'])}", mine)
    log |= {"picked": best["name"], "good": good(best),
            "candidates": [{k: c.get(k) for k in ("name", "supported", "useful", "missing", "target", "on_target",
                                                  "answer")} for c in cands]}
    LOG.setdefault(question, {}).update(log)
    return PipelineOutput(answer=best["answer"], contexts=texts, context_ids=[crag._by_id[i].doc_id for i in ids],
                          tokens_in=base.tokens_in + sum(r.tokens_in for r in mine) + usage["tokens_in"],
                          tokens_out=base.tokens_out + sum(r.tokens_out for r in mine) + usage["tokens_out"])


def sure_or_search(question: str) -> PipelineOutput:
    reply = chat(SURE.format(question=question), max_tokens=3)
    unsure = "unsure" in reply.text.lower()
    LOG.setdefault(question, {})["sure"] = "unsure" if unsure else "sure"
    out = (stacked.F6A_ROUTES["single"] if unsure else routes.no_search)(question)  # both reset CACHED_MS
    llm_cache.CACHED_MS["ms"] += reply.ms
    return PipelineOutput(answer=out.answer, contexts=out.contexts, context_ids=out.context_ids,
                          tokens_in=out.tokens_in + reply.tokens_in, tokens_out=out.tokens_out + reply.tokens_out)


def install(config: dict | None) -> None:
    """config None: the base (Adaptive + CRAG 3f) as it is. Otherwise Self-RAG on top, with these settings."""
    crag.install(selfrag_bridge.BASE)
    LOG.clear()
    if config is None:
        baseline.ROUTE_FNS["none"] = routes.no_search
        return
    CONFIG.update(config)
    routes.answer_from = answer_from
    baseline.ROUTE_FNS["none"] = sure_or_search if CONFIG["sure"] else routes.no_search

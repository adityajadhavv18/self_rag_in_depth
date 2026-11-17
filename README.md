# Self-RAG #6: cricket on Wikipedia

Ask "how many runs did Steve Smith score in the 2021 World Test Championship final?" and Corrective RAG #5 answers "121". The number is real, but it comes from the **2023** final; Australia did not play the 2021 final. Self-RAG adds a check after the answer is written: did the passages really say this, about this match? I built that check on top of Corrective #5 and measured whether it catches answers like this one.

**Result on tune (78 questions): Corrective #5 0.910 → with Self-RAG 0.897.** With gpt-4o-mini as the checker, it caught none of the 7 answers the system gets wrong, wrongly rejected one right answer, and cost $0.0019 more per question. A stronger checker (Gemini) caught 2 of 3 bad answers on a 5-question trial, so the idea works when the checker can reason. A small, fast model cannot.

```
                         ┌──────────────────────┐
                         │   User's question    │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ ROUTER (Adaptive #4) │   few-shot gpt-4o-mini, one word
                         └──┬────────┬────────┬─┘
                            │        │        │
              ┌─────────────┘        │        └──────────────┐
              ▼                      ▼                       ▼
   ┌──────────────────────┐   ┌──────────────┐   ┌─────────────────────────┐
   │ NONE                 │   │ SINGLE       │   │ MULTI (IRCoT loop)      │
   │ "are you sure?" (new)│   │ one f6a      │   │ f6a(question) → note    │
   │ sure → from memory   │   │ search, top 5│   │ → f6a(note) → …         │
   │ unsure → SINGLE ─────┼──▶│              │   │ (at most 3 searches)    │
   └─────────┬────────────┘   └──────┬───────┘   └────────────┬────────────┘
             │                      └────────────┬───────────┘
             │                                   ▼
             │              ┌─────────────────────────────────────┐
             │              │ CRAG 3f (Corrective #5, unchanged)  │  judges the PASSAGES,
             │              │ grade 10/5/0, fallback library      │  before the answer
             │              └──────────────────┬──────────────────┘
             │                                 ▼
             │              ┌─────────────────────────────────────┐
             │              │ CANDIDATES (new)                    │
             │              │ A1 … An: one answer per passage     │
             │              │ A_all: the usual answer, all of them│
             │              └──────────────────┬──────────────────┘
             │                                 ▼
             │              ┌─────────────────────────────────────┐
             │              │ CHECKER (new)                       │  judges the ANSWER,
             │              │ each candidate, against all the     │  after it is written
             │              │ passages:                           │
             │              │   backed?  full / partly / no       │
             │              │   useful?  1–5                      │
             │              └──────────────────┬──────────────────┘
             │                                 ▼
             │              pick: most backed, then most useful
             │              ("I don't know" beats only "no")
             │                                 │
             │              good enough? (full and useful ≥ 4)
             │                 │ yes                    │ no
             │                 │                        ▼
             │                 │      RETRY (new, once): a new query for what is
             │                 │      missing → f6a, skipping passages already seen
             │                 │      → CRAG's grader → candidates → checker
             │                 │                        │
             ▼                 ▼                        ▼
                         ┌──────────────────────┐
                         │ ANSWER: the best one │   still not good: the best seen
                         └──────────────────────┘
```

Think of a newspaper. The reporters (the answer step) write the stories, and CRAG already checks the source notes before anyone writes. Self-RAG adds an editor who reads each finished story next to its notes before it goes to print. A careful editor spikes a story that quotes the wrong match. A lenient one waves it through, and a nervous one spikes good stories. Self-RAG is only as good as its editor.

- **Self-RAG** (Asai et al., 2023, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"): the model emits reflection tokens. *Retrieve* decides whether to search, *IsRel* judges each passage, *IsSup* judges whether the answer is backed by the passage, and *IsUse* judges whether it answers the question. It writes one answer per passage and keeps the best. The paper fine-tunes Llama 2 to emit the tokens; I use prompts.
- **What was already there:** *Retrieve* is Adaptive #4's router, and *IsRel* is Corrective #5's grader. What is new here is *IsSup* + *IsUse* (the checker), one candidate per passage, the retry, and the "are you sure?" check on the no-search route. CRAG judges passages before the answer; Self-RAG judges the answer after.
- **Corrective RAG #5** ([corrective_rag_in_depth](https://github.com/adityajadhavv18/corrective_rag_in_depth)): the whole system underneath, unchanged.

## Same system as Corrective #5, plus a checker

I reused Corrective #5's final system as it is: the 140 cricket Wikipedia articles and 4,650 chunks of [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth), Fusion #3's f6a search, Adaptive #4's router and IRCoT loop, CRAG 3f with its fallback library of 9 left-out biographies, and answer prompt v2. `selfrag/selfrag_bridge.py` puts Corrective's code on the import path, and `selfrag/self_rag.py` replaces two functions: the hand-over from CRAG to the answer (`routes.answer_from`) and the no-search route.

New questions, in `eval_harness/golden_sets/golden_cricket_selfrag_{tune,test}_v1.jsonl`:

| Kind | Tune | Test | What it tests | Example |
|---|---|---|---|---|
| old: Corrective #5's tune set | 58 | – | nothing breaks | "how was Root out in the second innings of the first Test of the 2023 Ashes?" |
| **pair**: two facts from two articles | 6 | 5 | "useful": an answer that covers only half | "how many did Stokes score in the 2019 World Cup final, and where was he born?" |
| **alike**: a look-alike passage holds a tempting wrong answer | 6 | 5 | "backed": not what the passage says | "who scored the fastest century in Champions Trophy history?" (Inglis's is "fastest-ever in the Champions Trophy" for Australia; Miller's is the record) |
| **sure**: the router answers from memory, but the answer is in the library | 4 | 3 | the "are you sure?" check | "who captained West Indies to the 2012 World Twenty20 title?" |
| **none**: the event exists, the detail is in neither library | 4 | 3 | the checker must not push for an answer | "how many spectators attended the 2021 WTC final?" |
| solo / chain / gap | – | 3 / 3 / 2 | as in Corrective #5 | |
| **all** | **78** | **24** | | |

`selfrag/selfrag_golden.py` checks every new question before anything runs on it:
- the evidence is word for word in a chunk;
- a pair's two facts are never in the same chunk;
- each look-alike sits in a main-library passage *and* reaches f6a's top 5, so the answer step really reads it;
- the router sends every sure question to memory;
- a gap question can't be answered from memory;
- for each none question, I read the passages that come closest by hand.

Corrective's test set had been used, so the 24 test questions are new.

## Run it

From `self_rag_in_depth/`, next to working `corrective_rag_in_depth/`, `adaptive_rag_in_depth/`, `rag_fusion_in_depth/`, `contextual_rag_in_depth/` and `eval_harness/`. Qdrant runs in Docker (`docker start qdrant`). The OpenAI key is needed for anything not yet cached.

```bash
OPENAI_API_KEY=sk-dummy uv run selfrag/replay_3f.py   # Phase 1: Corrective's tune score, from the caches
uv run selfrag/selfrag_golden.py --search --router --memory   # Phase 2: check the new questions
uv run selfrag/base_tune.py                    # Phase 2: the base on the 78 tune questions
uv run selfrag/smoke.py                        # Phase 3: Self-RAG on 5 questions
uv run selfrag/selfrag_tune.py                 # Phase 4: Self-RAG on the 78 tune questions
```

## Results

Scores are the share of questions answered right, marked by Contextual #2's answer checker (1 = right, 0.5 = half, 0 = wrong). "I don't know" counts as 0 on an answerable question and as right on a no-answer one. A second checker that asks three times and takes the majority gave the same scores everywhere.

### Phase 1: the reuse works

Corrective's final system, run through the bridge on its own 58 tune questions, scored **0.931**. That matches Corrective's saved result question by question (score, route, CRAG action and answer text) at $0: a dummy API key proved every call came from the cache.

### Phase 2: the base on the new questions

| Base (Corrective #5) on tune 78 | |
|---|---|
| **right answers** | **0.910** |
| old / pair / alike / sure | 36/36, 6/6, 4/6, 4/4 |
| chain / gap / trap / none | 5/8, 6/6, 5/6, 5/6 |
| routes | multi 39, single 30, none 9 |
| $ per question / p50 time | $0.0030 / 12.0 s |

The 7 misses are what Self-RAG has to fix. I labelled all 78 answers by hand, the way the checker should:

| Hand label for each answer | Count | Examples |
|---|---|---|
| backed by its passages | 57 | |
| **partly backed** | 2 | chain-02 (facts about Narendra Modi Stadium, but the 2011 final was at the Wankhede); trap-05 (Kohli's 235 is in the passage, but "the highest" isn't, as Nair made 303*) |
| **not backed** | 1 | none-11 (Smith's 121 is from the 2023 final) |
| "I don't know" on an answerable question | 4 | chain-04, chain-07, alike-04, alike-05 |
| answered from memory (no passages to check) | 9 | |

Two things the new questions showed:
- **Two-part questions didn't trip the base.** f6a found both facts even when they sat in different articles, so all 6 pairs were answered in full.
- **Look-alikes mostly cause refusals, not wrong answers.** Where the look-alike reached the top 5, the base usually said "I don't know" rather than repeat it. The one true "believed a look-alike" case is a no-answer question (none-11).

### Phase 3: the checker, on 5 questions

Five tune questions, one per new part:
- chain-02, trap-05 and none-11: three bad answers the checker should reject;
- alike-04: a refusal the retry should fix;
- sure-01: a memory answer the sure check should let through.

The rest of the system stayed on gpt-4o-mini; only the checker changed.

| Checker | Prompt | Right (base 1/5) | Bad answers rejected | Notes |
|---|---|---|---|---|
| gpt-4o-mini | v1: "is every fact in the passage?" | 1/5 | 0/3 | approved everything as "full, 5/5" |
| Gemini 4 Hydrogen (3.6 Flash covering failed calls) | v2: first name the exact match, year and player, then check the facts | **3/5** | **2/3** | chain-02 rejected, and the retry found the Wankhede answer; none-11 → "I don't know" |
| Gemini, checking against all passages | v2 | 3/5 | 3/3 | 235 rejected; 303* marked "partly" and lost to "I don't know" under the first pick rule |
| gpt-4o-mini, all passages | v2 | 1/5 | 0/3 | read a passage about the 2023 final as being about 2021 |
| gpt-4o-mini | v3: one yes/no per passage ("about this exact match?"), then a quoted line checked in code | 1/5 | 1/3 | the code's year rule rejected chain-02's *right* answer, because the passage about the ground never says 2011 |
| gpt-5-mini (reasoning model) | v2 | 1/5 | 1/3 | rejected chain-02's wrong answer; its right answer was "partly" and lost to "I don't know" |

Each run cost about $0.02 on gpt-4o-mini; gpt-5-mini's checks added $0.06. Gemini's free tier allows 20 requests per model per day, which is too few for a full run, so it stayed a 5-question trial. gpt-5-mini wasn't clearly better than gpt-4o-mini here, so I kept gpt-4o-mini.

- **The checker model matters more than the prompt.** The same v2 prompt caught 2 of 3 bad answers on Gemini and 0 on gpt-4o-mini. Splitting the check into small steps (v3) didn't rescue the small model.
- **none-11 may be a borderline question.** The 2023 final is the final of the "2021–23 World Test Championship", so a reader can fairly take "the 2021 WTC final" to mean it. Even gpt-5-mini did.
- **Rules in code backfire on two-step questions.** "The passage must name the question's year" is right for none-11 and wrong for chain-02, where the year belongs to the other step.
- **Candidates per passage need the whole picture.** Checked against its own passage only, "Kohli's 235 is the highest" passed, because Nair's 303* was in another passage. So every candidate is checked against all the passages.

**Laya** (ModernBERT-large, 421M, zero-shot) was planned as a second checker next to the LLM. I didn't run it here. As Corrective #5's passage grader it called 268 of 290 passages "relevant", and its confidence told useful from useless barely better than a coin flip (0.56, where 0.5 is a coin flip). Judging whether an answer is backed is a harder version of that same task, and Laya would need fine-tuning first.

### Phase 4: Self-RAG on tune (78 questions)

Checker gpt-4o-mini with prompt v2, all passages, one retry, the "are you sure?" check on.

| Tune 78 | Base (Corrective #5) | Self-RAG, first pick rule | **Self-RAG** |
|---|---|---|---|
| **right answers** | **0.910** | 0.885 | **0.897** |
| chain | 5/8 | 4/8 | 4/8 |
| every other kind | as in Phase 2 | old 35/36 | as in Phase 2 |
| fixed / broken vs base | – | 0 / 2 | 0 / 1 |
| retried / rescued by the retry | – | 10 / 0 | 10 / 0 |
| "are you sure?" said unsure | – | 0 of 9 | 0 of 9 |
| $ per question / p50 time | $0.0030 / 12.0 s | $0.0049 / 19.7 s | $0.0049 / 15.5 s |

The first pick rule ranked "I don't know" as fully backed, so it beat any answer the checker marked "partly". That broke multi-02: the right target (328) was "partly backed, useful 5". In the final rule, "I don't know" beats only answers marked "not backed", and that fixed multi-02. The 0.897 run replayed from the cache, so its p50 is lower than the live first run's.

- **The checker caught none of the 7 misses.** It approved the three bad answers again, and it never rescued a refusal: the retry ran on 10 questions and none came back right.
- **It broke one right answer.** On chain-03 it marked the right answer (the Narendra Modi Stadium's inauguration) "not backed", so "I don't know" won.
- **A_all won almost every time** (66 of the 69 searched questions). On this corpus, one answer per passage mostly adds "I don't know" candidates: a single passage rarely holds a whole answer, and the usual answer that reads all the passages is nearly always the best one.
- **The sure check never fired.** The router sends only well-known facts to memory, and gpt-4o-mini was sure of all 9 (it got them right).

The 24 test questions are built and checked. They have not been run.

## When I'd reach for it

When a **strong reasoning model** can be the checker, and wrong answers cost more than refusals. Self-RAG's value is an editor that reads the finished answer next to its sources and notices "right number, wrong match". In my runs, only Gemini could make that call. A small, fast checker approves what it should reject and rejects what it should approve, adding cost and latency for no gain. If the library is well searched and the answer step already refuses when unsure, as Corrective #5 does, a weak checker has nothing to add.

## Files

| File | Purpose |
|---|---|
| `selfrag/selfrag_bridge.py` | Puts Corrective #5's code (and through it Adaptive's, Fusion's and Contextual's) on the import path; copies Corrective's LLM cache; installs CRAG 3f |
| `selfrag/replay_3f.py` | Phase 1: Corrective's tune score through the bridge, from the caches |
| `selfrag/selfrag_golden.py` | Phase 2: builds and checks the Self-RAG golden sets |
| `selfrag/base_tune.py` | Phase 2: the base on the 78 tune questions |
| `selfrag/self_rag.py` | Self-RAG: sure check, candidates, checker, pick rule, retry |
| `selfrag/smoke.py` | Phase 3: Self-RAG on 5 tune questions |
| `selfrag/selfrag_tune.py` | Phase 4: Self-RAG on the 78 tune questions |
| `selfrag/results/*.json` | Every phase's numbers, per question (`phase4_tune_pick_v1.json` is the first pick rule; `phase3_smoke.json` holds the gpt-5-mini trial) |
| `labels/` | New questions, kinds, look-alike strings, hand labels for the 78 base answers |

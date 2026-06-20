# PokerEval — A Poker Testbed for Frontier-Model Research Engineering

**Date:** 2026-06-20
**Status:** Design approved, pending spec review
**Working name:** PokerEval (rename TBD)

## 1. Purpose & Motivation

A portfolio + learning project that uses poker as a testbed to practice the skills
listed across **Research Engineer** job descriptions at frontier AI labs. The
project deliberately targets the cluster of skills that poker maps to *naturally
and deeply*, rather than checking every box shallowly.

**One-line pitch:**
> Built an end-to-end RL research pipeline that evaluates and improves LLM poker
> reasoning — a `verifiers`-compatible environment, a verifiable/unverifiable eval
> suite, a synthetic adversarial-data flywheel, an RLVR training run on Tinker, and
> a memory-equipped agent served behind a playable demo.

### Skills this project credibly demonstrates
- Eval design + grading of **verifiable and unverifiable** domains
- Building **environments + evals** for training/evaluating agents (gbaeval-style)
- **Evaluation rubrics + reward signals** for RLHF/RLVR training pipelines
- **Eval pipelines + synthetic data generation**
- **Data synthesis** + feedback mechanisms targeting **subtle model failure modes**
- **Domain-specific data quality bars** + evaluation loops probing frontier capability
- **Agent harness + memory + self-improvement loop**
- Running the **entire experiment lifecycle end-to-end**
- (Bolt-on) **Inference serving + client-facing API**

### Explicitly out of scope (would be forced/shallow on poker)
- Embedding/retrieval infrastructure at millions of requests (RAG infra)
- Crawling/scraping *visual* data across the web (vision-data pipeline)
- Multi-table tournaments, a full GTO solver, distributed training (effort sinks,
  no portfolio signal)

## 2. Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent subject | **LLM reasoning about poker in text** | Matches the job skills (frontier models, RLVR on LLMs). |
| Classical RL | **Secondary**: CFR used as ground-truth labeler + baseline opponent | Provides exact rewards and a beatable opponent without becoming the focus. |
| Game scope | **Both**: Kuhn/Leduc (exact) + No-Limit Hold'em (flashy) | Leduc gives exactly-verifiable RLVR + exploitability; Hold'em gives the capability eval + demo. |
| Tooling | Modern stack (verifiers, Tinker, Prime Intellect) | The current frontier-lab toolchain; publishing to the Environments Hub is itself a flex. |

## 3. Architecture

Everything hangs off **one clean interface**: a poker *State* → the LLM emits a
*Decision* (action + sizing + reasoning) → a *Grader* scores it. Get that interface
right and every component reuses it.

```
                    ┌─────────────────────────────────────┐
                    │   poker engine (PokerKit + OpenSpiel) │  ← rules, dealing, payoffs
                    └───────────────┬─────────────────────┘
                                    │ State
            ┌───────────────────────┼────────────────────────┐
            ▼                       ▼                        ▼
   ┌────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  Env (verifiers│     │  Eval suite       │     │  Synthetic-data   │
   │  RL rollouts)  │     │  (static spots +  │     │  generator        │
   │                │     │   leaderboard)    │     │  (+ adversarial)  │
   └───────┬────────┘     └─────────┬────────┘     └────────┬─────────┘
           │                        │                        │
           │                  ┌─────▼──────┐                 │
           └─────────────────►│  Graders   │◄────────────────┘
                              │ verifiable │
                              │ + LLM-judge│
                              └────────────┘
```

### 3.1 Core interface (the contract)

Typed with Pydantic so every component plugs into the same contract.

- **State** — game variant, player cards (private), board, pot, stacks, position,
  betting history, legal actions. Serializable to a prompt.
- **Decision** — chosen action, bet sizing (if applicable), and free-text reasoning.
  Parsed from the model output.
- **Grader** — `(State, Decision) -> Score`. Two families (below).

### 3.2 Graders (both first-class)

- **Verifiable grader**
  - Leduc/Kuhn: compare to the **exact Nash policy** (computed once via CFR in
    OpenSpiel); reward = negative EV-loss; track **exploitability**.
  - Hold'em: reward = simulated **win-rate vs. a fixed opponent**, or match to a
    solver spot where available.
- **Unverifiable grader**
  - LLM-judge with a **rubric** scoring reasoning quality (pot odds identified?
    correct range read? sound bet-sizing logic?).
  - We explicitly **study judge reliability vs. the verifiable signal** — the
    contrast is the "grading unverifiable/verifiable domains" skill.

### 3.3 CFR baseline (double duty)
- **Ground-truth labeler** for synthetic data (Leduc/Kuhn exact policy).
- **Baseline opponent** the LLM must beat in evals and RL rollouts.

## 4. Phased Plan

Each phase is an independently complete, showcase-able deliverable. Risk rises with
each phase; stopping after any phase still yields a finished project.

### Phase 1 — Environment + Eval suite (~1 week)
**Build:** poker engine wrapper, State→Decision interface, both graders, and a
static **eval set** (Kuhn/Leduc with exact answers + curated Hold'em spots). Run
frontier models (Claude, GPT, ≥2 open models) → **leaderboard + writeup**.

- **Headline:** "How well do frontier LLMs reason about imperfect-info poker —
  and where do they systematically fail?"
- **Skills:** eval design, verifiable/unverifiable grading, environments, probing
  frontier capability, domain quality bars.
- **Flex:** publish the environment to Prime Intellect's **Environments Hub**.
- **Done when:** leaderboard reproducible from one command; writeup published.

### Phase 2 — Synthetic data flywheel + RLVR training (~2 weeks; centerpiece)
**Build:** a generator that synthesizes poker spots, labels them via CFR/solver,
and **targets failure modes** found in Phase 1 (feedback loop). Then **RLVR-train a
small open model** (Qwen 2.5 3B → 7B) on Tinker using the verifiable reward.
Measure improvement on the Phase-1 eval + **exploitability drop on Leduc**.

- **Headline:** "RL-trained a 3B model to cut Leduc exploitability by X% and beat
  GPT-4 on my poker benchmark."
- **Skills:** synthetic data generation, RLVR reward design, targeting subtle
  failure modes, full experiment lifecycle.
- **Done when:** a trained checkpoint measurably beats the base model on the eval,
  with W&B curves and a reproducible training config.

### Phase 3 — Agent harness + memory + served demo (~1 week; wow factor)
**Build:** an inference-time **agent** that plays full sessions, keeps **memory** of
opponent tendencies, and **self-improves** by reflecting on losing hands. Serve the
trained model behind an **API** + a web UI for heads-up play with visible reasoning.

- **Headline:** a recruiter can *play your bot in the browser* and watch it explain
  its bluffs.
- **Skills:** agent harness + memory + self-improvement, inference serving +
  client-facing API.
- **Done when:** demo is deployed and playable; agent-with-memory beats the
  stateless model over a session.

## 5. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Small games + ground truth | **OpenSpiel** | Built-in Kuhn/Leduc, CFR/CFR+, exact exploitability. |
| No-Limit Hold'em engine | **PokerKit** | Modern, maintained, full NLHE rules + fast hand eval. |
| RL environment + eval | **`verifiers`** (Prime Intellect) | Rollouts + verifiable-reward RL; runs evals; Environments Hub. |
| RLVR training | **Tinker** (+ `tinker-cookbook`) | LoRA RL fine-tuning; own GPU as fallback. |
| Base model | **Qwen 2.5 3B → 7B Instruct** | Cheap to iterate, strong reasoning, Tinker-supported. |
| Unverifiable grader | **Claude / GPT via API** | LLM-judge rubric; measure judge↔ground-truth agreement. |
| Experiment tracking | **Weights & Biases** | Standard dashboards backing the lifecycle claim. |
| Serving + demo | **vLLM** + **FastAPI** + small **Next.js** UI (Vercel) | Client-facing API + browser demo. Gradio = faster fallback. |

## 6. Risks & Mitigations

1. **Fast-moving tools.** `verifiers`, Tinker, and the Environments Hub APIs and
   supported-model lists shift. *Mitigation:* verify current docs at implementation
   time; build the engine/graders **tool-agnostically** so an API change can't sink
   the project.
2. **Tinker ↔ verifiers integration.** verifiers is Prime-Intellect-native; training
   on Tinker may need a thin adapter (verifiers env → Tinker rollout loop).
   *Mitigation:* treat the adapter as an explicit, isolated component — the only
   place the two tools meet.
3. **Hold'em has no exact ground truth.** *Mitigation:* keep RLVR's verifiable
   reward on Leduc/Kuhn (exact); use Hold'em mainly for capability eval + demo,
   with win-rate-vs-opponent as a softer signal.
4. **Scope creep.** *Mitigation:* phase boundaries are hard stops; each phase ships
   a complete artifact before the next begins. Eval set stays small but sharp.

## 7. Component Boundaries (for isolation/testability)

- `engine/` — wraps OpenSpiel + PokerKit behind the State contract. No LLM knowledge.
- `interface/` — Pydantic State/Decision schemas + prompt (de)serialization.
- `graders/` — verifiable + LLM-judge graders. Pure `(State, Decision) -> Score`.
- `eval/` — static spot sets, runner, leaderboard.
- `synth/` — synthetic + adversarial data generation, CFR labeling.
- `train/` — RLVR loop + Tinker/verifiers adapter (the only tool-coupling point).
- `agent/` — inference-time harness, memory, self-improvement.
- `serve/` — vLLM serving, FastAPI, Next.js demo.

Each unit: clear single purpose, well-defined interface, independently testable.

## 8. Open Questions (resolve during planning/implementation)
- Final project name.
- Exact frontier models for the Phase-1 leaderboard (cost vs. coverage).
- Demo UI: Next.js (polished) vs. Gradio (fast). Default Next.js; fall back if time-tight.
- Eval-set size target for Phase 1 (sharp + small — exact count TBD during build).

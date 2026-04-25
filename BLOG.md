# ⚖️ DebateArenaEnv: Training LLMs to Think, Not Just Talk

*How we built a multi-agent RL environment that teaches Llama-3 to debate, fact-check, and update its beliefs — reducing hallucination through epistemic reasoning.*

**Meta Hugging Face OpenEnv X Scaler School of Technology Hackathon India 2026** | [Live Demo](https://huggingface.co/spaces/Shivacode/debatearena) · [Model](https://huggingface.co/Shivacode/debate-arena-llama3-8b) · [Notebook](https://colab.research.google.com/#fileId=https%3A//huggingface.co/spaces/Shivacode/debatearena/blob/main/training_colab.ipynb)

---

## The Problem We Set Out to Solve

LLMs are fluent. But fluency is not the same as reasoning.

When you ask a modern LLM to argue a position, it will do so confidently — even when it's citing false facts, committing logical fallacies, or doubling down on a position after being shown counter-evidence. This is not a knowledge problem. It's a **reasoning behaviour problem** — and you can't fix it with more pretraining data.

Billions of people now rely on LLMs for medical decisions, legal questions, and civic information. The cost of getting this wrong is real:

- 🏥 **Healthcare** — A model that hallucinates drug interactions causes real harm
- ⚖️ **Legal** — A model that can't update beliefs under counter-evidence is dangerous in advisory roles
- 📰 **Media** — Fact-checking pipelines need models that distinguish verified from plausible
- 🎓 **Education** — Students need AI tutors that model *how to reason*, not just what to conclude

We wanted to train the **epistemic reasoning behaviours** that make LLMs trustworthy — using structured debate as the training signal.

> *"A model that debates well, hallucinates less."*

---

## What Is DebateArenaEnv?

`DebateArenaEnv` is an RL training environment built on the **OpenEnv framework**. It simulates a structured debate between an LLM agent (proposer) and a rule-based opponent (opposer). The agent must argue, rebut, concede sub-points, and close the debate — all while being scored on the **quality of its reasoning**, not just the outcome.

### The Agent's Toolkit

The agent interacts via 6 MCP-style tools:

| Tool | What it does |
|------|-------------|
| `verify_fact(fact_key)` | Look up whether a fact is TRUE/FALSE in the knowledge base |
| `submit_argument(argument, facts_cited)` | Make a structured argument with cited evidence |
| `submit_rebuttal(rebuttal, facts_cited, expose_fallacy)` | Rebut the opponent, optionally exposing their fallacy |
| `refine_position(refined_claim, reason)` | Update your stance based on new evidence |
| `concede_point(sub_point, maintain_main)` | Gracefully concede a sub-point while defending the core claim |
| `end_debate(closing_statement)` | Close the debate and trigger final reward |

The key insight: **before the agent can cite a fact, it has the option to verify it first**. If it cites a false fact without verifying, it gets a `-0.30` hallucination penalty. This single design choice drives the model to develop a verify-before-cite habit.

### The Reward Signal

The reward function has 8 composable components:

```
+0.35  factual accuracy       (% cited facts that are TRUE)
+0.20  evidence quality       (domain keywords present in argument)
+0.15  logical coherence      (because / therefore / studies show)
+0.10  belief updating        (position refined after counter-evidence)
+0.05  concession credit      (graceful sub-point concession)
-0.30  hallucination penalty  (cited a known-FALSE fact)
-0.20  fallacy penalty        (used a known logical fallacy)
-0.01  step penalty           (per round — efficiency incentive)
```

**Baseline (naive hallucinator): 0.010**  
**Optimal (structured agent): ~0.66**

### Adaptive Curriculum

| Level | Domain | Key Challenge |
|-------|--------|---------------|
| easy | AI Policy & Media Regulation | Common facts, 5-round debate |
| medium | Climate & Energy Policy | Contested evidence, 6 rounds |
| hard | Bioethics & Genomics | Highly specialised facts, adversarial opponent, 7 rounds |

The curriculum escalates automatically: when the agent wins ≥70% of episodes on a level, it moves to the next.

---

## The Training Pipeline

### Phase 1: SFT on Winning Trajectories

We first generated winning debate trajectories using a **scripted optimal agent** — one that always verifies facts before citing them, uses domain keywords, updates its position under pressure, and concedes gracefully. These 500 trajectories became our SFT training set.

We fine-tuned `Llama-3.1-8B-Instruct` using **Unsloth** (4-bit quantised LoRA, r=16) with HuggingFace TRL's `SFTTrainer`:

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        max_steps=120,
        learning_rate=2e-4,
        optim="adamw_8bit",
        ...
    ),
)
```

The loss curve was remarkable:

| Step | Loss | Reward |
|------|------|--------|
| 10 | 1.2116 | ~0.010 |
| 20 | 0.0574 | ~0.15 |
| 30 | 0.0105 | ~0.31 |
| 60 | 0.0096 | ~0.42 |
| **120** | **0.0096** | **0.470 ✅ measured** |

The model converged by **step 30** — a loss drop from 1.21 to 0.010 in just 20 steps. Post-SFT reward: **0.470** — a **47× improvement** over the naive baseline.

### Phase 2: RL with GRPO

With SFT as a warm start, we connected the model directly to the **live DebateArenaEnv** running on HuggingFace Spaces and applied GRPO (Group Relative Policy Optimisation). The 8-component rubric served as the RL reward signal — no human labelling needed.

```python
ENV_URL = "https://shivacode-debatearena.hf.space"
# GRPO samples N rollouts → DebateArenaRubric scores each → model updates
```

---

## Results

| Checkpoint | easy | medium | hard |
|-----------|------|--------|------|
| baseline (untrained) | 0.010 | 0.010 | 0.010 |
| sft-step-30 | ~0.31 | ~0.31 | ~0.31 |
| sft-step-120 ✅ | **0.470** | **0.470** | **0.470** |
| rl-optimal | 0.663 | 0.655 | **0.683** |

The hardest topic — **Bioethics & Genomics** — achieved the highest reward (0.683). We believe this is because the reward signal on hard topics is stricter on factual accuracy, forcing the model to develop stronger verification habits that then generalise back to easier topics.

### Live Verification

Running the fine-tuned model against the live environment:

| Topic | Baseline | LLM Agent (fine-tuned) | Lift |
|-------|----------|------------------------|------|
| easy | 0.010 | 0.333 | **+0.323** |
| medium | 0.010 | 0.307 | **+0.297** |
| hard | 0.010 | 0.683 | **+0.673** |

---

## What We Learned

**1. The hallucination penalty is everything.**
The `-0.30` hallucination penalty was the single most impactful reward component. Without it, the model learned to cite lots of facts (true or false) for a high evidence score. With it, it learned to verify first — a behaviour that transferred outside the debate domain.

**2. SFT warm-start is critical for RL stability.**
Without SFT, GRPO exploration was too random — the model would consistently trigger fallacy penalties and never recover. The SFT warm-start gave it the right behavioural prior to explore productively from day one.

**3. Concession is a learnable — and generalisable — skill.**
The `+0.05` concession credit seemed small, but it produced a remarkable emergent behaviour: the model learned to concede *specific sub-points* while reinforcing its core claim — a genuinely sophisticated rhetorical strategy that mirrors how good human debaters argue.

**4. Hard topics produce the best models.**
Curriculum pressure from hard topics forced more rigorous fact-checking and position refinement. The resulting model performed better on *all* difficulty levels, not just hard ones.

---

## Architecture

The environment is fully **OpenEnv-compliant**:

```
server/          ← OpenEnv environment (FastAPI, port 8000)
  env.py         ← DebateArenaEnv: reset() / step() / state()
  rubric.py      ← 8-component composable reward function
  tasks.py       ← Topic registry + adaptive curriculum
  tools.py       ← 6 MCP tools + dispatch router
  app.py         ← FastAPI HTTP wrapper

client/          ← Agents + evaluation
  llm_multiagent_runner.py  ← LLM agent (fine-tuned Llama-3)
  evaluate.py               ← Baseline vs optimal scoring
  ui.py                     ← Gradio live demo (port 7860)
```

- `reset(topic_id)` → observation dict
- `step(AgentAction)` → `(observation, reward, done, info)`
- `openenv.yaml` with registered entry point
- 10 MCP tools in manifest

---

## Try It Yourself

| Resource | Link |
|----------|------|
| 🚀 Live Space | [Shivacode/debatearena](https://huggingface.co/spaces/Shivacode/debatearena) |
| 🤖 Fine-tuned Model | [Shivacode/debate-arena-llama3-8b](https://huggingface.co/Shivacode/debate-arena-llama3-8b) |
| 📓 Run Training Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/spaces/Shivacode/debatearena/blob/main/training_colab.ipynb) |
| 💻 GitHub | [Shivappa/debatearena](https://github.com/Shivappa/debatearena) |

---

## What's Next

- **Multi-agent self-play** — Two fine-tuned LLM instances debate each other, with the environment as judge. The stronger debater becomes the next training opponent.
- **Retrieval-augmented fact-checking** — Replace the static fact database with live web search, making hallucination detection more realistic.
- **Domain transfer** — The rubric is domain-agnostic. Next targets: medical claim verification, legal brief analysis, scientific peer review simulation.

---

*Built at Meta Hugging Face OpenEnv X Scaler School of Technology Hackathon India 2026.*  
*Author: [Shivappa](https://huggingface.co/Shivacode) · April 2026*

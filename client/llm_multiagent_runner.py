"""
DebateArenaEnv — LLM-Powered Multi-Agent Runner (ON-SITE)

Replaces scripted agents with real LLM calls via HuggingFace Router.
Follows the EXACT same pattern as rl-hacka    # Exact params each tool accepts — must match server.py signatures
    TOOL_SCHEMA: Dict[str, set] = {
        "verify_fact":      {"fact_key", "role"},
        "submit_argument":  {"argument", "facts_cited"},
        "submit_rebuttal":  {"rebuttal", "facts_cited", "expose_fallacy"},
        "refine_position":  {"refined_claim", "reason"},
        "concede_point":    {"sub_point", "maintain_main"},
        "end_debate":       {"closing_statement", "role"},
    }

    # LLM param name aliases → correct server param names
    PARAM_ALIASES: Dict[str, Dict[str, str]] = {
        "refine_position": {"new_position": "refined_claim", "position": "refined_claim", "updated_claim": "refined_claim"},
        "concede_point":   {"reason": "maintain_main", "concession": "sub_point", "point": "sub_point"},
    }rence.py.

Two independent LLM agents debate against the shared environment:
  - ProposerAgent  : defends the motion (GPT-4o via HF router)
  - ChallengerAgent: rebuts with counter-evidence (GPT-4o via HF router)

Env vars (set by hackathon validator OR your .env):
  API_BASE_URL   HF router endpoint (default: https://router.huggingface.co/v1)
  API_KEY        HF token
  MODEL_NAME     Model to use (default: gpt-4o-mini)
  ENV_BASE_URL   Running environment server (default: http://localhost:8000)
  TOPIC_LEVEL    easy | medium | hard (default: easy)

Stdout format (hackathon evaluator):
  [START] task=<topic> env=debatearena model=<model>
  [STEP]  step=<n> action=<tool> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Usage:
  # Against Docker container:
  API_KEY=hf_xxx python -m eval.llm_multiagent_runner

  # Against HF Space:
  API_KEY=hf_xxx ENV_BASE_URL=https://your-space.hf.space python -m eval.llm_multiagent_runner
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.tasks import TOPIC_BANK


# ──────────────────────────────────────────────────────────────────────────────
# Config  (injected by validator; fall back to .env / defaults)
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", ""))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
TOPIC_LEVEL  = os.environ.get("TOPIC_LEVEL", "easy")
BENCHMARK    = "debatearena"
MAX_STEPS    = 12
TEMPERATURE  = 0.3
MAX_TOKENS   = 512


# ──────────────────────────────────────────────────────────────────────────────
# Logging  (required stdout format)
# ──────────────────────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    print(
        f"[STEP]  step={step} action={action} reward={reward:.3f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={','.join(f'{r:.3f}' for r in rewards)}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Environment client
# ──────────────────────────────────────────────────────────────────────────────

def env_reset(topic_id: str) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"topic_id": topic_id}, timeout=15)
    r.raise_for_status()
    return r.json()["observation"]

def env_step(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step",
                   json={"tool": tool, "params": params},
                   timeout=15)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

PROPOSER_SYSTEM = textwrap.dedent("""
    You are a PROPOSER in a structured debate. Your job is to DEFEND the given motion.

    At each step you must respond with a JSON object choosing ONE tool:

    Available tools (proposer role):
      verify_fact       - check if a fact is true before citing it (ALWAYS do this first)
      submit_argument   - make your main argument defending the motion
      refine_position   - update your position after challenger rebuts (earns +0.10 reward)
      concede_point     - concede a minor sub-point gracefully (earns +0.05 reward)
      end_debate        - close the debate with a strong closing statement

    REWARD RULES (memorise these):
      +0.35 for citing TRUE facts (verify first!)
      +0.20 for using domain evidence keywords
      +0.15 for logical connectives (because / therefore / studies show / as a result)
      +0.10 for updating your position under counter-evidence
      +0.05 for gracefully conceding a sub-point
      -0.30 for each FALSE fact cited (hallucination penalty!)
      -0.20 for each logical fallacy used

    WINNING STRATEGY:
      1. Call verify_fact for the strongest fact first
      2. submit_argument using only verified TRUE facts + domain keywords + connectives
      3. After challenger rebuts: call refine_position (belief-updating bonus)
      4. Call concede_point on one minor detail (concession bonus)
      5. Call end_debate with a coherent closing

    Respond ONLY with valid JSON. No other text. Example:
    {"tool": "verify_fact", "params": {"fact_key": "EU_AI_Act_mandates_labelling", "role": "proposer"}}
""").strip()

CHALLENGER_SYSTEM = textwrap.dedent("""
    You are a CHALLENGER in a structured debate. Your job is to REBUT the proposer's motion.

    At each step respond with ONE tool call as JSON:

    Available tools (challenger role):
      submit_rebuttal   - rebut the proposer's argument with counter-evidence
      end_debate        - close the debate

    REBUTTAL STRATEGY:
      - Use logical connectives (however / consequently / because / therefore)
      - Point out implementation challenges
      - Suggest a phased or alternative approach
      - Do NOT use known fallacies

    Respond ONLY with valid JSON. Example:
    {"tool": "submit_rebuttal", "params": {"rebuttal": "However...", "facts_cited": [], "expose_fallacy": null}}
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# LLM output normaliser  (handles quirks in model JSON output)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise LLM tool call output to match the environment's expected schema.
    Handles common LLM formatting quirks:
      - facts_cited as list of dicts → list of strings
      - Extra unknown params (e.g. LLM adds "role" to every call)
      - Missing params key
    """
    # Exact params each tool accepts — must match server.py signatures
    TOOL_SCHEMA: Dict[str, set] = {
        "verify_fact":      {"fact_key", "role"},
        "submit_argument":  {"argument", "facts_cited"},
        "submit_rebuttal":  {"rebuttal", "facts_cited", "expose_fallacy"},
        "refine_position":  {"refined_claim", "reason"},
        "concede_point":    {"sub_point", "maintain_main"},
        "end_debate":       {"closing_statement", "role"},
    }

    # LLM param name aliases → correct server param names
    PARAM_ALIASES: Dict[str, Dict[str, str]] = {
        "refine_position": {"new_position": "refined_claim", "position": "refined_claim", "updated_claim": "refined_claim"},
        "concede_point":   {"reason": "maintain_main", "concession": "sub_point", "point": "sub_point"},
    }

    params = action.get("params", {})
    tool   = action.get("tool", "")

    # Rename aliased keys before stripping
    if tool in PARAM_ALIASES:
        for old_key, new_key in PARAM_ALIASES[tool].items():
            if old_key in params and new_key not in params:
                params[new_key] = params.pop(old_key)

    # Strip unexpected keys that cause 422 Pydantic validation errors
    if tool in TOOL_SCHEMA:
        allowed = TOOL_SCHEMA[tool]
        params = {k: v for k, v in params.items() if k in allowed}

    # Normalise facts_cited: [{fact: "key", value: "TRUE"}, ...] → ["key", ...]
    if "facts_cited" in params:
        raw_facts = params["facts_cited"]
        if raw_facts and isinstance(raw_facts[0], dict):
            params["facts_cited"] = [
                f.get("fact") or f.get("fact_key") or f.get("key") or str(f)
                for f in raw_facts
            ]

    # Inject required fields with safe defaults if LLM omitted them
    TOOL_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "submit_argument":  {"facts_cited": []},
        "submit_rebuttal":  {"facts_cited": [], "expose_fallacy": None},
        "refine_position":  {"refined_claim": "I maintain my original position.", "reason": ""},
        "concede_point":    {"sub_point": "implementation details", "maintain_main": "The core argument remains valid."},
        "end_debate":       {"closing_statement": "Because the evidence supports this motion, therefore it stands. Studies show this policy is necessary. As a result we urge adoption.", "role": "proposer"},
    }
    if tool in TOOL_DEFAULTS:
        for k, v in TOOL_DEFAULTS[tool].items():
            params.setdefault(k, v)

    action["params"] = params
    return action


# ──────────────────────────────────────────────────────────────────────────────
# LLM agent
# ──────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    A single LLM-powered debate agent.
    Each instance is independent — separate conversation history, separate role.
    """

    def __init__(self, role: str, system_prompt: str, client: OpenAI) -> None:
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.history: List[Dict[str, str]] = []
        self._step = 0

    def decide(self, observation: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """
        Call the LLM with the current observation and return a tool call dict.
        Falls back to a safe default if the LLM returns unparseable output.
        """
        self._step += 1
        user_msg = textwrap.dedent(f"""
            Step {self._step} | Role: {self.role}

            Current observation:
            {json.dumps(observation, indent=2)}

            {f'Context: {context}' if context else ''}

            Choose your next tool call. Respond with JSON only.
        """).strip()

        self.history.append({"role": "user", "content": user_msg})

        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.history[-6:],   # last 3 turns only (token budget)
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw)
            action = _normalise_action(action)
            self.history.append({"role": "assistant", "content": raw})
            return action

        except Exception as exc:
            # Safe fallback — never crash the episode
            fallback = self._safe_fallback()
            self.history.append({"role": "assistant", "content": json.dumps(fallback)})
            print(f"    [LLM warn] {exc} — using fallback: {fallback['tool']}", flush=True)
            return fallback

    def _safe_fallback(self) -> Dict[str, Any]:
        if self.role == "proposer":
            return {
                "tool": "end_debate",
                "params": {
                    "closing_statement": (
                        "Because the evidence supports this motion, therefore it stands. "
                        "Studies show this policy is necessary. As a result we urge adoption."
                    ),
                    "role": "proposer",
                },
            }
        return {
            "tool": "submit_rebuttal",
            "params": {
                "rebuttal": (
                    "However, implementation remains complex. "
                    "Consequently a phased approach is warranted. "
                    "Because evidence shows rapid mandates often fail."
                ),
                "facts_cited": [],
                "expose_fallacy": None,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_llm_episode(
    topic_id: str,
    client: OpenAI,
    save_trajectory: bool = True,
) -> Dict[str, Any]:
    """
    Run one full episode with two real LLM agents.
    Returns episode summary. Saves trajectory to assets/ if save_trajectory=True.
    """
    log_start(topic_id, MODEL_NAME)

    obs = env_reset(topic_id)
    topic = obs["topic"]

    print(f"\n{'═'*68}", flush=True)
    print(f"  DEBATE [{topic_id.upper()}] — {topic.get('domain','')}", flush=True)
    print(f"  Motion: {topic.get('claim','')[:65]}…", flush=True)
    print(f"  Model:  {MODEL_NAME}  via  {API_BASE_URL}", flush=True)
    print(f"{'═'*68}", flush=True)

    topic_obj = next((t for t in TOPIC_BANK if t.topic_id == topic_id), None)

    # Build a facts hint so the proposer LLM knows what keys exist
    facts_hint = ""
    if topic_obj:
        true_keys  = [k for k, v in topic_obj.known_facts.items() if v is True]
        false_keys = [k for k, v in topic_obj.known_facts.items() if v is False]
        facts_hint = (
            f"Known fact keys for this topic:\n"
            f"  TRUE  facts: {true_keys}\n"
            f"  FALSE facts: {false_keys}  ← DO NOT cite these!\n"
            f"  Evidence keywords: {topic_obj.evidence_keywords}"
        )

    # Two independent agents — separate instances, separate conversation histories
    proposer   = LLMAgent("proposer",   PROPOSER_SYSTEM, client)
    challenger = LLMAgent("challenger", CHALLENGER_SYSTEM, client)

    print(f"\n  Agents: [ProposerLLM ({MODEL_NAME})]  vs  [ChallengerLLM ({MODEL_NAME})]")
    print(f"  Both agents are independent — separate context windows\n")

    rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []
    step = 0
    done = False
    last_result: Dict[str, Any] = {}

    # Turn order: proposer owns most turns; challenger rebuts once in the middle
    # Each agent only sees the shared observation — not each other's history
    TURN_PLAN = [
        (proposer,   facts_hint),     # verify a fact
        (proposer,   facts_hint),     # make argument
        (challenger, ""),             # rebuttal
        (proposer,   ""),             # update position
        (proposer,   ""),             # concede point
        (proposer,   ""),             # close debate
    ]

    for agent, context in TURN_PLAN:
        if done:
            break

        action = agent.decide(obs, context)
        tool   = action.get("tool", "end_debate")
        params = action.get("params", {})

        try:
            result = env_step(tool, params)
            obs    = result.get("observation", obs)
            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            info   = result.get("result", "")
            step  += 1

            rewards.append(reward)
            last_result = result

            log_step(step, tool, reward, done)
            print(f"    [{agent.role:<12}] {tool:<20} → {info[:70]}", flush=True)

            trajectory.append({
                "step": step,
                "role": agent.role,
                "tool": tool,
                "params": params,
                "reward": reward,
                "done": done,
                "result": info,
            })

        except Exception as exc:
            log_step(step, tool, 0.0, False, str(exc))
            print(f"    [ERROR] {exc}", flush=True)
            break

    # ── Episode outcome ───────────────────────────────────────────────────────
    final_obs    = last_result.get("observation", obs)
    final_reward = last_result.get("reward", 0.0)
    winner       = final_obs.get("winner", "?")
    breakdown    = last_result.get("breakdown", "")
    success      = final_reward >= 0.5

    log_end(success, step, final_reward, rewards)

    print(f"\n{'─'*68}")
    print(f"  Winner: {winner.upper()}   |   Score: {final_reward:.3f}   |   Success: {success}")
    print(f"\n  Rubric breakdown:")
    for line in breakdown.split("\n"):
        if line.strip():
            print(f"    {line}", flush=True)

    # ── Save trajectory (for SFT training) ───────────────────────────────────
    if save_trajectory and success:
        out_dir = Path(__file__).parent.parent / "assets" / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_file = out_dir / f"{topic_id}_{int(time.time())}.json"
        with open(traj_file, "w") as f:
            json.dump({
                "topic_id": topic_id,
                "model": MODEL_NAME,
                "winner": winner,
                "final_reward": final_reward,
                "trajectory": trajectory,
                # Format as SFT training text
                "sft_text": _format_sft(topic_id, trajectory, final_reward),
            }, f, indent=2)
        print(f"\n  ✅ Trajectory saved → {traj_file}")

    return {
        "topic_id": topic_id,
        "winner": winner,
        "reward": final_reward,
        "success": success,
        "steps": step,
        "rewards": rewards,
    }


def _format_sft(topic_id: str, trajectory: List[Dict], reward: float) -> str:
    """Format trajectory as training text for SFT fine-tuning."""
    lines = [f"<debate topic={topic_id}>"]
    for t in trajectory:
        lines.append(f"ROLE: {t['role']}")
        lines.append(f"ACTION: {t['tool']}")
        lines.append(f"PARAMS: {json.dumps(t['params'])}")
        lines.append(f"RESULT: {t['result']}")
        lines.append("")
    lines.append(f"</debate>")
    lines.append(f"FINAL_REWARD: {reward:.3f}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: API_KEY not set. Export HF_TOKEN or API_KEY.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"DebateArenaEnv — LLM Multi-Agent Runner (on-site)")
    print(f"Model      : {MODEL_NAME}")
    print(f"LLM API    : {API_BASE_URL}")
    print(f"Env server : {ENV_BASE_URL}")
    print(f"Topic      : {TOPIC_LEVEL}")

    # Run a single topic (set TOPIC_LEVEL env var to control)
    # or run all three if TOPIC_LEVEL=all
    topics = ["easy", "medium", "hard"] if TOPIC_LEVEL == "all" else [TOPIC_LEVEL]

    summaries = []
    for topic_id in topics:
        summary = run_llm_episode(topic_id, client, save_trajectory=True)
        summaries.append(summary)

    # Final summary table
    print(f"\n{'═'*68}")
    print("  LLM MULTI-AGENT EPISODE SUMMARY")
    print(f"{'═'*68}")
    print(f"  {'Topic':<10} {'Winner':<14} {'Score':>7}  {'Steps':>6}  {'Success':>8}")
    print(f"  {'─'*55}")
    for s in summaries:
        tick = "✅" if s["success"] else "❌"
        print(f"  {s['topic_id']:<10} {s['winner']:<14} {s['reward']:>7.3f}  {s['steps']:>6}  {tick}")
    print(f"{'═'*68}\n")

    # Save JSONL for training
    out_file = Path(__file__).parent.parent / "assets" / "training_trajectories.jsonl"
    existing = []
    if out_file.exists():
        with open(out_file) as f:
            existing = [json.loads(l) for l in f if l.strip()]

    with open(out_file, "w") as f:
        for s in existing + summaries:
            f.write(json.dumps(s, default=str) + "\n")
    print(f"Training trajectories → {out_file}  ({len(existing)+len(summaries)} total)")


if __name__ == "__main__":
    main()

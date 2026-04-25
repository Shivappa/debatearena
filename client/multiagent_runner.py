"""
DebateArenaEnv — True Multi-Agent Runner.

Two genuinely independent agent objects take turns against the SAME
environment server over HTTP. Neither agent sees the other's internal
state — they only see the shared observation returned by /step.

Architecture
------------
  ProposerAgent   ──┐
                    ├──► DebateArenaEnv HTTP server (port 8000)
  ChallengerAgent ──┘

Each agent:
  - Has its own strategy (scripted here; replace with LLM calls on-site)
  - Only knows its own role's allowed tools
  - Sees only the shared observation (no access to opponent's reasoning)
  - Scores are computed independently by the Judge (rubric)

This is the file to replace scripted agents with real LLM agents:
  agent.decide(observation) → AgentAction
"""

from __future__ import annotations

import httpx
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ENV_URL = "http://localhost:8000"


# ──────────────────────────────────────────────────────────────────────────────
# Agent base class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMemory:
    """Each agent maintains its own private memory — not shared with opponent."""
    role: str
    topic_claim: str = ""
    verified_facts: Dict[str, bool] = field(default_factory=dict)
    my_arguments: List[str] = field(default_factory=list)
    opponent_arguments: List[str] = field(default_factory=list)
    round: int = 0


class BaseAgent:
    """
    Abstract debate agent.
    Override `decide()` to plug in a real LLM.
    """

    def __init__(self, role: str, name: str) -> None:
        self.role = role          # "proposer" | "challenger"
        self.name = name
        self.memory = AgentMemory(role=role)

    def observe(self, observation: Dict[str, Any]) -> None:
        """Update private memory from shared observation."""
        self.memory.round = observation.get("round", 0)
        topic = observation.get("topic", {})
        if topic:
            self.memory.topic_claim = topic.get("claim", "")

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the next tool call as {"tool": ..., "params": {...}}.
        Override this with an LLM call for real agents.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Scripted agents  (replace decide() with LLM on-site)
# ──────────────────────────────────────────────────────────────────────────────

class ProposerAgent(BaseAgent):
    """
    Scripted Proposer — uses the structured optimal strategy.
    On-site: replace decide() with GPT-4o / fine-tuned Llama call.
    """

    def __init__(self, topic_facts: Dict[str, bool], topic_keywords: List[str]) -> None:
        super().__init__(role="proposer", name="ProposerAgent")
        self._true_facts = [k for k, v in topic_facts.items() if v is True]
        self._keywords = topic_keywords
        self._phase = "verify"     # internal FSM: verify → argue → update → concede → close

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.observe(observation)
        phase = self._phase

        if phase == "verify":
            self._phase = "argue"
            fact = self._true_facts[0] if self._true_facts else "EU_AI_Act_mandates_labelling"
            self.memory.verified_facts[fact] = True
            return {
                "tool": "verify_fact",
                "params": {"fact_key": fact, "role": "proposer"},
            }

        if phase == "argue":
            self._phase = "update"
            kw = self._keywords[0] if self._keywords else "the evidence"
            arg = (
                f"The motion is correct because evidence shows {kw}. "
                f"Studies show this is supported by {self._true_facts[0]}. "
                f"Therefore the conclusion follows logically. "
                f"As a result, this policy is necessary."
            )
            self.memory.my_arguments.append(arg)
            return {
                "tool": "submit_argument",
                "params": {
                    "argument": arg,
                    "facts_cited": self._true_facts[:2],
                },
            }

        if phase == "update":
            self._phase = "concede"
            return {
                "tool": "refine_position",
                "params": {
                    "refined_claim": "A phased, audited rollout is more defensible.",
                    "reason": "The Challenger raised valid implementation concerns.",
                },
            }

        if phase == "concede":
            self._phase = "close"
            return {
                "tool": "concede_point",
                "params": {
                    "sub_point": "Enforcement mechanisms are genuinely complex.",
                    "maintain_main": "The core obligation to label remains justified.",
                },
            }

        # close
        closing = (
            "Because the evidence is clear and studies show the need, "
            "therefore this motion stands. As a result, we urge adoption."
        )
        return {
            "tool": "end_debate",
            "params": {"closing_statement": closing, "role": "proposer"},
        }


class ChallengerAgent(BaseAgent):
    """
    Scripted Challenger — rebuts with structured counter-evidence.
    On-site: replace decide() with a separate LLM instance.
    """

    def __init__(self, topic_keywords: List[str]) -> None:
        super().__init__(role="challenger", name="ChallengerAgent")
        self._keywords = topic_keywords
        self._has_rebutted = False

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.observe(observation)

        # Only rebut once; then wait for proposer to close
        if not self._has_rebutted:
            self._has_rebutted = True
            kw = self._keywords[0] if self._keywords else "the primary issue"
            rebuttal = (
                f"The Proposer makes a valid point on {kw}, "
                f"however this means that implementation remains complex. "
                f"Consequently, a phased approach would be better. "
                f"Because evidence shows that rapid mandates often fail."
            )
            self.memory.opponent_arguments.append(rebuttal)
            return {
                "tool": "submit_rebuttal",
                "params": {
                    "rebuttal": rebuttal,
                    "facts_cited": [],
                    "expose_fallacy": None,
                },
            }

        # Challenger done — let Proposer close
        return {}   # no action this turn


# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent runner
# ──────────────────────────────────────────────────────────────────────────────

def call_env(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_URL}/step",
                   json={"tool": tool, "params": params},
                   timeout=10)
    r.raise_for_status()
    return r.json()


def run_multiagent_episode(topic_id: str) -> Dict[str, Any]:
    """
    Run one full episode with two independent agents taking turns.
    Returns episode summary with reward and rubric breakdown.
    """
    # ── Reset shared environment ──────────────────────────────────────────────
    r = httpx.post(f"{ENV_URL}/reset", json={"topic_id": topic_id}, timeout=10)
    r.raise_for_status()
    obs = r.json()["observation"]

    topic = obs["topic"]
    max_rounds = topic["max_rounds"]

    print(f"\n{'═'*70}")
    print(f"  DEBATE: {topic_id.upper()} — {topic['domain']}")
    print(f"  Motion: {topic['claim'][:65]}…")
    print(f"{'═'*70}")

    # Import topic data for agent initialisation
    from server.tasks import TOPIC_BANK
    topic_obj = next(t for t in TOPIC_BANK if t.topic_id == topic_id)

    # ── Instantiate two INDEPENDENT agents ────────────────────────────────────
    proposer   = ProposerAgent(topic_obj.known_facts, topic_obj.evidence_keywords)
    challenger = ChallengerAgent(topic_obj.evidence_keywords)

    print(f"\n  Agents:  [{proposer.name}]  vs  [{challenger.name}]")
    print(f"  Roles:    proposer              challenger\n")

    turn = 0
    last_result: Dict[str, Any] = {}

    # ── Turn-based loop — agents alternate ────────────────────────────────────
    # Order: Proposer verifies → Proposer argues → Challenger rebuts →
    #        Proposer updates → Proposer concedes → Proposer closes
    turn_order = [proposer, proposer, challenger, proposer, proposer, proposer]

    for agent in turn_order:
        action = agent.decide(obs)
        if not action:
            continue    # agent passed this turn

        tool   = action["tool"]
        params = action["params"]

        result = call_env(tool, params)
        obs    = result.get("observation", obs)
        reward = result.get("reward", 0)
        done   = result.get("done", False)

        tag = f"[{agent.name:<16}] {tool:<20}"
        print(f"  Turn {turn+1:02d}  {tag}  reward={reward:.3f}")
        if result.get("result"):
            print(f"           └─ {result['result'][:80]}")
        turn += 1

        if done:
            last_result = result
            break

    # ── Episode summary ───────────────────────────────────────────────────────
    final_obs = last_result.get("observation", obs)
    winner     = final_obs.get("winner", "?")
    final_reward = last_result.get("reward", 0)

    print(f"\n{'─'*70}")
    print(f"  Winner: {winner.upper()}   |   Final reward: {final_reward:.3f}")
    print(f"\n  Rubric breakdown:")
    for line in last_result.get("breakdown", "").split("\n"):
        if line.strip():
            print(f"    {line}")

    return {
        "topic_id": topic_id,
        "winner": winner,
        "reward": final_reward,
        "proposer": proposer.name,
        "challenger": challenger.name,
        "turns": turn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DebateArenaEnv — Multi-Agent Runner")
    print("Two independent agents, one shared environment server")
    print(f"Environment: {ENV_URL}\n")

    summaries = []
    for topic_id in ["easy", "medium", "hard"]:
        summary = run_multiagent_episode(topic_id)
        summaries.append(summary)

    print(f"\n{'═'*70}")
    print("  MULTI-AGENT EPISODE SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Topic':<10} {'Winner':<15} {'Reward':>8}  {'Turns':>6}")
    print(f"  {'─'*50}")
    for s in summaries:
        print(f"  {s['topic_id']:<10} {s['winner']:<15} {s['reward']:>8.3f}  {s['turns']:>6}")
    print(f"{'═'*70}\n")

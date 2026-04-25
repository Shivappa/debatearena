"""
DebateArenaEnv — Core RL environment.

Implements the OpenEnv gym-style contract:
  env.reset(topic_id)  → observation
  env.step(action)     → observation, reward, done, info
  env.state()          → current state dict
  env.close()          → None

AgentRole  : PROPOSER | CHALLENGER
EpisodePhase: OPEN | DONE
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from server.tasks import REGISTRY, TOPIC_BANK, Topic
from server.rubric import DebateArenaRubric

# ── OpenEnv base shim ─────────────────────────────────────────────────────────
try:
    from openenv import Environment as _OpenEnvBase  # type: ignore
except ImportError:
    class _OpenEnvBase:  # type: ignore
        """Minimal shim when openenv package is not installed."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    PROPOSER   = "proposer"
    CHALLENGER = "challenger"


class EpisodePhase(str, Enum):
    OPEN = "open"
    DONE = "done"


# ──────────────────────────────────────────────────────────────────────────────
# Tool permission sets
# ──────────────────────────────────────────────────────────────────────────────

PROPOSER_TOOLS = {
    "get_topic", "fact_check",
    "make_argument", "update_position",
    "concede_sub_point", "close_debate",
}
CHALLENGER_TOOLS = {
    "get_topic", "fact_check",
    "challenge_argument", "close_debate",
}
# Tools that do NOT advance the round counter
INFO_TOOLS = {"get_topic", "fact_check"}


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentAction:
    role:      AgentRole
    tool:      str
    params:    Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeState:
    topic:            Topic
    phase:            EpisodePhase           = EpisodePhase.OPEN
    round_number:     int                    = 0
    proposer_args:    List[str]              = field(default_factory=list)
    challenger_args:  List[str]              = field(default_factory=list)
    facts_cited:      List[str]              = field(default_factory=list)
    position_updated: bool                   = False
    conceded:         bool                   = False
    winner:           Optional[str]          = None
    final_reward:     float                  = 0.0
    history:          List[AgentAction]      = field(default_factory=list)

    @property
    def debate_closed(self) -> bool:
        return self.phase == EpisodePhase.DONE

    def all_text(self) -> str:
        return " ".join(self.proposer_args + self.challenger_args)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase":            self.phase.value,
            "round_number":     self.round_number,
            "proposer_args":    self.proposer_args,
            "challenger_args":  self.challenger_args,
            "facts_cited":      self.facts_cited,
            "position_updated": self.position_updated,
            "conceded":         self.conceded,
            "winner":           self.winner,
            "final_reward":     self.final_reward,
            "all_text":         self.all_text(),
            "rounds_used":      self.round_number,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class DebateArenaEnv(_OpenEnvBase):
    """
    Multi-agent epistemic debate environment.

    Two agents (Proposer, Challenger) take turns using MCP tools.
    The episode ends when either agent calls end_debate or max_rounds is hit.
    Reward is computed by DebateArenaRubric at episode close.
    """

    def __init__(self) -> None:
        self._rubric  = DebateArenaRubric()
        self._state:  Optional[EpisodeState] = None
        self._topic:  Optional[Topic]        = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, topic_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new episode. Returns initial observation."""
        if topic_id:
            topic = REGISTRY.get_topic(topic_id)
            if topic is None:
                topic = REGISTRY.current_topic()
        else:
            topic = REGISTRY.current_topic()

        self._topic = topic
        self._state = EpisodeState(topic=topic)

        return self._observation("Episode started. Proposer goes first.")

    def step(self, action: AgentAction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one agent action.
        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.debate_closed:
            raise RuntimeError("Episode is already done. Call reset().")

        self._validate_tool(action)
        self._state.history.append(action)

        # Dispatch to internal handler
        result_msg, done = self._dispatch(action)

        # Advance round counter only for non-info tools
        if action.tool not in INFO_TOOLS:
            self._state.round_number += 1

        # Auto-close on max_rounds
        if self._state.round_number >= self._topic.max_rounds and not done:  # type: ignore[union-attr]
            done = True
            result_msg += " (Max rounds reached — debate auto-closed.)"

        reward = 0.01   # interim reward; final computed on close
        if done:
            reward, _ = self._close_episode()

        obs = self._observation(result_msg)
        return obs, reward, done, {"result": result_msg}

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            return {"error": "No active episode. Call reset() first."}
        return {
            "topic":  self._topic.to_dict() if self._topic else {},  # type: ignore[union-attr]
            "state":  self._state.to_dict(),
        }

    def close(self) -> None:
        self._state = None
        self._topic = None

    def render(self) -> str:
        if self._state is None:
            return "No active episode."
        s = self._state
        lines = [
            f"=== DebateArenaEnv ===",
            f"Topic   : {self._topic.claim if self._topic else '—'}",  # type: ignore[union-attr]
            f"Phase   : {s.phase.value}   Round: {s.round_number}",
            f"Proposer args : {len(s.proposer_args)}",
            f"Challenger args: {len(s.challenger_args)}",
            f"Facts cited   : {s.facts_cited}",
            f"Updated position: {s.position_updated}   Conceded: {s.conceded}",
        ]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_tool(self, action: AgentAction) -> None:
        """Raise ValueError if the tool is not permitted for this role."""
        # We allow all tools via HTTP dispatch; role enforcement is advisory here
        pass

    def _dispatch(self, action: AgentAction) -> Tuple[str, bool]:
        """Route action.tool → handler. Returns (message, done)."""
        tool   = action.tool
        params = action.params
        state  = self._state

        if tool == "verify_fact":
            fact_key = params.get("fact_key", "")
            known    = self._topic.known_facts if self._topic else {}  # type: ignore[union-attr]
            if fact_key in known:
                verdict = "✅ TRUE" if known[fact_key] else "❌ FALSE"
                safe    = "Safe to cite." if known[fact_key] else "Do NOT cite — will trigger hallucination penalty!"
                return f"[Fact Check] '{fact_key}' is {verdict}. {safe}", False
            return f"[Fact Check] '{fact_key}' is ❓ UNKNOWN.", False

        elif tool == "submit_argument":
            argument    = params.get("argument", "")
            facts_cited = params.get("facts_cited", [])
            state.proposer_args.append(argument)  # type: ignore[union-attr]
            state.facts_cited.extend(facts_cited)  # type: ignore[union-attr]
            return (
                f"Argument submitted (round {state.round_number + 1}). "  # type: ignore[union-attr]
                "Challenger is preparing a rebuttal. "
                f"Round {state.round_number + 1}/{self._topic.max_rounds}."  # type: ignore[union-attr]
            ), False

        elif tool == "submit_rebuttal":
            rebuttal    = params.get("rebuttal", "")
            facts_cited = params.get("facts_cited", [])
            expose      = params.get("expose_fallacy")
            state.challenger_args.append(rebuttal)  # type: ignore[union-attr]
            state.facts_cited.extend(facts_cited)  # type: ignore[union-attr]
            msg = "Rebuttal submitted. Proposer may now respond or update their position."
            if expose:
                msg += f" Fallacy exposed: {expose}."
            return msg, False

        elif tool == "refine_position":
            refined_claim = params.get("refined_claim", "")
            reason        = params.get("reason", "")
            state.position_updated = True  # type: ignore[union-attr]
            state.proposer_args.append(f"[Refined] {refined_claim} — {reason}")  # type: ignore[union-attr]
            return "Position updated. Belief-updating reward (+0.10) will be applied at close.", False

        elif tool == "concede_point":
            sub_point    = params.get("sub_point", "")
            maintain_main = params.get("maintain_main", "")
            state.conceded = True  # type: ignore[union-attr]
            state.proposer_args.append(f"[Concede] {sub_point}. But: {maintain_main}")  # type: ignore[union-attr]
            return "Concession noted (+0.05 concession credit). Judges reward strategic concessions.", False

        elif tool == "end_debate":
            closing   = params.get("closing_statement", "")
            role      = params.get("role", "proposer")
            state.proposer_args.append(f"[Closing] {closing}")  # type: ignore[union-attr]
            state.phase = EpisodePhase.DONE  # type: ignore[union-attr]
            return f"Debate closed by {role}.", True

        return f"Unknown tool: {tool}", False

    def _close_episode(self) -> Tuple[float, List[Any]]:
        """Score the episode, update curriculum, set winner."""
        state = self._state
        topic = self._topic

        state.phase = EpisodePhase.DONE  # type: ignore[union-attr]

        reward, results = self._rubric.score(
            topic_dict=topic.to_dict(),  # type: ignore[union-attr]
            state_dict=state.to_dict(),  # type: ignore[union-attr]
        )

        state.final_reward = reward  # type: ignore[union-attr]
        state.winner = "proposer" if reward >= 0.5 else "challenger"  # type: ignore[union-attr]

        # Update adaptive curriculum
        REGISTRY.update_curriculum(topic.topic_id, won=(reward >= 0.5))  # type: ignore[union-attr]

        return reward, results

    def _observation(self, message: str) -> Dict[str, Any]:
        state = self._state
        topic = self._topic
        return {
            "message":    message,
            "topic":      topic.to_dict() if topic else {},  # type: ignore[union-attr]
            "round":      state.round_number if state else 0,  # type: ignore[union-attr]
            "done":       state.debate_closed if state else False,  # type: ignore[union-attr]
            "winner":     state.winner if state else None,  # type: ignore[union-attr]
            "reward":     state.final_reward if state else 0.0,  # type: ignore[union-attr]
        }


# Singleton used by the HTTP server
_ENV = DebateArenaEnv()


def get_env() -> DebateArenaEnv:
    return _ENV

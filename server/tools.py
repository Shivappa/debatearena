"""
DebateArenaEnv — MCP tool registry.

Each function here is an MCP tool callable via:
  dispatch(tool_name, params_dict)  → JSON string

Also exported as TOOL_REGISTRY dict for the FastAPI wrapper.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from server.env import get_env

# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

def tool_new_episode(topic_id: Optional[str] = None) -> Dict[str, Any]:
    """Start a new debate episode."""
    obs = get_env().reset(topic_id)
    return {"observation": obs, "result": obs["message"]}


def tool_topic_info() -> Dict[str, Any]:
    """Return the current topic details."""
    s = get_env().state()
    return {"topic": s.get("topic", {})}


def tool_verify_fact(fact_key: str, role: str = "proposer") -> Dict[str, Any]:
    """Check whether a named fact is TRUE or FALSE before citing it."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole(role),
        tool="verify_fact",
        params={"fact_key": fact_key, "role": role},
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "result": info["result"]}


def tool_submit_argument(argument: str, facts_cited: list) -> Dict[str, Any]:
    """Proposer submits a reasoned argument defending the motion."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole.PROPOSER,
        tool="submit_argument",
        params={"argument": argument, "facts_cited": facts_cited},
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "result": info["result"]}


def tool_submit_rebuttal(
    rebuttal: str,
    facts_cited: list,
    expose_fallacy: Optional[str] = None,
) -> Dict[str, Any]:
    """Challenger submits a rebuttal to the proposer's argument."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole.CHALLENGER,
        tool="submit_rebuttal",
        params={"rebuttal": rebuttal, "facts_cited": facts_cited, "expose_fallacy": expose_fallacy},
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "result": info["result"]}


def tool_refine_position(refined_claim: str, reason: str) -> Dict[str, Any]:
    """Proposer updates their position after hearing the challenger (belief-updating bonus)."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole.PROPOSER,
        tool="refine_position",
        params={"refined_claim": refined_claim, "reason": reason},
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "result": info["result"]}


def tool_concede_point(sub_point: str, maintain_main: str) -> Dict[str, Any]:
    """Proposer concedes a minor sub-point while maintaining the core claim (concession bonus)."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole.PROPOSER,
        tool="concede_point",
        params={"sub_point": sub_point, "maintain_main": maintain_main},
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "result": info["result"]}


def tool_end_debate(closing_statement: str, role: str = "proposer") -> Dict[str, Any]:
    """Close the debate with a final statement. Triggers reward computation."""
    from server.env import AgentAction, AgentRole
    env    = get_env()
    action = AgentAction(
        role=AgentRole(role),
        tool="end_debate",
        params={"closing_statement": closing_statement, "role": role},
    )
    obs, reward, done, info = env.step(action)
    # Attach rubric breakdown
    s         = get_env().state()
    state_obj = get_env()._state
    if state_obj is not None:
        from server.rubric import DebateArenaRubric
        topic_dict = get_env()._topic.to_dict() if get_env()._topic else {}
        _, results = DebateArenaRubric().score(topic_dict, state_obj.to_dict())
        breakdown  = DebateArenaRubric.format_breakdown(results)
    else:
        breakdown = ""
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "result":      info["result"],
        "winner":      obs.get("winner", "?"),
        "breakdown":   breakdown,
    }


def tool_get_state() -> Dict[str, Any]:
    """Return the current environment state."""
    return get_env().state()


def tool_curriculum() -> Dict[str, Any]:
    """Return the curriculum win-rate summary."""
    from server.tasks import REGISTRY
    return REGISTRY.summary()


# ──────────────────────────────────────────────────────────────────────────────
# Registry + dispatch
# ──────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "new_episode":      tool_new_episode,
    "topic_info":       tool_topic_info,
    "verify_fact":      tool_verify_fact,
    "submit_argument":  tool_submit_argument,
    "submit_rebuttal":  tool_submit_rebuttal,
    "refine_position":  tool_refine_position,
    "concede_point":    tool_concede_point,
    "end_debate":       tool_end_debate,
    "get_state":        tool_get_state,
    "curriculum":       tool_curriculum,
}


def dispatch(tool_name: str, params: Dict[str, Any]) -> str:
    """Call a tool by name with a params dict. Returns JSON string."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = fn(**params)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# MCP manifest
# ──────────────────────────────────────────────────────────────────────────────

MCP_MANIFEST = {
    "name": "debatearena",
    "version": "1.0.0",
    "tools": [
        {"name": "new_episode",     "description": "Start a new debate episode"},
        {"name": "topic_info",      "description": "Get current topic details"},
        {"name": "verify_fact",     "description": "Check a fact before citing it"},
        {"name": "submit_argument", "description": "Proposer submits argument"},
        {"name": "submit_rebuttal", "description": "Challenger submits rebuttal"},
        {"name": "refine_position", "description": "Proposer updates position (belief-updating bonus)"},
        {"name": "concede_point",   "description": "Proposer concedes sub-point (concession bonus)"},
        {"name": "end_debate",      "description": "Close debate and compute reward"},
        {"name": "get_state",       "description": "Get current environment state"},
        {"name": "curriculum",      "description": "Get curriculum win-rate summary"},
    ],
}

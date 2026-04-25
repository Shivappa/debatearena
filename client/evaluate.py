"""
DebateArenaEnv — Evaluation Suite.

Runs a battery of episodes comparing:
  - BASELINE agent  : hallucinating, no fact-check, uses fallacy phrases → ~0.010
  - OPTIMAL agent   : fact-checked, structured, updates beliefs → ~0.65

Usage
-----
    cd hackathon-finale
    source .venv/bin/activate
    python -m eval.evaluate

Output
------
Prints a reward comparison table and saves reward curves to assets/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from repo root or eval/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.env import AgentAction, AgentRole, DebateArenaEnv  # noqa: E402
from server.tasks import TOPIC_BANK, Topic  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Agent strategies
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_episode(env: DebateArenaEnv, topic: Topic) -> Dict[str, Any]:
    """
    Baseline agent:
      - Skips fact_check
      - Cites a known-False fact (hallucination)
      - Uses a known fallacy phrase in argument text
      - No belief update or concession
    """
    env.reset(topic_id=topic.topic_id)

    false_facts = [k for k, v in topic.known_facts.items() if v is False]
    fallacy_fragment = topic.known_fallacies[0][:30].lower() if topic.known_fallacies else ""

    # Argument with hallucinated fact + fallacy
    arg_action = AgentAction(
        role=AgentRole.PROPOSER,
        tool="make_argument",
        params={
            "argument": (
                f"The motion is clearly correct. "
                f"Everyone knows this. {fallacy_fragment}. "
                f"The data is obvious so no regulation is needed."
            ),
            "facts_cited": false_facts[:1],   # intentionally cite a FALSE fact
        },
    )
    env.step(arg_action)

    # Weak challenger with no facts
    reb_action = AgentAction(
        role=AgentRole.CHALLENGER,
        tool="challenge_argument",
        params={"rebuttal": "I disagree entirely but have no evidence.", "facts_cited": []},
    )
    env.step(reb_action)

    # Close
    close_action = AgentAction(
        role=AgentRole.PROPOSER,
        tool="close_debate",
        params={"closing_statement": "My position stands."},
    )
    _, reward, _, info = env.step(close_action)

    return {
        "agent": "baseline",
        "topic": topic.topic_id,
        "reward": reward,
        "breakdown": info.get("breakdown", ""),
    }


def run_optimal_episode(env: DebateArenaEnv, topic: Topic) -> Dict[str, Any]:
    """
    Optimal agent:
      - fact_checks before citing
      - Uses evidence keywords from the topic
      - Structural connectives (because / therefore / studies show)
      - Updates position after challenger rebuttal
      - Concedes a sub-point gracefully
      - Closes with coherent summary
    """
    env.reset(topic_id=topic.topic_id)

    true_facts = [k for k, v in topic.known_facts.items() if v is True]
    kw = topic.evidence_keywords

    # 1. Verify a fact before citing
    env.step(AgentAction(
        role=AgentRole.PROPOSER, tool="fact_check",
        params={"fact_key": true_facts[0]},
    ))

    # 2. Strong argument with verified facts + evidence keywords + connectives
    argument = (
        f"The motion is correct because evidence shows {kw[0] if kw else 'the data'}. "
        f"Studies show this is supported by {true_facts[0]}. "
        f"Therefore the conclusion follows logically. "
        f"As a result, this policy is necessary and proportionate."
    )
    env.step(AgentAction(
        role=AgentRole.PROPOSER, tool="make_argument",
        params={"argument": argument, "facts_cited": true_facts[:2]},
    ))

    # 3. Structured challenger rebuttal with connectives
    rebuttal = (
        f"The Proposer makes a valid point on {kw[0] if kw else 'the primary issue'}, "
        f"however this means that implementation remains complex. "
        f"Consequently, a phased approach would be better. "
        f"Because evidence shows that rapid mandates often fail."
    )
    env.step(AgentAction(
        role=AgentRole.CHALLENGER, tool="challenge_argument",
        params={"rebuttal": rebuttal, "facts_cited": []},
    ))

    # 4. Update position (belief-updating bonus)
    env.step(AgentAction(
        role=AgentRole.PROPOSER, tool="update_position",
        params={
            "refined_claim": "A phased, audited rollout is more defensible.",
            "reason": "The Challenger raised valid implementation concerns.",
        },
    ))

    # 5. Concede a sub-point (concession credit)
    env.step(AgentAction(
        role=AgentRole.PROPOSER, tool="concede_sub_point",
        params={
            "sub_point": "Enforcement mechanisms are genuinely complex.",
            "maintain_main": "The core obligation to label remains justified.",
        },
    ))

    # 6. Close with structured summary
    closing = (
        f"In conclusion, because the evidence is clear and studies show the need, "
        f"therefore this motion stands. As a result, we urge adoption."
    )
    _, reward, _, info = env.step(AgentAction(
        role=AgentRole.PROPOSER, tool="close_debate",
        params={"closing_statement": closing},
    ))

    return {
        "agent": "optimal",
        "topic": topic.topic_id,
        "reward": reward,
        "breakdown": info.get("breakdown", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(n_runs: int = 1) -> List[Dict[str, Any]]:
    env = DebateArenaEnv(adaptive=False)
    results: List[Dict[str, Any]] = []

    for topic in TOPIC_BANK:
        for _ in range(n_runs):
            results.append(run_baseline_episode(env, topic))
            results.append(run_optimal_episode(env, topic))

    return results


def print_report(results: List[Dict[str, Any]]) -> None:
    from collections import defaultdict
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        scores[r["topic"]][r["agent"]].append(r["reward"])

    topics = [t.topic_id for t in TOPIC_BANK]
    divider = "─" * 65

    print(f"\n{divider}")
    print(f"  {'Topic':<10} {'Baseline':>12}  {'Optimal':>10}  {'Lift':>10}")
    print(divider)
    for topic_id in topics:
        b = scores[topic_id].get("baseline", [0])
        o = scores[topic_id].get("optimal", [0])
        avg_b = sum(b) / len(b)
        avg_o = sum(o) / len(o)
        lift = avg_o - avg_b
        print(f"  {topic_id:<10} {avg_b:>12.3f}  {avg_o:>10.3f}  {lift:>+10.3f}")
    print(divider)
    print("\nSelf-improvement loop:")
    for r in results:
        if r["agent"] == "optimal" and "🔼" in r.get("breakdown", ""):
            print(f"  ✅ [{r['topic']}] Self-improvement triggered")
    print()


def save_results(results: List[Dict[str, Any]]) -> None:
    out_dir = Path(__file__).parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved → {out_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        topics = [t.topic_id for t in TOPIC_BANK]
        baseline_rewards = [
            next(r["reward"] for r in results if r["topic"] == t and r["agent"] == "baseline")
            for t in topics
        ]
        optimal_rewards = [
            next(r["reward"] for r in results if r["topic"] == t and r["agent"] == "optimal")
            for t in topics
        ]

        x = range(len(topics))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([i - 0.2 for i in x], baseline_rewards, 0.4, label="Baseline", color="#EF5350")
        ax.bar([i + 0.2 for i in x], optimal_rewards, 0.4, label="Optimal", color="#42A5F5")
        ax.set_xticks(list(x))
        ax.set_xticklabels(topics)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Episode Reward")
        ax.set_title("DebateArenaEnv — Baseline vs Optimal Agent Reward")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig_path = out_dir / "reward_comparison.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Chart saved  → {fig_path}")
    except ImportError:
        print("(matplotlib not installed — skipping chart)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running DebateArenaEnv evaluation…")
    t0 = time.time()
    results = run_evaluation(n_runs=1)
    print_report(results)
    save_results(results)
    print(f"Completed in {time.time() - t0:.1f}s")

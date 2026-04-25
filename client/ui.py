"""
DebateArenaEnv — Gradio UI for HuggingFace Spaces (port 7860).

Calls the env directly (same process) — no HTTP needed for the UI.
The FastAPI server (port 8000) runs alongside for the judge/eval harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr  # type: ignore

from server.env import AgentAction, AgentRole, DebateArenaEnv
from server.tasks import TOPIC_BANK

# ── Singleton env (shared with the Gradio session) ────────────────────────────
_env = DebateArenaEnv()
TOPIC_CHOICES = [t.topic_id for t in TOPIC_BANK]


def _fmt_score(score: float) -> str:
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    return f"{bar}  {score:.3f}"


# ── Handlers ──────────────────────────────────────────────────────────────────

def start_episode(topic_id: str) -> tuple:
    obs = _env.reset(topic_id=topic_id)
    topic = obs.get("topic", {})
    status = (
        f"🟢 **Episode started** | "
        f"Motion: **{topic.get('claim','')[:70]}…** | "
        f"Difficulty: `{topic.get('difficulty','?')}` | "
        f"Max rounds: `{topic.get('max_rounds','?')}`"
    )
    return status, "", "0.010"


def verify_fact(fact_key: str) -> str:
    if not fact_key.strip():
        return "⚠️  Enter a fact key."
    action = AgentAction(role=AgentRole.PROPOSER, tool="verify_fact",
                         params={"fact_key": fact_key.strip(), "role": "proposer"})
    try:
        _, _, _, info = _env.step(action)
        return info.get("result", "")
    except Exception as exc:
        return f"❌ {exc}"


def submit_argument(argument: str, facts_cited_str: str) -> tuple:
    if not argument.strip():
        return "⚠️  Write an argument first.", "", ""
    facts = [f.strip() for f in facts_cited_str.split(",") if f.strip()]
    action = AgentAction(role=AgentRole.PROPOSER, tool="submit_argument",
                         params={"argument": argument, "facts_cited": facts})
    try:
        obs, reward, done, info = _env.step(action)
        return (
            f"✅ Argument submitted | reward: **{_fmt_score(reward)}**\n\n→ {info.get('result','')}",
            "",
            f"{reward:.3f}",
        )
    except Exception as exc:
        return f"❌ {exc}", "", ""


def submit_rebuttal(rebuttal: str, expose_fallacy: str) -> tuple:
    if not rebuttal.strip():
        return "⚠️  Write a rebuttal first.", "", ""
    action = AgentAction(role=AgentRole.CHALLENGER, tool="submit_rebuttal",
                         params={"rebuttal": rebuttal, "facts_cited": [],
                                 "expose_fallacy": expose_fallacy or None})
    try:
        obs, reward, done, info = _env.step(action)
        return (
            f"✅ Rebuttal submitted | reward: **{_fmt_score(reward)}**\n\n→ {info.get('result','')}",
            "",
            f"{reward:.3f}",
        )
    except Exception as exc:
        return f"❌ {exc}", "", ""


def close_debate(closing_statement: str) -> tuple:
    if not closing_statement.strip():
        return "⚠️  Write a closing statement.", ""
    action = AgentAction(role=AgentRole.PROPOSER, tool="end_debate",
                         params={"closing_statement": closing_statement, "role": "proposer"})
    try:
        obs, reward, done, info = _env.step(action)
        winner = obs.get("winner", "?")
        from server.rubric import DebateArenaRubric
        state_obj = _env._state
        topic_dict = _env._topic.to_dict() if _env._topic else {}
        breakdown = ""
        if state_obj:
            _, results = DebateArenaRubric().score(topic_dict, state_obj.to_dict())
            breakdown = DebateArenaRubric.format_breakdown(results)
        status = (
            f"🏁 **Debate Closed!**\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| 🏆 Winner | **{winner.upper()}** |\n"
            f"| 🎯 Final Reward | **{_fmt_score(reward)}** |"
        )
        return status, breakdown
    except Exception as exc:
        return f"❌ {exc}", ""


# ── Layout ────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# ⚖️ DebateArenaEnv — Multi-Agent Epistemic Reasoning

Train LLM agents to reason about truth, update beliefs under counter-evidence,
and avoid hallucination — through adversarial structured debate.

| Role | Goal |
|------|------|
| 👤 **Proposer** | Defend the motion with verified facts + logical connectives |
| 🗣 **Challenger** | Rebut with counter-evidence, expose fallacies |
| ⚖️ **Judge** | Automated rubric: factual accuracy · coherence · belief-updating |

**Tip:** Always call *Verify Fact* before citing — FALSE facts cost −0.30 each!
"""

with gr.Blocks(title="DebateArenaEnv") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        topic_dd  = gr.Dropdown(choices=TOPIC_CHOICES, value="easy", label="Topic")
        start_btn = gr.Button("🚀 Start New Debate", variant="primary")

    status_box = gr.Markdown(label="Status")
    reward_box = gr.Textbox(label="Current Reward", interactive=False)

    gr.Markdown("---")

    with gr.Tab("👤 Proposer"):
        with gr.Row():
            fact_key_in = gr.Textbox(label="Fact key to verify")
            verify_btn  = gr.Button("🔍 Verify Fact")
        fact_result    = gr.Textbox(label="Fact Check Result", interactive=False)
        argument_in    = gr.Textbox(label="Argument", lines=4)
        facts_in       = gr.Textbox(label="Facts cited (comma-separated keys)",
                                     placeholder="EU_AI_Act_mandates_labelling")
        submit_arg_btn = gr.Button("📝 Submit Argument", variant="primary")
        arg_status     = gr.Markdown()

    with gr.Tab("🗣 Challenger"):
        rebuttal_in    = gr.Textbox(label="Rebuttal", lines=4)
        fallacy_in     = gr.Textbox(label="Fallacy to expose (optional)")
        submit_reb_btn = gr.Button("⚔️  Submit Rebuttal", variant="secondary")
        reb_status     = gr.Markdown()

    with gr.Tab("🏁 Close Debate"):
        closing_in      = gr.Textbox(label="Closing statement", lines=3)
        close_btn       = gr.Button("⚖️  Close & Score", variant="stop")
        close_status    = gr.Markdown()
        close_breakdown = gr.Textbox(label="Rubric Breakdown", interactive=False, lines=14)

    # Wire events
    start_btn.click(start_episode, inputs=[topic_dd],
                    outputs=[status_box, arg_status, reward_box])
    verify_btn.click(verify_fact, inputs=[fact_key_in], outputs=[fact_result])
    submit_arg_btn.click(submit_argument, inputs=[argument_in, facts_in],
                         outputs=[status_box, arg_status, reward_box])
    submit_reb_btn.click(submit_rebuttal, inputs=[rebuttal_in, fallacy_in],
                         outputs=[status_box, reb_status, reward_box])
    close_btn.click(close_debate, inputs=[closing_in],
                    outputs=[close_status, close_breakdown])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

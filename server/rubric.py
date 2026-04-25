"""
DebateArenaEnv — Composable reward rubrics.

Each rubric scores one aspect of the debate and returns a RubricResult.
DebateArenaRubric orchestrates all rubrics and returns a final clamped reward.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RubricResult:
    name:      str
    score:     float
    weight:    float
    rationale: str
    details:   str = ""

    def weighted(self) -> float:
        return self.score * self.weight

    def __str__(self) -> str:
        sign = "+" if self.weighted() >= 0 else "-"
        return (
            f"[{self.name:<28}] {sign}+{abs(self.weighted()):.3f}"
            f"  — {self.rationale}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Base rubric
# ──────────────────────────────────────────────────────────────────────────────

class BaseRubric(ABC):
    name:   str = "base"
    weight: float = 0.0

    @abstractmethod
    def score(self, **kwargs: Any) -> RubricResult:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Individual rubrics
# ──────────────────────────────────────────────────────────────────────────────

class FactualAccuracyRubric(BaseRubric):
    """+0.35 per TRUE fact cited; triggered by facts_cited list."""
    name   = "factual_accuracy"
    weight = 0.35

    def score(self, facts_cited: List[str], known_facts: Dict[str, bool], **_: Any) -> RubricResult:
        if not facts_cited:
            return RubricResult(self.name, 0.0, self.weight, "No facts cited.", "")
        correct = [f for f in facts_cited if known_facts.get(f) is True]
        ratio   = len(correct) / len(facts_cited)
        return RubricResult(
            self.name, ratio, self.weight,
            f"{len(correct)}/{len(facts_cited)} cited facts are correct.",
            str(correct),
        )


class EvidenceQualityRubric(BaseRubric):
    """+0.20 for using domain evidence keywords in the argument text."""
    name   = "evidence_quality"
    weight = 0.20

    def score(self, text: str, evidence_keywords: List[str], **_: Any) -> RubricResult:
        if not evidence_keywords:
            return RubricResult(self.name, 0.0, self.weight, "No keywords defined.", "")
        hits  = [kw for kw in evidence_keywords if kw.lower() in text.lower()]
        ratio = len(hits) / len(evidence_keywords)
        return RubricResult(
            self.name, ratio, self.weight,
            f"{len(hits)}/{len(evidence_keywords)} evidence keywords present.",
            str(hits),
        )


class LogicalCoherenceRubric(BaseRubric):
    """+0.15 for using logical connectives in the argument."""
    name   = "logical_coherence"
    weight = 0.15

    MARKERS = [
        "because", "therefore", "studies show", "as a result",
        "consequently", "however", "furthermore", "evidence suggests",
        "this demonstrates", "thus",
    ]

    def score(self, text: str, **_: Any) -> RubricResult:
        hits = [m for m in self.MARKERS if m.lower() in text.lower()]
        ratio = min(len(hits) / 3, 1.0)   # saturates at 3 connectives
        return RubricResult(
            self.name, ratio, self.weight,
            f"{len(hits)} logical connectives detected.",
            str(hits),
        )


class BeliefUpdatingRubric(BaseRubric):
    """+0.10 if the proposer updated their position after counter-evidence."""
    name   = "belief_updating"
    weight = 0.10

    def score(self, position_updated: bool, **_: Any) -> RubricResult:
        s = 1.0 if position_updated else 0.0
        msg = "Position was updated under counter-evidence." if position_updated else "Position not updated."
        return RubricResult(self.name, s, self.weight, msg)


class ConcessionCreditRubric(BaseRubric):
    """+0.05 if the proposer gracefully conceded a sub-point."""
    name   = "concession_credit"
    weight = 0.05

    def score(self, conceded: bool, **_: Any) -> RubricResult:
        s = 1.0 if conceded else 0.0
        msg = "Sub-point conceded gracefully." if conceded else "No concession made."
        return RubricResult(self.name, s, self.weight, msg)


class HallucinationPenaltyRubric(BaseRubric):
    """-0.30 per FALSE fact cited."""
    name   = "hallucination_penalty"
    weight = -0.30

    def score(self, facts_cited: List[str], known_facts: Dict[str, bool], **_: Any) -> RubricResult:
        false_facts = [f for f in facts_cited if known_facts.get(f) is False]
        penalty     = len(false_facts)   # weight is negative; multiplied below
        msg = f"{len(false_facts)} hallucinated fact(s) cited." if false_facts else "No hallucinations detected."
        return RubricResult(self.name, penalty, self.weight, msg, str(false_facts))


class FallacyPenaltyRubric(BaseRubric):
    """-0.20 per known logical fallacy used."""
    name   = "fallacy_penalty"
    weight = -0.20

    def score(self, text: str, known_fallacies: List[str], **_: Any) -> RubricResult:
        hits = [f for f in known_fallacies if f.replace("_", " ").lower() in text.lower()]
        msg  = f"{len(hits)} fallacy/fallacies used." if hits else "No fallacies detected."
        return RubricResult(self.name, len(hits), self.weight, msg, str(hits))


class StepPenaltyRubric(BaseRubric):
    """-0.01 per round used (encourages efficiency)."""
    name   = "step_penalty"
    weight = -0.01

    def score(self, rounds_used: int, **_: Any) -> RubricResult:
        return RubricResult(
            self.name, rounds_used, self.weight,
            f"{rounds_used} rounds used (−{rounds_used * 0.01:.2f} step penalty).",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class DebateArenaRubric:
    """
    Computes the final episode reward by running all rubrics and summing
    weighted scores. Reward is clamped to [0.01, 0.99].
    """

    def __init__(self) -> None:
        self._rubrics: List[BaseRubric] = [
            FactualAccuracyRubric(),
            EvidenceQualityRubric(),
            LogicalCoherenceRubric(),
            BeliefUpdatingRubric(),
            ConcessionCreditRubric(),
            HallucinationPenaltyRubric(),
            FallacyPenaltyRubric(),
            StepPenaltyRubric(),
        ]

    def score(
        self,
        topic_dict:  Dict[str, Any],
        state_dict:  Dict[str, Any],
    ) -> Tuple[float, List[RubricResult]]:
        """
        Returns (reward, results_list).

        Expected keys in state_dict:
          all_text        : str  — concatenated proposer arguments
          facts_cited     : list — fact keys cited across the episode
          position_updated: bool
          conceded        : bool
          rounds_used     : int
        """
        known_facts     = topic_dict.get("known_facts", {})
        known_fallacies = topic_dict.get("known_fallacies", [])
        evidence_kws    = topic_dict.get("evidence_keywords", [])

        all_text         = state_dict.get("all_text", "")
        facts_cited      = state_dict.get("facts_cited", [])
        position_updated = state_dict.get("position_updated", False)
        conceded         = state_dict.get("conceded", False)
        rounds_used      = state_dict.get("rounds_used", 0)

        kwargs = dict(
            text             = all_text,
            facts_cited      = facts_cited,
            known_facts      = known_facts,
            known_fallacies  = known_fallacies,
            evidence_keywords= evidence_kws,
            position_updated = position_updated,
            conceded         = conceded,
            rounds_used      = rounds_used,
        )

        results = [r.score(**kwargs) for r in self._rubrics]
        total   = sum(r.weighted() for r in results)
        reward  = max(0.01, min(0.99, total))
        return reward, results

    @staticmethod
    def format_breakdown(results: List[RubricResult]) -> str:
        lines = ["Rubric Breakdown:"]
        for r in results:
            lines.append(f"  {r}")
        return "\n".join(lines)

"""
DebateArenaEnv — Topic bank and adaptive curriculum.

TopicRegistry tracks per-topic win-rates and escalates difficulty
when the agent wins >= 70% of episodes over the last >= 3 attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Topic dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Topic:
    topic_id: str
    difficulty: str                      # "easy" | "medium" | "hard"
    domain: str
    claim: str
    narrative: str
    known_facts: Dict[str, bool]         # fact_key → True/False
    known_fallacies: List[str]
    evidence_keywords: List[str]
    max_rounds: int = 5
    self_play_seed: int = 42

    def to_prompt(self) -> str:
        return (
            f"Motion: {self.claim}\n"
            f"Domain: {self.domain}\n"
            f"Background: {self.narrative}\n"
            f"Max rounds: {self.max_rounds}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id":         self.topic_id,
            "difficulty":       self.difficulty,
            "domain":           self.domain,
            "claim":            self.claim,
            "narrative":        self.narrative,
            "known_facts":      self.known_facts,
            "known_fallacies":  self.known_fallacies,
            "evidence_keywords": self.evidence_keywords,
            "max_rounds":       self.max_rounds,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Topic bank  (3 topics: easy / medium / hard)
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_BANK: List[Topic] = [
    Topic(
        topic_id="easy",
        difficulty="easy",
        domain="AI Policy & Media",
        claim="Social media platforms should be legally required to label AI-generated content.",
        narrative=(
            "Deepfakes and synthetic media are increasingly used in disinformation campaigns. "
            "The EU AI Act mandates labelling; the US DEFIANCE Act is under debate. "
            "Proponents argue transparency protects democratic discourse."
        ),
        known_facts={
            "EU_AI_Act_mandates_labelling":       True,
            "deepfakes_used_in_elections_2024":   True,
            "labelling_eliminates_disinformation": False,
            "US_DEFIANCE_Act_passed":              False,
        },
        known_fallacies=["slippery_slope", "appeal_to_fear"],
        evidence_keywords=[
            "EU AI Act", "DEFIANCE Act", "synthetic media",
            "disinformation", "transparency", "deepfake",
        ],
        max_rounds=5,
    ),
    Topic(
        topic_id="medium",
        difficulty="medium",
        domain="Climate & Energy Policy",
        claim="Governments should mandate a complete phase-out of fossil fuels by 2035.",
        narrative=(
            "IPCC reports warn that net-zero by 2050 requires rapid decarbonisation. "
            "Several EU nations have set 2035 targets; critics cite energy security risks. "
            "Renewables now account for 30% of global electricity."
        ),
        known_facts={
            "IPCC_net_zero_2050_target":           True,
            "renewables_30pct_global_electricity": True,
            "fossil_fuel_phaseout_causes_blackouts": False,
            "EU_2035_fossil_fuel_ban_enacted":      False,
        },
        known_fallacies=["false_dilemma", "appeal_to_nature"],
        evidence_keywords=[
            "IPCC", "net-zero", "decarbonisation", "renewables",
            "energy security", "carbon emissions",
        ],
        max_rounds=6,
    ),
    Topic(
        topic_id="hard",
        difficulty="hard",
        domain="Bioethics & Genomics",
        claim="Human germline gene editing for disease prevention should be legally permitted.",
        narrative=(
            "CRISPR-Cas9 enables heritable edits. The He Jiankui case triggered global bans. "
            "Proponents cite eradication of hereditary diseases; opponents warn of eugenics slippery slopes. "
            "WHO guidelines call for a moratorium pending governance frameworks."
        ),
        known_facts={
            "CRISPR_enables_heritable_edits":         True,
            "WHO_moratorium_on_germline_editing":      True,
            "germline_editing_eradicates_all_disease": False,
            "He_Jiankui_case_led_to_global_ban":       False,  # led to bans in many, not all countries
        },
        known_fallacies=["slippery_slope", "appeal_to_nature", "false_dilemma"],
        evidence_keywords=[
            "CRISPR", "germline", "WHO", "hereditary disease",
            "bioethics", "eugenics", "governance",
        ],
        max_rounds=7,
    ),
]

# Lookup by topic_id
_TOPIC_MAP: Dict[str, Topic] = {t.topic_id: t for t in TOPIC_BANK}


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive curriculum
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicRegistry:
    """
    Tracks per-topic win-rates and escalates difficulty.

    Escalation rule:
      win_rate >= 0.70  over the last >= 3 attempts → bump to next difficulty
    """
    _history: Dict[str, List[bool]] = field(default_factory=dict)

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        return _TOPIC_MAP.get(topic_id)

    def current_topic(self) -> Topic:
        """Return the easiest topic that hasn't been mastered yet."""
        order = ["easy", "medium", "hard"]
        for tid in order:
            if not self._is_mastered(tid):
                return _TOPIC_MAP[tid]
        return _TOPIC_MAP["hard"]

    def update_curriculum(self, topic_id: str, won: bool) -> None:
        self._history.setdefault(topic_id, []).append(won)

    def win_rate(self, topic_id: str) -> float:
        hist = self._history.get(topic_id, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def attempts(self, topic_id: str) -> int:
        return len(self._history.get(topic_id, []))

    def _is_mastered(self, topic_id: str) -> bool:
        return self.attempts(topic_id) >= 3 and self.win_rate(topic_id) >= 0.70

    def summary(self) -> Dict[str, Any]:
        return {
            tid: {
                "attempts":  self.attempts(tid),
                "win_rate":  round(self.win_rate(tid), 3),
                "mastered":  self._is_mastered(tid),
            }
            for tid in _TOPIC_MAP
        }


# Singleton used by the environment
REGISTRY = TopicRegistry()

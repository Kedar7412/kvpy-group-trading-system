"""Abstention / selective prediction: turn a score + confidence + crash risk
into an honest call -- including the option to say nothing.

The whole point: most of the time the right answer is "no edge, don't trade."
Emitting a confident-looking number on noise is exactly the dishonesty this
product exists to avoid.

Calls
-----
NO-EDGE       confidence below the floor -> we abstain
AVOID-BUBBLE  high crash probability -> stay away regardless of score
FAVORED       high score with real conviction
AVOID         low score with real conviction
NEUTRAL       in-between
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AbstentionConfig


@dataclass(frozen=True)
class Recommendation:
    call: str        # NO-EDGE | AVOID-BUBBLE | FAVORED | AVOID | NEUTRAL
    rationale: str

    @property
    def actionable(self) -> bool:
        return self.call in ("FAVORED", "AVOID")


def recommend(
    score: float,
    confidence: float,
    bubble_probability: float,
    cfg: AbstentionConfig,
    longs_ok: bool = True,
) -> Recommendation:
    if bubble_probability is not None and bubble_probability >= cfg.bubble_avoid_prob:
        return Recommendation(
            "AVOID-BUBBLE",
            f"crash probability {bubble_probability:.0%} -- looks like hype, not value",
        )
    if confidence != confidence or confidence < cfg.min_confidence:
        return Recommendation(
            "NO-EDGE",
            f"confidence {0.0 if confidence != confidence else confidence:.0%}"
            f" below {cfg.min_confidence:.0%} -- no reliable edge today",
        )
    if score >= cfg.favored_score:
        if not longs_ok:
            return Recommendation(
                "NEUTRAL",
                f"score {score:.0f} but risk-off market -- longs suppressed",
            )
        return Recommendation("FAVORED", f"score {score:.0f} with sufficient conviction")
    if score <= cfg.avoid_score:
        return Recommendation("AVOID", f"score {score:.0f} with sufficient conviction")
    return Recommendation("NEUTRAL", f"score {score:.0f} -- nothing decisive")

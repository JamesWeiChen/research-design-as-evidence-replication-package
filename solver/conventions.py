"""Shared equilibrium conventions for the NHB solver.

The model requires two rules beyond its primitive parameters:

* ``completion`` determines the reviewer's belief when the relevant type
  window has zero prior mass. The public solver uses the face-value rule,
  which assigns zero conjectured adjustment and evaluates the manuscript at
  its observed value.
* ``tie_rule`` determines the reviewer's action when expected payoff is
  numerically zero. The default accepts an indifferent manuscript at or above
  the equilibrium cutoff and rejects one strictly below it.

Non-adjusting scholars submit their natural manuscript. This convention is
fixed by the model and is checked explicitly by the mixed-TR diagnostics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "Completion",
    "Conventions",
    "TieRule",
    "face_value_completion",
]


@dataclass(eq=False)
class Completion:
    """Belief used when a payoff-relevant type window has zero mass.

    ``mean_adjustment(Y)`` returns the conjectured mean adjustment in a
    window of width ``Y``. The value must lie in ``[0, Y]``. Completions
    compare by name so independently constructed instances of the same rule
    have stable equality semantics.
    """

    name: str = "face_value"
    mean_adjustment: Callable[[float], float] = field(
        default=lambda window_width: 0.0
    )
    face_value: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Completion):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def cond_mean(self, manuscript_state: float, window_width: float) -> float:
        """Return the conjectured conditional mean of the scholar type."""

        adjustment = float(self.mean_adjustment(window_width))
        if not 0.0 <= adjustment <= window_width:
            raise ValueError(
                "completion mean adjustment must lie in [0, Y]; got "
                f"{adjustment!r} for Y={window_width!r}"
            )
        return manuscript_state - adjustment

    @property
    def is_face_value(self) -> bool:
        """Return whether the rule is the declared face-value completion."""

        return self.face_value


def face_value_completion() -> Completion:
    """Return the selected null-window rule: read at face value."""

    return Completion(
        name="face_value",
        mean_adjustment=lambda window_width: 0.0,
        face_value=True,
    )


class TieRule(str, enum.Enum):
    """Reviewer action at numerical indifference, ``G = 0``."""

    ACCEPT_ABOVE_CUTOFF = "accept_above_cutoff"
    ALL_ACCEPT = "all_accept"


@dataclass(frozen=True)
class Conventions:
    """Immutable rules shared by fixed-design and mixed-design solvers.

    ``mass_tol`` classifies a prior window as charged or null. ``tie_tol``
    detects numerical indifference when ``TieRule.ALL_ACCEPT`` is selected.
    """

    completion: Completion = field(default_factory=face_value_completion)
    tie_rule: TieRule = TieRule.ACCEPT_ABOVE_CUTOFF
    mass_tol: float = 1e-12
    tie_tol: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.completion, Completion):
            raise TypeError("completion must be a Completion instance")
        if not isinstance(self.tie_rule, TieRule):
            raise TypeError("tie_rule must be a TieRule member")
        if not 0.0 <= float(self.mass_tol) < 1e-6:
            raise ValueError("mass_tol must lie in [0, 1e-6)")
        if not 0.0 < float(self.tie_tol) < 1e-3:
            raise ValueError("tie_tol must lie in (0, 1e-3)")

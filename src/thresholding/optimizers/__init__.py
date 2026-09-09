"""DE variant registry (Assignment 1.3).

Five variants target the mark rubric. NOTE: the assignment PDF is internally
inconsistent -- section 1.3 lists 5 (DE, JADE, SHADE, L-SHADE, LADE), Table 1
shows 6 (with CoDE + SaDE instead of LADE), and the rubric says 5. We build the
rubric's 5; confirm with the lecturer whether CoDE/SaDE are also required.

Bodies are stubs for now; each raises NotImplementedError until its owner fills
it in. See README "DE variant split".
"""

from __future__ import annotations

from .base import BaseOptimizer, OptResult
from .de import StandardDE
from .jade import JADE
from .lade import LateAcceptanceDE
from .lshade import LSHADE
from .shade import SHADE

OPTIMIZERS: dict[str, type[BaseOptimizer]] = {
    StandardDE.name: StandardDE,
    JADE.name: JADE,
    SHADE.name: SHADE,
    LSHADE.name: LSHADE,
    LateAcceptanceDE.name: LateAcceptanceDE,
}

__all__ = [
    "OPTIMIZERS",
    "BaseOptimizer",
    "OptResult",
    "StandardDE",
    "JADE",
    "SHADE",
    "LSHADE",
    "LateAcceptanceDE",
]

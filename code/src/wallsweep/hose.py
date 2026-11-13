"""Hose model: the drone is fed from a pump on the ground, so the load grows with height.

The hose runs from the pump, along the ground, then hangs up to the drone. The part hanging
below the drone is about as long as the drone is high, so:

    lifted hose mass = hose weight per metre x drone height

The simulator adds this to the drone's mass each control step. The hanging hose also pulls the
drone slightly towards the pump; we model that as a small fraction of the hose weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wallsweep.world import Facade

G = 9.8


@dataclass
class Hose:
    grams_per_m: float = 3.0
    pump_local: tuple[float, float, float] = (-0.6, 0.0, 0.0)  # on the ground, 60 cm out, centred
    pull_fraction: float = 0.1  # sideways pull as a fraction of the lifted hose weight

    def lifted_mass(self, height: float) -> float:
        """Hose mass (kg) hanging below the drone at this height."""
        return self.grams_per_m / 1000.0 * max(float(height), 0.0)

    def pull(self, pos_world: np.ndarray, facade: Facade) -> np.ndarray:
        """Horizontal force (N, world frame) towards the pump."""
        to_pump = facade.to_world(self.pump_local) - pos_world
        to_pump[2] = 0.0
        d = np.linalg.norm(to_pump)
        if d < 1e-6:
            return np.zeros(3)
        return self.pull_fraction * self.lifted_mass(pos_world[2]) * G * to_pump / d

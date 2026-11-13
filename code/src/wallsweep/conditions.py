"""Fixed wall sets: everything that defines one flight's conditions, from a set name and an index.

Each set uses its own seed range, so no wall is ever in two sets:

    train: unlimited walls, seeds from 0. Used by NSGA-II and random search.
    val: 50 walls, seeds from 1,000,000. Used to pick controllers from the Pareto front.
    test: 100 walls, seeds from 2,000,000. Each wall is flown at all 3 gust levels (300 flights).
    harder: 50 walls, seeds from 3,000,000. Deeper features, strong gusts, more glass dropouts
        and a heavier hose.

Per flight we randomize the facade, gust level, glass dropout rate, hose weight and pump
position. The same (set, index, gust) always gives the same flight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wallsweep.hose import Hose
from wallsweep.world import Facade, FacadeAviary, Gusts, sample_facade

GUST_LEVELS = ("none", "moderate", "strong")

SETS = {  # name: (first seed, number of walls or None for unlimited)
    "train": (0, None),
    "val": (1_000_000, 50),
    "test": (2_000_000, 100),
    "harder": (3_000_000, 50),
}
SET_CODE = {name: i for i, name in enumerate(SETS)}


@dataclass(frozen=True)
class Conditions:
    set: str
    index: int
    seed: int  # facade seed, unique across all sets
    facade: Facade
    gust_level: str
    glass_dropout: float
    hose: Hose

    def make_env(self, gui: bool = False) -> FacadeAviary:
        """A fresh simulator for this flight."""
        return FacadeAviary(
            self.facade,
            gusts=Gusts(self.gust_level, seed=self.seed),
            hose=self.hose,
            gui=gui,
            noise_seed=self.seed,
            glass_dropout=self.glass_dropout,
        )

    def __str__(self) -> str:
        return (
            f"{self.set}[{self.index}] seed {self.seed}: gusts {self.gust_level}, "
            f"glass dropout {self.glass_dropout:.0%}, hose {self.hose.grams_per_m:.1f} g/m"
        )


def set_size(name: str) -> int | None:
    return SETS[name][1]


def conditions(set_name: str, index: int, gust: str | None = None) -> Conditions:
    """Conditions for wall `index` of a set. `gust` overrides the set's gust level (the test set
    is meant to be flown at all three)."""
    first, size = SETS[set_name]
    if index < 0 or (size is not None and index >= size):
        raise IndexError(f"{set_name} has {size} walls; index {index} is out of range")
    seed = first + index
    harder = set_name == "harder"
    rng = np.random.default_rng([SET_CODE[set_name], index, 642])  # separate from the facade's

    if gust is None:
        gust = "strong" if harder else str(rng.choice(GUST_LEVELS))
    elif gust not in GUST_LEVELS:
        raise ValueError(f"gust must be one of {GUST_LEVELS}")

    dropout = rng.uniform(0.3, 0.5) if harder else rng.uniform(0.0, 0.3)
    grams = (4.5 if harder else 3.0) * rng.uniform(0.8, 1.2)  # harder: 1.5x heavier hose
    pump = (rng.uniform(-0.8, -0.5), rng.uniform(-0.3, 0.3), 0.0)

    return Conditions(
        set=set_name,
        index=index,
        seed=seed,
        facade=sample_facade(seed, harder=harder),
        gust_level=gust,
        glass_dropout=float(dropout),
        hose=Hose(grams_per_m=float(grams), pump_local=pump),
    )


def final_test_flights() -> list[Conditions]:
    """The 300 final test flights: every test wall at every gust level."""
    return [conditions("test", i, g) for i in range(SETS["test"][1]) for g in GUST_LEVELS]

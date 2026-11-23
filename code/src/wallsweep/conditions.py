"""Fixed wall sets: everything that defines one flight's conditions, from a set name and an index.

Each set has its own set code and a fixed number of walls:

    train: code 0, 20,000 walls. Used by NSGA-II and random search.
    val: code 1, 50 walls. Used to pick controllers from the Pareto front.
    dev: code 2, 100 walls. Former test walls, kept for development and debugging.
    dev_harder: code 3, 50 walls. Former harder walls, kept for development and debugging.
    test: code 4, 100 walls. Held out. Each wall is flown at all 3 gust levels (300 flights).
    harder: code 5, 50 walls. Held out. Deeper features, strong gusts, more glass dropouts
        and a heavier hose.

Each wall gets three independent seeds, drawn from np.random.SeedSequence([set_code, index,
stream]) with stream 0 for the facade, 1 for the gusts and 2 for the sensor noise. The gust seed
does not depend on the gust level, so one wall flown at none, moderate and strong sees gusts at
the same times. An index outside a set's size raises IndexError. check_disjoint() confirms that
no facade seed appears twice across all sets.

Per flight we randomize the facade, gust level, glass dropout rate, hose weight and pump
position. The same (set, index, gust) always gives the same flight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wallsweep.hose import Hose
from wallsweep.world import Facade, FacadeAviary, Gusts, sample_facade

GUST_LEVELS = ("none", "moderate", "strong")

SETS = {
    "train": (0, 20_000),
    "val": (1, 50),
    "dev": (2, 100),
    "dev_harder": (3, 50),
    "test": (4, 100),
    "harder": (5, 50),
}
SET_CODE = {name: code for name, (code, _) in SETS.items()}
HARDER_SETS = ("harder", "dev_harder")

FACADE_STREAM = 0
GUST_STREAM = 1
NOISE_STREAM = 2
MISC_STREAM = 642


def _seed(set_code: int, index: int, stream: int) -> int:
    ss = np.random.SeedSequence([set_code, index, stream])
    return int(ss.generate_state(1, np.uint64)[0])


@dataclass(frozen=True)
class Conditions:
    set: str
    index: int
    seed: int
    gust_seed: int
    noise_seed: int
    facade: Facade
    gust_level: str
    glass_dropout: float
    hose: Hose

    def make_env(self, gui: bool = False) -> FacadeAviary:
        """A fresh simulator for this flight."""
        return FacadeAviary(
            self.facade,
            gusts=Gusts(self.gust_level, seed=self.gust_seed),
            hose=self.hose,
            gui=gui,
            noise_seed=self.noise_seed,
            glass_dropout=self.glass_dropout,
        )

    def __str__(self) -> str:
        return (
            f"{self.set}[{self.index}] seed {self.seed}: gusts {self.gust_level}, "
            f"glass dropout {self.glass_dropout:.0%}, hose {self.hose.grams_per_m:.1f} g/m"
        )


def set_size(name: str) -> int:
    return SETS[name][1]


def conditions(set_name: str, index: int, gust: str | None = None) -> Conditions:
    """Conditions for wall `index` of a set. `gust` overrides the set's gust level (the test set
    is meant to be flown at all three)."""
    code, size = SETS[set_name]
    if index < 0 or index >= size:
        raise IndexError(f"{set_name} has {size} walls; index {index} is out of range")
    harder = set_name in HARDER_SETS
    rng = np.random.default_rng([code, index, MISC_STREAM])

    if gust is None:
        gust = "strong" if harder else str(rng.choice(GUST_LEVELS))
    elif gust not in GUST_LEVELS:
        raise ValueError(f"gust must be one of {GUST_LEVELS}")

    dropout = rng.uniform(0.3, 0.5) if harder else rng.uniform(0.0, 0.3)
    grams = (4.5 if harder else 3.0) * rng.uniform(0.8, 1.2)
    pump = (rng.uniform(-0.8, -0.5), rng.uniform(-0.3, 0.3), 0.0)

    facade_seed = _seed(code, index, FACADE_STREAM)
    return Conditions(
        set=set_name,
        index=index,
        seed=facade_seed,
        gust_seed=_seed(code, index, GUST_STREAM),
        noise_seed=_seed(code, index, NOISE_STREAM),
        facade=sample_facade(facade_seed, harder=harder),
        gust_level=gust,
        glass_dropout=float(dropout),
        hose=Hose(grams_per_m=float(grams), pump_local=pump),
    )


def check_disjoint() -> None:
    """Assert that no facade seed appears twice across all sets."""
    seeds = [_seed(code, i, FACADE_STREAM) for code, size in SETS.values() for i in range(size)]
    total = sum(size for _, size in SETS.values())
    assert len(seeds) == total
    assert len(set(seeds)) == total, f"{total - len(set(seeds))} duplicate facade seeds"


TRAINING_WALLS = ((0, "none"), (1, "moderate"), (2, "moderate"), (3, "strong"), (4, "strong"))


def training_flights() -> list[Conditions]:
    """The 5 training flights every search candidate flies: (train index, gust level)."""
    return [conditions("train", i, g) for i, g in TRAINING_WALLS]


def final_test_flights() -> list[Conditions]:
    """The 300 final test flights: every test wall at every gust level."""
    return [conditions("test", i, g) for i in range(SETS["test"][1]) for g in GUST_LEVELS]

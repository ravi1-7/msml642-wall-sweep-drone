"""Flight scores: glass coverage, time and failure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wallsweep.world import Facade

CELL_M = 0.05
FOOTPRINT_M = 0.10
BAND = (0.25, 0.35)
MIN_HEIGHT_M = 0.15
REF_ROW_M = 0.10
REF_SPEED = 0.25
TIME_MARGIN = 1.3
TIME_EXTRA_S = 20.0

SURFACE_KINDS = ("wall", "glass")
PROTRUSION_KINDS = ("sill", "hood", "ledge")


@dataclass(frozen=True)
class CleanableMap:
    """Grid of 5 cm cells over the wall face; rows are z (bottom up), columns are v."""

    cleanable: np.ndarray
    v_centers: np.ndarray
    z_centers: np.ndarray
    v_min: float
    counts: dict[str, int]


@dataclass(frozen=True)
class ScoreResult:
    coverage: float
    raw_coverage: float
    time: float
    elapsed: float
    failure: str | None
    failure_time: float | None
    T_max: float
    cleanable: int
    cleaned: int
    success: bool
    no_target: bool


def _inside_any(boxes, vv, zz) -> np.ndarray:
    hit = np.zeros(vv.shape, dtype=bool)
    for b in boxes:
        (_, cv, cz), (_, hv, hz) = b.center, b.half
        hit |= (np.abs(vv - cv) <= hv) & (np.abs(zz - cz) <= hz)
    return hit


def cleanable_map(facade: Facade) -> CleanableMap:
    """Glass cells that count for coverage, with exclusion counts under the first reason that applies."""
    v_min = -facade.width / 2
    n_v = math.ceil(facade.width / CELL_M - 1e-9)
    n_z = math.ceil(facade.height / CELL_M - 1e-9)
    v_c = v_min + (np.arange(n_v) + 0.5) * CELL_M
    z_c = (np.arange(n_z) + 0.5) * CELL_M
    vv, zz = np.meshgrid(v_c, z_c)

    surface = _inside_any([b for b in facade.boxes if b.kind in SURFACE_KINDS], vv, zz)
    glass = _inside_any([b for b in facade.boxes if b.kind == "glass"], vv, zz)
    hidden = _inside_any([b for b in facade.boxes if b.kind in PROTRUSION_KINDS], vv, zz)

    outside = ~surface
    below = ~outside & (zz < MIN_HEIGHT_M)
    hid = ~outside & ~below & hidden
    wall = ~outside & ~below & ~hid & ~glass
    cleanable = ~outside & ~below & ~hid & ~wall
    counts = {
        "total": int(vv.size),
        "cleanable": int(cleanable.sum()),
        "outside": int(outside.sum()),
        "below_min": int(below.sum()),
        "hidden": int(hid.sum()),
        "wall": int(wall.sum()),
    }
    return CleanableMap(cleanable, v_c, z_c, v_min, counts)


def time_limit(facade: Facade) -> float:
    """T_max from the reference path: rows every REF_ROW_M from the lowest footprint row to the top."""
    half = FOOTPRINT_M / 2
    z_first = MIN_HEIGHT_M + half
    z_last = max(facade.height - half, z_first)
    n_rows = math.ceil((z_last - z_first) / REF_ROW_M - 1e-9) + 1
    length = n_rows * (facade.width - FOOTPRINT_M) + (z_last - z_first)
    return TIME_MARGIN * length / REF_SPEED + TIME_EXTRA_S


class Scorer:
    """Marks sprayed glass cells during a flight and scores it at the end."""

    def __init__(self, facade: Facade):
        self.facade = facade
        self.map = cleanable_map(facade)
        self.cleanable = self.map.cleanable
        self.cleaned = np.zeros_like(self.cleanable)
        self.n_cleanable = int(self.cleanable.sum())
        self.T_max = time_limit(facade)
        self.n_z, self.n_v = self.cleanable.shape
        self.failure: str | None = None
        self.failure_time: float | None = None
        self.t = 0.0

    def update(self, env, t: float) -> None:
        """Call once per control step after env.step()."""
        self.t = t
        if self.failure is None and env.failure is not None:
            self.failure = env.failure
            self.failure_time = env.failure_time
        if self.failure is not None or not env.spraying:
            return
        u, v, z = self.facade.to_local(env.pos[0])
        if not BAND[0] <= -u <= BAND[1]:
            return
        h = FOOTPRINT_M / 2
        i0 = max(math.floor((z - h) / CELL_M), 0)
        i1 = min(math.ceil((z + h) / CELL_M), self.n_z)
        j0 = max(math.floor((v - h - self.map.v_min) / CELL_M), 0)
        j1 = min(math.ceil((v + h - self.map.v_min) / CELL_M), self.n_v)
        if i0 < i1 and j0 < j1:
            self.cleaned[i0:i1, j0:j1] |= self.cleanable[i0:i1, j0:j1]

    def finish(self, t_end: float, completed: bool) -> ScoreResult:
        """Score the flight. On failure coverage is 0 and time is T_max; time is capped at T_max."""
        n_clean = int(self.cleaned.sum())
        no_target = self.n_cleanable == 0
        raw = 0.0 if no_target else n_clean / self.n_cleanable
        failed = self.failure is not None
        coverage = 0.0 if failed else raw
        time = self.T_max if failed or not completed else min(t_end, self.T_max)
        return ScoreResult(
            coverage=coverage,
            raw_coverage=raw,
            time=time,
            elapsed=t_end,
            failure=self.failure,
            failure_time=self.failure_time,
            T_max=self.T_max,
            cleanable=self.n_cleanable,
            cleaned=n_clean,
            success=not failed and not no_target and coverage >= 0.90,
            no_target=no_target,
        )

    def save_map(self, path) -> None:
        """PNG of the grid: cleaned green, missed red, not cleanable grey."""
        from matplotlib.colors import ListedColormap
        from matplotlib.figure import Figure

        img = np.where(self.cleaned, 2, np.where(self.cleanable, 1, 0))
        cmap = ListedColormap(["#bdbdbd", "#d62728", "#2ca02c"])
        f = self.facade
        fig = Figure(figsize=(6, 6 * f.height / f.width + 0.5))
        ax = fig.add_subplot()
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest",
                  extent=(self.map.v_min, self.map.v_min + self.n_v * CELL_M, 0, self.n_z * CELL_M))
        ax.set_xlabel("v (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"seed {f.seed}: cleaned {int(self.cleaned.sum())} / {self.n_cleanable} glass cells")
        fig.savefig(path, dpi=120, bbox_inches="tight")

"""Kalman filter for the distance to the wall (the standoff), robust to glass dropouts.

The controller should not act on raw front-sensor readings: they are noisy, and on glass they
sometimes jump to "nothing in range" (see world.py). This filter keeps one number, the
estimated distance d to the wall, and its uncertainty P.

Predict (every control step): the wall doesn't move, so d changes only by the drone's own
forward speed, which the Flow deck measures:
    d <- d - vx * dt,      P <- P + Q
Update (when the reading is trusted): a standard 1D Kalman update with the sensor's noise R.
If the reading is more than 3 sigma from the estimate, the wall's shape changed (a sill or a
window recess), so the uncertainty is widened and the estimate follows the reading. A closer
surface is followed at once (safety first); a farther one only after two readings in a row,
so a single noisy reading can't throw the estimate off.

Which readings are trusted:
  * A reading of "nothing in range" (4 m) is a dropout: skip it, keep predicting.
  * A reading much FARTHER than the estimate (more than `jump_m`) is treated as a dropout too.
  * A reading CLOSER than the estimate is always trusted, however big the jump: a sill, hood
    or ledge appearing in front of the drone must never be filtered away.
  * If readings are rejected for longer than `max_reject_s`, the filter restarts from the next
    reading, so it can't stay stuck if the wall really did move away (e.g. past its edge).

All settings are fixed, set from the sensor noise model, not evolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

RANGE_MAX = 4.0


@dataclass
class StandoffKF:
    dt: float
    speed_noise: float = 0.05  # m/s, error in the Flow-deck forward speed
    noise_base: float = 0.005  # m, sensor noise at zero range (matches world.py)
    noise_frac: float = 0.01  # sensor noise grows 1 cm per metre of range
    jump_m: float = 0.25  # farther-away jumps bigger than this are treated as dropouts
    max_reject_s: float = 0.5

    def __post_init__(self):
        self.d: float | None = None  # estimated distance to the wall (m)
        self.P = 1.0  # its variance (m^2)
        self.rejected = 0  # readings rejected in a row
        self.far_step = False  # previous reading was a 3-sigma jump away from the wall
        self.total_rejected = 0

    def reset(self):
        self.__post_init__()

    def update(self, front: float, vx: float) -> float:
        """Feed one front reading (m) and the forward body speed (m/s). Returns the estimate."""
        if self.d is None:  # first usable reading starts the filter
            if front < RANGE_MAX:
                self.d, self.P = front, self._R(front)
            return front if self.d is None else self.d

        # predict
        self.d -= vx * self.dt
        self.P += (self.speed_noise * self.dt) ** 2

        # decide whether to trust the reading
        dropout = front >= RANGE_MAX - 1e-6 or front - self.d > self.jump_m
        if dropout:
            self.rejected += 1
            self.total_rejected += 1
            if self.rejected * self.dt > self.max_reject_s:
                self.d = None  # give up on the old estimate; restart from the next reading
                self.rejected = 0
                return front
            return self.d

        # update
        self.rejected = 0
        R = self._R(front)
        innov = front - self.d
        big = innov**2 > 9 * (self.P + R)
        if big and (innov < 0 or self.far_step):
            # More than 3 sigma off: the wall's shape changed (sill, hood, window recess),
            # not noise. Widen the uncertainty so the estimate jumps to the new surface.
            self.P += innov**2
        self.far_step = big and innov > 0
        K = self.P / (self.P + R)
        self.d += K * (front - self.d)
        self.P *= 1 - K
        return self.d

    def _R(self, z: float) -> float:
        return (self.noise_base + self.noise_frac * z) ** 2


class Status(str, Enum):
    """How fresh the front range data is."""

    VALID = "valid"
    COASTING = "coasting"
    STALE = "stale"
    LOST = "lost"


class Surface(str, Enum):
    """What the last front reading hit, relative to the main wall estimate."""

    WALL = "wall"
    RECESS = "recess"
    PROTRUSION = "protrusion"
    NONE = "none"


@dataclass(frozen=True)
class WallEstimate:
    """One WallEstimator output. Distances are None only when status is LOST."""

    wall: float | None
    wall_var: float | None
    nearest: float | None
    status: Status
    surface: Surface
    along: float
    blind_time: float
    reacquiring: bool
    wall_slope: float | None


class WallEstimator:
    """Tracks the main wall plane and the nearest surface ahead from one front range sensor.

    Call update() once per sweep step with the front reading (None if no new reading),
    the body velocities and dt.
    """

    USABLE_MAX = 3.5
    NOISE_BASE = 0.005
    NOISE_FRAC = 0.01
    SPEED_NOISE = 0.02
    GATE_SIGMA = 3.0
    GATE_MAX = 0.02
    RECESS_MAX = 0.08
    CONTINUITY = 0.01
    NEAREST_AGREE = 0.03
    VALID_S = 0.1
    COASTING_S = 0.3
    STALE_S = 1.0
    NO_WALL_S = 2.0
    REACQUIRE_S = 2.0
    REACQUIRE_N = 3
    REACQUIRE_AGREE = 0.01
    TIME_EPS = 1e-9
    REVERT_S = 5.0
    SLOPE_MIN_POINTS = 6
    SLOPE_MIN_SPAN_M = 0.15
    PROTRUSION_RELOCK_TRAVEL_M = 0.60
    PRED_SLOPE_SPAN_M = 0.5
    PRED_SLOPE_SPACING_M = 0.01
    PRED_SLOPE_MIN_DA = 0.15
    PRED_SLOPE_MIN_PAIRS = 100
    PRED_SLOPE_STEP_M = 0.03
    PRED_SLOPE_MAX = 0.18
    PRED_SLOPE_MAX_POINTS = 64

    def __init__(self, initial_distance: float | None = None):
        self.reset(initial_distance)

    def reset(self, initial_distance: float | None = None) -> None:
        """Start over; without an initial distance, wait for a reading in re-acquire mode."""
        self.along = 0.0
        self.forward = 0.0
        self.slope_points: list[tuple[float, float]] = []
        self.raw_points: list[tuple[float, float]] = []
        self.pred_slope = 0.0
        self.carry_slope = 0.0
        self.lost_count = 0
        self.relock_count = 0
        self.prot_travel: float | None = None
        self.prot_recent: list[float] = []
        self.reacquire_count = 0
        self.surface = Surface.NONE
        self.pending: float | None = None
        self.prev_z: float | None = None
        self.prev_wall = False
        self.prev_in_gate = False
        self.recent: list[float] = []
        self.no_wall_time = 0.0
        self.reacquire_left: float | None = None
        self.backup = None
        self.backup_left = 0.0
        self.revert_count = 0
        if initial_distance is None:
            self.wall: float | None = None
            self.nearest: float | None = None
            self.P = 0.0
            self.blind_time = math.inf
            self.reacquiring = True
        else:
            self.wall = self.nearest = float(initial_distance)
            self.P = self._R(self.wall)
            self.blind_time = 0.0
            self.reacquiring = False

    def reset_along(self, value: float = 0.0) -> None:
        """Set the along-wall position, e.g. at a detected edge."""
        shift = float(value) - self.along
        self.along = float(value)
        self.slope_points = [(a + shift, x) for a, x in self.slope_points]
        self.raw_points = [(a + shift, x) for a, x in self.raw_points]

    def reset_slope(self, yaw_change_rad: float | None = None) -> None:
        """Clear the slope points at a pass start; carry the prediction slope rotated by the yaw change, or restart it from 0."""
        base = self.wall_slope
        if base is None:
            base = self.pred_slope
        self.slope_points = []
        self.raw_points = []
        if yaw_change_rad is None:
            self.carry_slope = 0.0
        else:
            s = math.tan(math.atan(base) + float(yaw_change_rad))
            self.carry_slope = min(max(s, -self.PRED_SLOPE_MAX), self.PRED_SLOPE_MAX)
        self.pred_slope = self.carry_slope

    @property
    def wall_slope(self) -> float | None:
        """Least squares slope of wall distance vs along position over WALL readings since reset_slope()."""
        pts = self.slope_points
        if len(pts) < self.SLOPE_MIN_POINTS:
            return None
        a = [p[0] for p in pts]
        if max(a) - min(a) < self.SLOPE_MIN_SPAN_M:
            return None
        x = [p[1] for p in pts]
        ma, mx = sum(a) / len(a), sum(x) / len(x)
        saa = sum((ai - ma) ** 2 for ai in a)
        return sum((ai - ma) * (xi - mx) for ai, xi in zip(a, x)) / saa

    @property
    def status(self) -> Status:
        if self.wall is None:
            return Status.LOST
        if self.blind_time <= self.VALID_S + self.TIME_EPS:
            return Status.VALID
        if self.blind_time < self.COASTING_S - self.TIME_EPS:
            return Status.COASTING
        return Status.STALE

    @property
    def estimate(self) -> WallEstimate:
        lost = self.wall is None
        return WallEstimate(
            wall=None if lost else self.wall,
            wall_var=None if lost else self.P,
            nearest=None if lost else self.nearest,
            status=self.status,
            surface=self.surface,
            along=self.along,
            blind_time=self.blind_time,
            reacquiring=self.reacquiring,
            wall_slope=self.wall_slope,
        )

    def update(self, front: float | None, vx: float, vy: float, dt: float) -> WallEstimate:
        """Advance one sweep step and return the new estimate."""
        self.along += vy * dt
        self.forward += vx * dt
        if self.prot_travel is not None:
            self.prot_travel += abs(vy) * dt
        self.blind_time += dt
        self.no_wall_time += dt
        if self.wall is not None:
            self._predict((vx - self.pred_slope * vy) * dt, dt)
        if front is not None:
            if 0.0 <= front < self.USABLE_MAX:
                self._observe(float(front))
                self._record_slope(float(front))
                self._record_raw(float(front))
            else:
                self.surface = Surface.NONE
        if self.wall is not None and (
            self.blind_time > self.STALE_S + self.TIME_EPS or self.wall > self.USABLE_MAX
        ):
            self._lose()
        if self.reacquiring and self.reacquire_left is not None and self.reacquire_left <= self.TIME_EPS:
            self.reacquiring = False
            self.reacquire_left = None
            self.no_wall_time = 0.0
        if (
            not self.reacquiring
            and self.status is Status.VALID
            and self.no_wall_time >= self.NO_WALL_S - self.TIME_EPS
        ):
            self.reacquiring = True
            self.reacquire_left = None
        return self.estimate

    def _predict(self, shift: float, dt: float) -> None:
        self.wall = max(self.wall - shift, 0.0)
        self.nearest = min(max(self.nearest - shift, 0.0), self.wall)
        self.P += (self.SPEED_NOISE * dt) ** 2
        if self.pending is not None:
            self.pending -= shift
        if self.prev_z is not None:
            self.prev_z -= shift
        self.recent = [r - shift for r in self.recent]
        self.prot_recent = [r - shift for r in self.prot_recent]
        if self.reacquire_left is not None:
            self.reacquire_left -= dt
        if self.backup is not None:
            self.backup -= shift
            self.backup_left -= dt
            if self.backup_left <= 0:
                self.backup = None

    def _observe(self, z: float) -> None:
        self.blind_time = 0.0
        R = self._R(z)
        if self.wall is None:
            self.wall = self.nearest = z
            self.P = R
            self.surface = Surface.WALL
            self.prev_z, self.prev_wall = z, True
            self.prev_in_gate = False
            self.recent = [z]
            self.pending = None
            self.no_wall_time = 0.0
            self.reacquiring = True
            self.reacquire_left = self.REACQUIRE_S
            return

        innov = z - self.wall
        gate = min(self.GATE_SIGMA * math.sqrt(self.P + R), self.GATE_MAX)
        continuous = self.prev_wall and abs(z - self.prev_z) < self.CONTINUITY
        confirmed = False
        if continuous or (abs(innov) <= gate and (self.prev_wall or self.prev_in_gate)):
            self.surface = Surface.WALL
            confirmed = True
            K = self.P / (self.P + R)
            self.wall += K * innov
            self.P *= 1 - K
        elif abs(innov) <= gate:
            self.surface = Surface.WALL
        elif innov < 0:
            self.surface = Surface.PROTRUSION
        elif innov <= self.RECESS_MAX:
            self.surface = Surface.RECESS
        else:
            self.surface = Surface.NONE
        self.prev_z, self.prev_wall = z, confirmed
        self.prev_in_gate = self.surface is Surface.WALL and not confirmed
        self.recent = (self.recent + [z])[-self.REACQUIRE_N:]

        if self.surface is Surface.PROTRUSION:
            if self.prot_travel is None:
                self.prot_travel, self.prot_recent = 0.0, []
            self.prot_recent = (self.prot_recent + [z])[-self.REACQUIRE_N:]
            if (
                self.prot_travel > self.PROTRUSION_RELOCK_TRAVEL_M
                and len(self.prot_recent) == self.REACQUIRE_N
                and max(self.prot_recent) - min(self.prot_recent) <= self.REACQUIRE_AGREE
            ):
                mean = sum(self.prot_recent) / self.REACQUIRE_N
                if self.wall - mean <= self.RECESS_MAX:
                    self.wall = self.nearest = mean
                    self.P = self._R(mean)
                    self.surface = Surface.WALL
                    self.prev_wall, self.prev_in_gate = True, False
                    self.pending = None
                    self.no_wall_time = 0.0
                    self.prot_travel, self.prot_recent = None, []
                    self.relock_count += 1
                    return
        else:
            self.prot_travel, self.prot_recent = None, []

        if confirmed:
            self.no_wall_time = 0.0
            if self.reacquiring and self.reacquire_left is None:
                self.reacquiring = False

        if (
            self.backup is not None
            and len(self.recent) == self.REACQUIRE_N
            and max(self.recent) - min(self.recent) <= self.REACQUIRE_AGREE
        ):
            mean = sum(self.recent) / self.REACQUIRE_N
            if abs(mean - self.backup) <= self.GATE_MAX and self.wall - mean > gate:
                self.wall = self.nearest = mean
                self.P = self._R(mean)
                self.surface = Surface.WALL
                self.prev_wall, self.prev_in_gate = True, False
                self.pending = None
                self.backup = None
                self.no_wall_time = 0.0
                self.slope_points = []
                self.revert_count += 1
                return

        if (
            self.reacquiring
            and len(self.recent) == self.REACQUIRE_N
            and max(self.recent) - min(self.recent) <= self.REACQUIRE_AGREE
        ):
            mean = sum(self.recent) / self.REACQUIRE_N
            if mean - self.wall > gate:
                self.backup, self.backup_left = self.wall, self.REVERT_S
                self.wall = self.nearest = mean
                self.P = self._R(mean)
                self.surface = Surface.WALL
                self.prev_wall = True
                self.prev_in_gate = False
                self.pending = None
                self.reacquiring = False
                self.reacquire_left = None
                self.no_wall_time = 0.0
                self.slope_points = []
                self.reacquire_count += 1
                return

        self._update_nearest(z)

    def _record_slope(self, z: float) -> None:
        if self.surface is not Surface.WALL:
            return
        self.slope_points.append((self.along, z + self.forward))

    def _record_raw(self, z: float) -> None:
        pts = self.raw_points
        if pts and abs(self.along - pts[-1][0]) < self.PRED_SLOPE_SPACING_M:
            return
        pts.append((self.along, z + self.forward))
        pts = [p for p in pts if abs(p[0] - self.along) <= self.PRED_SLOPE_SPAN_M]
        self.raw_points = pts[-self.PRED_SLOPE_MAX_POINTS:]
        a = np.array([p[0] for p in self.raw_points])
        x = np.array([p[1] for p in self.raw_points])
        run = np.concatenate([[0], np.cumsum(np.abs(np.diff(x)) > self.PRED_SLOPE_STEP_M)])
        i, j = np.triu_indices(a.size, 1)
        da = a[j] - a[i]
        keep = (np.abs(da) >= self.PRED_SLOPE_MIN_DA) & (run[i] == run[j])
        if keep.sum() < self.PRED_SLOPE_MIN_PAIRS:
            self.pred_slope = self.carry_slope
            return
        s = float(np.median((x[j] - x[i])[keep] / da[keep]))
        self.pred_slope = min(max(s, -self.PRED_SLOPE_MAX), self.PRED_SLOPE_MAX)

    def _update_nearest(self, z: float) -> None:
        if self.surface is Surface.PROTRUSION and z < self.nearest:
            self.nearest = z
            self.pending = None
        elif self.surface is Surface.PROTRUSION:
            if self.pending is not None and abs(z - self.pending) <= self.NEAREST_AGREE:
                self.nearest = min(z, self.pending)
                self.pending = None
            else:
                self.pending = z
        elif self.nearest < self.wall:
            if self.pending is not None and abs(z - self.pending) <= self.NEAREST_AGREE:
                self.nearest = self.wall
                self.pending = None
            else:
                self.pending = z
        self.nearest = min(self.nearest, self.wall)

    def _lose(self) -> None:
        self.wall = self.nearest = None
        self.P = 0.0
        self.pending = self.prev_z = None
        self.prev_wall = False
        self.prev_in_gate = False
        self.recent = []
        self.surface = Surface.NONE
        self.reacquiring = True
        self.reacquire_left = None
        self.lost_count += 1
        self.backup = None
        self.prot_travel, self.prot_recent = None, []
        self.slope_points = []

    def _R(self, z: float) -> float:
        return (self.NOISE_BASE + self.NOISE_FRAC * z) ** 2

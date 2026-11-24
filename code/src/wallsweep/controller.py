"""Sweep controller: standoff PID, side-to-side sweep, sill back-off; 9 tunable settings.

Pure Python, no simulator imports. Call step() once per sweep step (SWEEP_HZ) with an
observation (height, vx, vy, yaw), the latest WallEstimate and the new front reading (None when
no new reading arrived this step). It returns an Action: a body-frame velocity command for the
Adapter, the spray flag, and requests the caller applies to the WallEstimator in this order:
reset_estimator, reset_along, then reset_slope(slope_yaw_change) when slope_yaw_change is not None.
slope_yaw_change is the drone yaw change applied at the pass end (0.0 without a trim).

States: TAKEOFF to 0.10 m, ALIGN (yaw scan and least-squares fit, then turn), CLIMB to the next
row with an extra back-off and the spray off, SWEEP along the row, INSET back onto the wall after
an edge, DONE. The first row starts with a short pass to the left edge, then the passes alternate.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass, field, fields
from enum import Enum
from typing import Protocol

import numpy as np

from wallsweep.estimator import Status, WallEstimate

SWEEP_HZ = 24
MAX_ACCEL_XY = 0.5
STANDOFF = 0.30
BAND = (0.25, 0.35)
FOOTPRINT = 0.10
MIN_HEIGHT = 0.15
RANGE_USABLE = 3.5

TAKEOFF_Z = 0.10
Z_TOL = 0.015
SETTLE_S = 0.5
ROW_HOLD_S = 0.25

SCAN_AMPLITUDE = math.radians(30.0)
SCAN_S = 6.0
FIT_WINDOW = math.radians(25.0)
FIT_GRID = 501
FIT_MIN_POINTS = 10
TURN_TOL = math.radians(0.5)
TURN_HOLD_S = 0.25
TURN_TIMEOUT_S = 3.0
YAW_KP = 2.0
YAW_RATE_MAX = math.radians(30.0)
TRIM_MAX = math.radians(1.5)

START_ALONG = 0.30
EDGE_WINDOW_FRAC = 0.10
EDGE_LOST_N = 3
EDGE_SPEED = 0.10
EDGE_TRIGGER_OFFSET = 0.02
BRAKE_MARGIN = 0.05
INSET_VALID_N = 2
INSET_MAX = 0.15
COASTING_FACTOR = 0.5

CLIMB_BACKOFF = 0.04
WALL_MAX = 0.60  # a wall estimate farther than this is not the wall (e.g. the ground past an edge)
GUARD_M = 0.15  # any surface closer than this on the front sensor: back away (= lowest sill_response)
GUARD_SPEED = 0.10
CLIMB_BACKOFF_TOL = 0.01
CLIMB_WAIT_MAX_S = 1.5
LEDGE_DROP = 0.08
LEDGE_BAND = (-0.01, 0.05)
LEDGE_CLEAR = 0.005

KP_CAP = 5.5
I_LIMIT = 0.05
VX_MAX = 0.30

BOUNDS: dict[str, tuple[float, float]] = {
    "standoff_kp": (0.3, 5.5),
    "standoff_ki": (0.0, 2.0),
    "standoff_kd": (0.0, 0.5),
    "sweep_speed": (0.10, 0.50),
    "vertical_step": (0.04, 0.09),
    "sill_response": (0.15, 0.28),
    "yaw_trim_gain": (0.0, 1.0),
    "cmd_smoothing": (0.0, 0.8),
    "height_gain_scale": (0.0, 0.5),
}


@dataclass(frozen=True)
class Params:
    """The 9 evolved settings. Gains act on the standoff error in metres; speeds in m/s."""

    standoff_kp: float
    standoff_ki: float
    standoff_kd: float
    sweep_speed: float
    vertical_step: float
    sill_response: float
    yaw_trim_gain: float
    cmd_smoothing: float
    height_gain_scale: float

    def __post_init__(self):
        for name, value in zip(self.names(), astuple(self)):
            lo, hi = BOUNDS[name]
            if not lo - 1e-12 <= value <= hi + 1e-12:
                raise ValueError(f"{name}={value} outside [{lo}, {hi}]")

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(f.name for f in fields(Params))

    @staticmethod
    def lower() -> np.ndarray:
        return np.array([BOUNDS[n][0] for n in Params.names()])

    @staticmethod
    def upper() -> np.ndarray:
        return np.array([BOUNDS[n][1] for n in Params.names()])

    def to_vector(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_vector(cls, x) -> Params:
        """Build from a length-9 vector, clipped into the bounds."""
        x = np.clip(np.asarray(x, dtype=float), cls.lower(), cls.upper())
        if x.shape != (len(BOUNDS),):
            raise ValueError(f"expected {len(BOUNDS)} values, got shape {x.shape}")
        return cls(*(float(v) for v in x))


HAND_TUNED = Params(
    standoff_kp=1.5,
    standoff_ki=0.2,
    standoff_kd=0.1,
    sweep_speed=0.25,
    vertical_step=0.08,
    sill_response=0.20,
    yaw_trim_gain=0.5,
    cmd_smoothing=0.3,
    height_gain_scale=0.2,
)


class Obs(Protocol):
    height: float
    vx: float
    vy: float
    yaw: float


class State(str, Enum):
    TAKEOFF = "takeoff"
    ALIGN = "align"
    CLIMB = "climb"
    SWEEP = "sweep"
    INSET = "inset"
    DONE = "done"


@dataclass(frozen=True)
class Action:
    """Body-frame velocity (m/s), yaw rate (rad/s), target height (m), spray flag and
    requests for the WallEstimator."""

    vx: float
    vy: float
    yaw_rate: float
    z: float
    spraying: bool
    done: bool
    state: State
    reset_estimator: float | None = None
    reset_along: float | None = None
    slope_yaw_change: float | None = None


@dataclass
class Stats:
    fit: tuple[float, float] | None = None
    passes: list[tuple[float, int, bool]] = field(default_factory=list)
    ledges: list[tuple[float, float]] = field(default_factory=list)
    trims: list[float] = field(default_factory=list)


def wrap(a):
    """Angle(s) wrapped to [-pi, pi)."""
    if isinstance(a, np.ndarray):
        return (a + math.pi) % math.tau - math.pi
    return math.remainder(float(a), math.tau)


def fit_wall_yaw(samples, center: float | None = None) -> tuple[float, float] | None:
    """Least-squares fit of r = (d - p.n) / cos(yaw - yaw0) to (range, yaw) or
    (range, yaw, x, y) samples.

    (x, y) is how far the drone has moved since the scan started (world frame, from the Flow
    deck), and n = (cos yaw0, sin yaw0) points at the wall, so a gust that pushes the drone
    during the scan is not mistaken for a different wall angle. Without (x, y) the drone is
    taken to stay still. Readings at or past RANGE_USABLE are ignored. Returns (yaw0, d), with d
    the distance from the scan's starting point, or None with too few points.
    """
    pts = [(s[0], s[1], *(s[2:4] or (0.0, 0.0))) for s in samples
           if s[0] is not None and 0.0 < s[0] < RANGE_USABLE]
    if len(pts) < FIT_MIN_POINTS:
        return None
    r0 = np.array([p[0] for p in pts])
    yaw = np.array([p[1] for p in pts])
    px = np.array([p[2] for p in pts])
    py = np.array([p[3] for p in pts])
    if center is None:
        center = math.atan2(np.sin(yaw).mean(), np.cos(yaw).mean())

    def sse(y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cos = np.cos(wrap(yaw[None, :] - y0[:, None]))
        bad = (cos < 0.1).any(axis=1)
        c = 1.0 / np.maximum(cos, 0.1)
        moved = px[None, :] * np.cos(y0)[:, None] + py[None, :] * np.sin(y0)[:, None]
        r = r0[None, :] + moved * c  # reading had the drone stayed at the start
        d = (c * r).sum(axis=1) / (c * c).sum(axis=1)
        err = ((r - d[:, None] * c) ** 2).sum(axis=1)
        return np.where(bad, np.inf, err), d

    grid = center + np.linspace(-FIT_WINDOW, FIT_WINDOW, FIT_GRID)
    err, _ = sse(grid)
    i = int(np.argmin(err))
    step = grid[1] - grid[0]
    fine = grid[i] + np.linspace(-step, step, 201)
    err, d = sse(fine)
    j = int(np.argmin(err))
    if not np.isfinite(err[j]):
        return None
    return float(wrap(fine[j])), float(d[j])


class SweepController:
    """Flies one wall of known width and height from the bottom-left start."""

    def __init__(self, params: Params, width: float, height: float, dt: float = 1.0 / SWEEP_HZ):
        self.p = params
        self.width = float(width)
        self.height = float(height)
        self.dt = float(dt)
        self.z_first = MIN_HEIGHT + FOOTPRINT / 2
        self.z_last = max(self.height - FOOTPRINT / 2, self.z_first)
        self.reset()

    def reset(self) -> None:
        self.state = State.TAKEOFF
        self.t_state = 0.0
        self.hold = 0.0
        self.wall: float | None = None
        self.nearest: float | None = None
        self.integral = 0.0
        self.out = np.zeros(2)
        self.yaw_target: float | None = None
        self.scan: list[tuple[float, ...]] = []
        self.scan_pos = np.zeros(2)
        self.scan_center: float | None = None
        self.turning = False
        self.turn_t = 0.0
        self.dir = -1
        self.homing = True
        self.row_z = self.z_first
        self.row_target: float | None = self.z_first
        self.z_cmd = TAKEOFF_Z
        self.front_last: float | None = None
        self.climb_prev_front: float | None = None
        self.lost_run = 0
        self.valid_run = 0
        self.inset_from: float | None = None
        self.raising = False
        self.ledges: list[tuple[float, float]] = []
        self.stats = Stats(ledges=self.ledges)
        self._req: dict = {}
        self._entered = False

    def step(self, obs: Obs, est: WallEstimate, front: float | None = None) -> Action:
        self._track(obs, est)
        if front is not None:
            self.front_last = float(front) if front < RANGE_USABLE else None
        if self.yaw_target is None:
            self.yaw_target = float(obs.yaw)
        self._req = {}
        self._entered = False
        handler = {
            State.TAKEOFF: self._takeoff,
            State.ALIGN: self._align,
            State.CLIMB: self._climb,
            State.SWEEP: self._sweep,
            State.INSET: self._inset,
            State.DONE: self._done,
        }[self.state]
        vx, vy, yaw_rate, spraying = handler(obs, est, front)
        if not self._entered:
            self.t_state += self.dt
        a = self.p.cmd_smoothing
        self.out = a * self.out + (1.0 - a) * np.array([vx, vy])
        return Action(
            vx=float(self.out[0]),
            vy=float(self.out[1]),
            yaw_rate=float(yaw_rate),
            z=float(self.z_cmd),
            spraying=bool(spraying),
            done=self.state is State.DONE,
            state=self.state,
            reset_estimator=self._req.get("reset_estimator"),
            reset_along=self._req.get("reset_along"),
            slope_yaw_change=self._req.get("slope_yaw_change"),
        )

    def _enter(self, state: State) -> None:
        self.state = state
        self.t_state = 0.0
        self.hold = 0.0
        self.lost_run = 0
        self.valid_run = 0
        self.inset_from = None
        self.raising = False
        self.climb_prev_front = None
        self._entered = True

    def _track(self, obs: Obs, est: WallEstimate) -> None:
        if est.wall is not None and est.wall <= WALL_MAX:
            self.wall, self.nearest = est.wall, est.nearest
        elif self.wall is not None:
            self.wall -= obs.vx * self.dt
            self.nearest = min(self.nearest - obs.vx * self.dt, self.wall)

    def gains(self, height: float) -> tuple[float, float, float]:
        g = 1.0 + self.p.height_gain_scale * max(float(height), 0.0)
        return min(self.p.standoff_kp * g, KP_CAP), self.p.standoff_ki * g, self.p.standoff_kd * g

    def _standoff(self, obs: Obs, est: WallEstimate, target: float) -> float:
        if self.wall is None:
            return 0.0
        kp, ki, kd = self.gains(obs.height)
        e = self.wall - target
        if est.status in (Status.VALID, Status.COASTING):
            self.integral = float(np.clip(self.integral + e * self.dt, -I_LIMIT, I_LIMIT))
        vx = kp * e + ki * self.integral - kd * obs.vx
        if self.nearest is not None and self.nearest < self.p.sill_response:
            vx = min(vx, kp * (self.nearest - self.p.sill_response))
        if self.front_last is not None and self.front_last < GUARD_M:
            vx = min(vx, -GUARD_SPEED)
        return float(np.clip(vx, -VX_MAX, VX_MAX))

    def _yaw_rate(self, obs: Obs) -> float:
        err = wrap(self.yaw_target - obs.yaw)
        return float(np.clip(YAW_KP * err, -YAW_RATE_MAX, YAW_RATE_MAX))

    def _takeoff(self, obs, est, front):
        self.z_cmd = TAKEOFF_Z
        self.hold = self.hold + self.dt if abs(obs.height - TAKEOFF_Z) < Z_TOL else 0.0
        if self.hold >= SETTLE_S - 1e-9:
            self._enter(State.ALIGN)
            self.scan_center = float(obs.yaw)
            self.scan = []
            self.scan_pos = np.zeros(2)
            self.turning = False
        return 0.0, 0.0, 0.0, False

    def _align(self, obs, est, front):
        self.z_cmd = TAKEOFF_Z
        if not self.turning:
            c, s = math.cos(obs.yaw), math.sin(obs.yaw)
            self.scan_pos += np.array([c * obs.vx - s * obs.vy, s * obs.vx + c * obs.vy]) * self.dt
            if front is not None:
                self.scan.append((float(front), float(obs.yaw), *self.scan_pos))
            w = math.tau / SCAN_S
            rate = SCAN_AMPLITUDE * w * math.cos(w * self.t_state)
            if self.t_state + self.dt >= SCAN_S - 1e-9:
                fit = fit_wall_yaw(self.scan, self.scan_center)
                if fit is not None:  # distance from where the drone is now, not where it started
                    n = np.array([math.cos(fit[0]), math.sin(fit[0])])
                    fit = (fit[0], fit[1] - float(self.scan_pos @ n))
                self.stats.fit = fit
                self.yaw_target = fit[0] if fit is not None else self.scan_center
                self.turning = True
                self.turn_t = 0.0
                self.hold = 0.0
            return 0.0, 0.0, rate, False
        self.turn_t += self.dt
        self.hold = self.hold + self.dt if abs(wrap(self.yaw_target - obs.yaw)) < TURN_TOL else 0.0
        rate = self._yaw_rate(obs)
        if self.hold >= TURN_HOLD_S - 1e-9 or self.turn_t >= TURN_TIMEOUT_S - 1e-9:
            if self.stats.fit is not None:
                self._req["reset_estimator"] = self.stats.fit[1]
                self.wall = self.nearest = self.stats.fit[1]
            self._req["reset_along"] = START_ALONG
            self.row_target = self.z_first
            self._enter(State.CLIMB)
            rate = 0.0
        return 0.0, 0.0, rate, False

    def _climb(self, obs, est, front):
        target = STANDOFF + CLIMB_BACKOFF
        vx = self._standoff(obs, est, target)
        if not self.raising and (
            (self.wall is not None and self.wall >= target - CLIMB_BACKOFF_TOL)
            or self.t_state >= CLIMB_WAIT_MAX_S - 1e-9
        ):
            self.raising = True
        if front is not None and front < RANGE_USABLE:
            if (
                self.raising
                and self.climb_prev_front is not None
                and self.climb_prev_front < WALL_MAX
                and self.climb_prev_front - front > LEDGE_DROP
            ):
                self._log_ledge(obs.height)
            self.climb_prev_front = front
        if self.raising and self.state is State.CLIMB:
            self.z_cmd = self.row_target
            self.hold = self.hold + self.dt if abs(obs.height - self.row_target) < Z_TOL else 0.0
            if self.hold >= ROW_HOLD_S - 1e-9:
                self.row_z = self.row_target
                self._enter(State.SWEEP)
        return vx, 0.0, self._yaw_rate(obs), False

    def _sweep(self, obs, est, front):
        self.z_cmd = self.row_z
        vx = self._standoff(obs, est, STANDOFF)
        expected = self.width if self.dir > 0 else 0.0
        window = EDGE_WINDOW_FRAC * self.width
        if front is not None:
            self.lost_run = self.lost_run + 1 if front >= RANGE_USABLE else 0
        if abs(est.along - expected) <= window and self.lost_run >= EDGE_LOST_N:
            self._pass_end(est, expected, anchored=True)
            return vx, 0.0, self._yaw_rate(obs), False
        if self.dir * (est.along - expected) > window:
            self._pass_end(est, expected, anchored=False)
            return vx, 0.0, self._yaw_rate(obs), False

        speed = self.p.sweep_speed
        remaining = self.dir * (expected - est.along)
        brake = max(speed**2 - EDGE_SPEED**2, 0.0) / (2 * MAX_ACCEL_XY) + BRAKE_MARGIN
        if remaining < brake:
            speed = min(speed, EDGE_SPEED)
        speed *= {Status.VALID: 1.0, Status.COASTING: COASTING_FACTOR}.get(est.status, 0.0)
        spraying = (
            est.status in (Status.VALID, Status.COASTING)
            and est.wall is not None
            and BAND[0] <= est.wall <= BAND[1]
        )
        return vx, self.dir * speed, self._yaw_rate(obs), spraying

    def _pass_end(self, est: WallEstimate, expected: float, anchored: bool) -> None:
        if anchored:
            self._req["reset_along"] = expected + self.dir * EDGE_TRIGGER_OFFSET
        yaw_change = 0.0
        if est.wall_slope is not None:
            delta = float(np.clip(self.p.yaw_trim_gain * math.atan(est.wall_slope),
                                  -TRIM_MAX, TRIM_MAX))
            self.yaw_target = wrap(self.yaw_target - delta)
            self.stats.trims.append(delta)
            yaw_change = -delta
        self._req["slope_yaw_change"] = yaw_change
        self.stats.passes.append((self.row_z, self.dir, anchored))
        self.dir = -self.dir
        if self.homing:
            self.homing = False
            self.row_target = None
        else:
            self.row_target = self._next_row()
            if self.row_target is None:
                self._enter(State.DONE)
                return
        self._enter(State.INSET)

    def _inset(self, obs, est, front):
        self.z_cmd = self.row_z
        vx = self._standoff(obs, est, STANDOFF)
        if self.inset_from is None:
            self.inset_from = est.along
        if front is not None:
            self.valid_run = self.valid_run + 1 if front < RANGE_USABLE else 0
        if self.valid_run >= INSET_VALID_N or abs(est.along - self.inset_from) >= INSET_MAX:
            self._enter(State.SWEEP if self.row_target is None else State.CLIMB)
            return vx, 0.0, self._yaw_rate(obs), False
        return vx, self.dir * EDGE_SPEED, self._yaw_rate(obs), False

    def _done(self, obs, est, front):
        self.z_cmd = self.row_z
        return self._standoff(obs, est, STANDOFF), 0.0, self._yaw_rate(obs), False

    def _overlapping(self, z: float) -> tuple[float, float] | None:
        h = FOOTPRINT / 2
        for band in self.ledges:
            if z + h > band[0] and z - h < band[1]:
                return band
        return None

    def _place(self, z: float, prev: float) -> float | None:
        for _ in range(len(self.ledges) + 2):
            if z > self.z_last + 1e-9:
                if prev >= self.z_last - 1e-9:
                    return None
                z = self.z_last
            band = self._overlapping(z)
            if band is None:
                return z
            if z >= self.z_last - 1e-9:
                return None
            z = band[1] + FOOTPRINT / 2 + LEDGE_CLEAR
        return None

    def _next_row(self) -> float | None:
        return self._place(self.row_z + self.p.vertical_step, self.row_z)

    def row_plan(self) -> list[float]:
        """Row heights from the current row up, given the ledges logged so far."""
        rows, z = [self.row_z], self.row_z
        while (z := self._place(z + self.p.vertical_step, z)) is not None:
            rows.append(z)
        return rows

    def _log_ledge(self, z: float) -> None:
        if any(lo <= z <= hi for lo, hi in self.ledges):
            return
        self.ledges.append((z + LEDGE_BAND[0], z + LEDGE_BAND[1]))
        if self.row_target is not None and self._overlapping(self.row_target) is not None:
            self.row_target = self._place(self.row_target, self.row_z)
            if self.row_target is None:
                self._enter(State.DONE)

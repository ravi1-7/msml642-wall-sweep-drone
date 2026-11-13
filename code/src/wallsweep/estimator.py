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

from dataclasses import dataclass

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

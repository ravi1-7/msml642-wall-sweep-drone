"""One complete simulated flight: world, sensors, estimator, sweep controller, adapter and scorer.

Rates (see adapter.py): physics 240 Hz, motor control 48 Hz, estimator and sweep controller
24 Hz, new front range reading 12 Hz. Each env.step() is one 48 Hz control step.

The flight ends when the controller reports DONE, when the time limit T_max is reached, or on
the first failure (facade contact, ground, tumble, out of bounds). The result is the score from
scoring.py plus a few numbers that explain it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wallsweep.adapter import PID_HZ, SWEEP_HZ, Adapter, Command, schedule
from wallsweep.conditions import Conditions
from wallsweep.controller import BAND, STANDOFF, Params, SweepController
from wallsweep.estimator import WallEstimator
from wallsweep.scoring import Scorer, ScoreResult


@dataclass
class Flight:
    score: ScoreResult
    completed: bool  # the controller reached DONE
    end_state: str  # controller state when the flight ended
    sweep_steps: int
    spray_steps: int  # sweep steps with the spray on
    in_band_steps: int  # of those, steps truly 25 to 35 cm from the main wall
    rms_standoff_error: float | None  # metres, while spraying, against the true distance
    min_standoff: float | None  # metres, closest true distance to the main wall face once airborne
    ledges_logged: int
    trace: list[tuple] = field(default_factory=list)  # filled when fly(..., trace=True)
    scorer: Scorer | None = None  # kept when fly(..., trace=True), for scorer.save_map()

    @property
    def in_band_fraction(self) -> float | None:
        return self.in_band_steps / self.spray_steps if self.spray_steps else None


def fly(params: Params, cond: Conditions, gui: bool = False, trace: bool = False,
        on_sweep_step: Callable | None = None) -> Flight:
    """Fly one wall with the given settings and return its score.

    `on_sweep_step(env, t, action, estimate)` is called after every sweep step, for example to
    draw the flight in the simulator window."""
    env = cond.make_env(gui=gui)
    try:
        return _fly(params, cond, env, trace, on_sweep_step)
    finally:
        env.close()


def _fly(params: Params, cond: Conditions, env, trace: bool, on_sweep_step) -> Flight:
    facade = cond.facade
    obs, _ = env.reset()
    adapter = Adapter()
    adapter.reset(obs[0])
    controller = SweepController(params, facade.width, facade.height)
    estimator = WallEstimator()
    scorer = Scorer(facade)

    dt = 1.0 / PID_HZ
    sweep_dt = 1.0 / SWEEP_HZ
    max_steps = int(np.ceil(scorer.T_max * PID_HZ))
    completed = False
    sweep_steps = spray_steps = in_band = 0
    sq_err = 0.0
    min_standoff = None
    rows = []
    t = 0.0

    for k in range(max_steps):
        when = schedule(k)
        if when.run_sweep:
            s = env.sense()
            front = s.front if when.new_range else None
            est = estimator.update(front, s.vx, s.vy, sweep_dt)
            a = controller.step(s, est, front)
            if a.reset_estimator is not None:
                estimator.reset(a.reset_estimator)
            if a.reset_along is not None:
                estimator.reset_along(a.reset_along)
            if a.slope_yaw_change is not None:
                estimator.reset_slope(a.slope_yaw_change)
            adapter.set_command(Command(a.vx, a.vy, a.yaw_rate, a.z))
            env.spraying = a.spraying

            u, v, z = facade.to_local(env.pos[0])
            standoff = -float(u)
            sweep_steps += 1
            if z > 0.15:
                min_standoff = standoff if min_standoff is None else min(min_standoff, standoff)
            if a.spraying:
                spray_steps += 1
                in_band += BAND[0] <= standoff <= BAND[1]
                sq_err += (standoff - STANDOFF) ** 2
            if trace:
                rows.append((t, float(v), float(z), standoff, est.wall, a.state.value, a.spraying))
            if on_sweep_step is not None:
                on_sweep_step(env, t, a, est)
            if a.done:
                completed = True
                break

        obs, *_ = env.step(adapter.compute(obs[0], dt)[None, :])
        t = (k + 1) * dt
        scorer.update(env, t)
        if env.failure is not None:
            break

    return Flight(
        score=scorer.finish(t, completed),
        completed=completed,
        end_state=controller.state.value,
        sweep_steps=sweep_steps,
        spray_steps=spray_steps,
        in_band_steps=in_band,
        rms_standoff_error=float(np.sqrt(sq_err / spray_steps)) if spray_steps else None,
        min_standoff=min_standoff,
        ledges_logged=len(controller.stats.ledges),
        trace=rows,
        scorer=scorer if trace else None,
    )

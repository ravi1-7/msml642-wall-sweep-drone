"""Fly the sweep controller on chosen walls and print the scores.

Examples (from the code/ folder):
    uv run python scripts/fly_walls.py                          # hand-tuned settings, 5 training walls
    uv run python scripts/fly_walls.py --set dev_val --index 0 1 2  # old validation walls at their own gusts
    uv run python scripts/fly_walls.py --set train --index 4 --gust strong --gui   # watch a flight
    uv run python scripts/fly_walls.py --gui --speed 8          # watch all 5, 8x faster than real time
    uv run python scripts/fly_walls.py --set train --index 4 --gust strong --change sill_response=0.28
    uv run python scripts/fly_walls.py --maps results/maps      # also save cleaned/missed glass maps
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from demo import add_heldout_flag, camera, check_heldout

from wallsweep.conditions import SETS, conditions, training_flights
from wallsweep.controller import HAND_TUNED
from wallsweep.fly import fly


class Viewer:
    """Draws the flight in the simulator window and keeps it to `speed` x real time.

    Trail: green while spraying inside the 25-35 cm band, red while spraying outside it,
    grey with the spray off. The label shows the controller state and the distances.
    Drawing is the slow part, so the trail is drawn 8 times and the label 4 times per second
    of flight."""

    def __init__(self, cond, speed):
        self.facade, self.speed = cond.facade, speed
        self.last = None
        self.label = None
        self.start = None
        self.n = 0

    def __call__(self, env, t, a, est):
        import pybullet as p

        cid = env.CLIENT
        if self.start is None:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
            target, _, dist, yaw, pitch = camera(self.facade)
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target.tolist(), physicsClientId=cid)
            self.start = time.perf_counter()
        self.n += 1
        if self.n % 3:
            return
        pos = env.pos[0].tolist()
        standoff = -float(self.facade.to_local(env.pos[0])[0])
        if a.spraying:
            color = [0, 0.7, 0] if 0.25 <= standoff <= 0.35 else [0.9, 0, 0]
        else:
            color = [0.5, 0.5, 0.5]
        if self.last is not None:
            p.addUserDebugLine(self.last, pos, color, 3, physicsClientId=cid)
        self.last = pos
        wall = "-" if est.wall is None else f"{100 * est.wall:.0f}"
        text = f"{a.state.value}  true {100 * standoff:.0f} cm  estimate {wall} cm  t {t:.0f} s"
        if self.n % 6 == 0:
            kw = {"replaceItemUniqueId": self.label} if self.label is not None else {}
            self.label = p.addUserDebugText(text, [pos[0], pos[1], pos[2] + 0.25], [0, 0, 0], 1.2,
                                            physicsClientId=cid, **kw)
        ahead = t / self.speed - (time.perf_counter() - self.start)
        if ahead > 0:
            time.sleep(ahead)


def one(args):
    cond, gui, maps, params, speed = args
    t0 = time.perf_counter()
    viewer = Viewer(cond, speed) if gui else None
    f = fly(params, cond, gui=gui, trace=maps is not None, on_sweep_step=viewer)
    if maps is not None:
        save_map(cond, f, maps)
    return cond, f, time.perf_counter() - t0


def save_map(cond, f, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f.scorer.save_map(folder / f"{cond.set}_{cond.index}_{cond.gust_level}.png")
    f.scorer = None  # not needed back in the main process


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), help="fly walls from this set instead of the training walls")
    ap.add_argument("--index", type=int, nargs="+", default=[0], help="wall numbers within --set")
    ap.add_argument("--gust", choices=["none", "moderate", "strong"], help="override the gust level")
    ap.add_argument("--gui", action="store_true", help="watch the flight (one wall at a time)")
    ap.add_argument("--maps", help="folder for glass maps (cleaned green, missed red)")
    ap.add_argument("--workers", type=int, default=None, help="parallel flights (default: all cores)")
    ap.add_argument("--speed", type=float, default=4.0, help="with --gui: times faster than real time")
    ap.add_argument("--change", nargs="+", default=[], metavar="NAME=VALUE",
                    help="change hand-tuned settings, e.g. sill_response=0.28 sweep_speed=0.4")
    add_heldout_flag(ap)
    a = ap.parse_args()
    check_heldout(ap, a)

    params = replace(HAND_TUNED, **{k: float(v) for k, v in (c.split("=") for c in a.change)})
    if a.change:
        print("settings:", ", ".join(f"{k}={v:g}" for k, v in zip(params.names(), params.to_vector())))

    conds = [conditions(a.set, i, a.gust) for i in a.index] if a.set else training_flights()
    jobs = [(c, a.gui, a.maps, params, a.speed) for c in conds]
    if a.gui:
        import pybullet

        results = []
        for j in jobs:
            try:
                results.append(one(j))
            except pybullet.error:
                print("The simulator window was closed, so the flight stopped.")
                break
        if not results:
            return 0
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            results = list(pool.map(one, jobs))

    print(f"{'wall':<18}{'gusts':<10}{'coverage':>9}{'time':>8}{'T_max':>8}{'done':>6}"
          f"{'in band':>9}{'rms cm':>8}{'min cm':>8}  failure / end state   run s")
    for c, f, secs in results:
        s = f.score
        band = "-" if f.in_band_fraction is None else f"{f.in_band_fraction:.0%}"
        rms = "-" if f.rms_standoff_error is None else f"{100 * f.rms_standoff_error:.1f}"
        mn = "-" if f.min_standoff is None else f"{100 * f.min_standoff:.1f}"
        print(f"{c.set + '[' + str(c.index) + ']':<18}{c.gust_level:<10}{s.raw_coverage:>9.1%}{s.time:>8.1f}"
              f"{s.T_max:>8.1f}{'yes' if f.completed else 'no':>6}{band:>9}{rms:>8}{mn:>8}  "
              f"{(s.failure or '-') + ' / ' + f.end_state:<21}{secs:>6.1f}")
    ok = [f for _, f, _ in results]
    print(f"mean coverage {sum(f.score.coverage for f in ok) / len(ok):.1%} "
          f"(failed flights count 0), mean time {sum(f.score.time for f in ok) / len(ok):.1f} s, "
          f"failures {sum(f.score.failure is not None for f in ok)}/{len(ok)}")


if __name__ == "__main__":
    sys.exit(main())

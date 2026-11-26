from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet

from wallsweep.conditions import GUST_LEVELS, TEST_SET, conditions, final_test_flights, set_size
from wallsweep.controller import HAND_TUNED, Params
from wallsweep.fly import fly
from wallsweep.nsga import _git_hash, _quiet
from wallsweep.validate import HAND, NAMES

HARDER_SET = "harder"
CONTROLLERS = (("nsga", "fast"), ("nsga", "balanced"), ("nsga", "thorough"), ("random", "pick"))

FLIGHT_FIELDS = (
    "setting", *NAMES, "set", "wall", "gust", "coverage", "raw_coverage", "time", "T_max",
    "time_ratio", "failure", "completed", "end_state", "min_standoff", "rms_standoff", "ledges",
    "run_s",
)
SUMMARY_FIELDS = (
    "controller", "setting", "set", "gust", "flights", "fail", "coverage", "time_ratio", "time",
    "done", "min_standoff",
)


def test_flights() -> list[tuple[str, int, str]]:
    """(set, wall, gust) for the 300 test2 flights and the 50 harder flights."""
    out = [(c.set, c.index, c.gust_level) for c in final_test_flights()]
    out += [(HARDER_SET, i, conditions(HARDER_SET, i).gust_level)
            for i in range(set_size(HARDER_SET))]
    return out


def fly_one(job) -> dict:
    setting, x, set_name, index, gust = job
    cond = conditions(set_name, index, gust)
    base = {"setting": setting, **dict(zip(NAMES, map(float, x))), "set": set_name,
            "wall": index, "gust": gust}
    t0 = time.perf_counter()
    try:
        f = fly(Params.from_vector(x), cond)
    except (pybullet.error, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        return {
            **base, "coverage": 0.0, "raw_coverage": 0.0, "time": math.nan, "T_max": math.nan,
            "time_ratio": 1.0, "failure": f"error:{type(e).__name__}:{e}", "completed": False,
            "end_state": "error", "min_standoff": None, "rms_standoff": None, "ledges": 0,
            "run_s": time.perf_counter() - t0,
        }
    s = f.score
    return {
        **base, "coverage": s.coverage, "raw_coverage": s.raw_coverage, "time": s.time,
        "T_max": s.T_max, "time_ratio": s.time / s.T_max, "failure": s.failure or "",
        "completed": f.completed, "end_state": f.end_state, "min_standoff": f.min_standoff,
        "rms_standoff": f.rms_standoff_error, "ledges": f.ledges_logged,
        "run_s": time.perf_counter() - t0,
    }


def load_choice(path: Path) -> dict:
    choice = json.loads(path.read_text())
    if choice.get("provisional", True):
        sys.exit(f"{path} is a provisional validation run; the final test needs seeds 0, 1 and 2")
    return choice


def load_controllers(val_dirs: dict[str, Path]) -> tuple[dict[str, str | None], dict[str, list]]:
    """Controller name -> setting id (None when the pick is missing), setting id -> vector."""
    names: dict[str, str | None] = {HAND: HAND}
    vectors: dict[str, list] = {HAND: [float(v) for v in HAND_TUNED.to_vector()]}
    for method, pick in CONTROLLERS:
        p = load_choice(val_dirs[method] / "choice.json")["picks"].get(pick)
        name = f"{method}_{pick}"
        if p is None:
            names[name] = None
            continue
        names[name] = p["setting"]
        vectors[p["setting"]] = [float(p["params"][n]) for n in NAMES]
    return names, vectors


def read_flights(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen, out = set(), []
    for r in rows:
        key = (r["setting"], r["set"], r["wall"], r["gust"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**r, "wall": int(r["wall"]), "coverage": float(r["coverage"]),
                    "time": math.nan if r["time"] in ("", "nan") else float(r["time"]),
                    "time_ratio": float(r["time_ratio"]), "completed": r["completed"] == "True",
                    "min_standoff": (math.nan if r["min_standoff"] in ("", "None")
                                     else float(r["min_standoff"]))})
    return out


def group_row(controller: str, setting: str, set_name: str, gust: str, rows: list[dict]) -> dict:
    times = [r["time"] for r in rows if not math.isnan(r["time"])]
    close = [r["min_standoff"] for r in rows if not math.isnan(r["min_standoff"])]
    return {
        "controller": controller, "setting": setting, "set": set_name, "gust": gust,
        "flights": len(rows), "fail": sum(bool(r["failure"]) for r in rows),
        "coverage": float(np.mean([r["coverage"] for r in rows])),
        "time_ratio": float(np.mean([r["time_ratio"] for r in rows])),
        "time": float(np.mean(times)) if times else math.nan,
        "done": sum(r["completed"] for r in rows),
        "min_standoff": min(close) if close else math.nan,
    }


def summarize(names: dict[str, str | None], flights: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in flights:
        by[(r["setting"], r["set"])].append(r)
    out = []
    for controller, setting in names.items():
        if setting is None:
            continue
        for set_name, gusts in ((TEST_SET, ("all", *GUST_LEVELS)), (HARDER_SET, ("all",))):
            rows = by[(setting, set_name)]
            for g in gusts:
                sel = rows if g == "all" else [r for r in rows if r["gust"] == g]
                if sel:
                    out.append(group_row(controller, setting, set_name, g, sel))
    return out


def report(names: dict[str, str | None], summary: list[dict]) -> None:
    for set_name, n in ((TEST_SET, 300), (HARDER_SET, 50)):
        print(f"\n{set_name} ({n} flights per controller)")
        print(f"{'controller':<15}{'setting':<10}{'fail':>8}{'coverage':>10}{'t/T':>8}"
              f"{'mean s':>8}{'done':>6}{'closest':>9}")
        for controller, setting in names.items():
            if setting is None:
                print(f"{controller:<15}missing (no pick on val2)")
                continue
            s = next(r for r in summary if r["controller"] == controller and r["set"] == set_name
                     and r["gust"] == "all")
            print(f"{controller:<15}{setting:<10}{s['fail']:>4}/{s['flights']:<3}"
                  f"{s['coverage']:>10.2%}{s['time_ratio']:>8.4f}{s['time']:>8.1f}{s['done']:>6}"
                  f"{s['min_standoff'] * 100:>7.1f}cm")
    print(f"\n{TEST_SET} by gust level (failures, coverage, t/T)")
    print(f"{'controller':<15}" + "".join(f"{g:>26}" for g in GUST_LEVELS))
    for controller, setting in names.items():
        if setting is None:
            continue
        cells = []
        for g in GUST_LEVELS:
            s = next(r for r in summary if r["controller"] == controller and r["set"] == TEST_SET
                     and r["gust"] == g)
            cells.append(f"{s['fail']:>3}/{s['flights']:<3} {s['coverage']:>7.2%} "
                         f"{s['time_ratio']:.3f}")
        print(f"{controller:<15}" + "".join(f"{c:>26}" for c in cells))


def run(val_dirs: dict[str, Path], out: Path, workers: int | None, resume: bool) -> None:
    names, vectors = load_controllers(val_dirs)
    flights_all = test_flights()
    config = {
        "controllers": names, "settings": {k: dict(zip(NAMES, v)) for k, v in vectors.items()},
        "choices": {m: str(d / "choice.json") for m, d in val_dirs.items()},
        "choices_sha256": {m: hashlib.sha256((d / "choice.json").read_bytes()).hexdigest()
                           for m, d in val_dirs.items()},
        "sets": {TEST_SET: 300, HARDER_SET: set_size(HARDER_SET)}, "git": _git_hash(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    out.mkdir(parents=True, exist_ok=True)
    flights_path = out / "flights.csv"
    cfg_path = out / "config.json"
    flights: list[dict] = []
    if flights_path.exists():
        if not resume:
            sys.exit(f"{out} already has flights; the final test is flown once "
                     "(pass --resume only to finish an interrupted run)")
        saved = json.loads(cfg_path.read_text())
        diff = [k for k in ("controllers", "settings", "choices_sha256", "sets")
                if saved.get(k) != config[k]]
        if diff:
            sys.exit(f"{out} holds a different final test (differs in {', '.join(diff)})")
        flights = read_flights(flights_path)
        print(f"resuming {out}: {len(flights)} flights already done")
    else:
        cfg_path.write_text(json.dumps(config, indent=2))

    done = {(r["setting"], r["set"], r["wall"], r["gust"]) for r in flights}
    jobs = [(sid, x, s, i, g) for sid, x in vectors.items() for s, i, g in flights_all
            if (sid, s, i, g) not in done]
    if jobs:
        print(f"flying {len(jobs)} final test flights ({len(vectors)} settings x "
              f"{len(flights_all)} flights)", flush=True)
        t0 = time.perf_counter()
        new = not flights_path.exists()
        with ProcessPoolExecutor(workers, initializer=_quiet) as pool, \
                open(flights_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FLIGHT_FIELDS)
            if new:
                w.writeheader()
                fh.flush()
            futures = [pool.submit(fly_one, j) for j in jobs]
            for k, fut in enumerate(as_completed(futures), 1):
                w.writerow(fut.result())
                fh.flush()
                if k % 100 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} flights, {time.perf_counter() - t0:.0f} s", flush=True)
        flights = read_flights(flights_path)

    expected = len(vectors) * len(flights_all)
    if len(flights) != expected:
        sys.exit(f"{len(flights)} of {expected} final test flights recorded; run again with --resume")
    summary = summarize(names, flights)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    report(names, summary)
    print(f"\nresults in {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fly the val2 picks and HAND_TUNED once on test2 (300 flights) and harder "
                    "(50 flights), by the rule in docs/DECISIONS.md.")
    ap.add_argument("--results", default="results", help="folder holding the val2 runs")
    ap.add_argument("--out", help="output folder (default: <results>/final_test)")
    ap.add_argument("--workers", type=int, default=None, help="parallel flights (default: all cores)")
    ap.add_argument("--resume", action="store_true", help="finish an interrupted final test")
    a = ap.parse_args(argv)
    results = Path(a.results)
    val_dirs = {m: results / f"val2_{m}_s0-1-2" for m in ("nsga", "random")}
    missing = [str(d / "choice.json") for d in val_dirs.values() if not (d / "choice.json").exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    run(val_dirs, Path(a.out) if a.out else results / "final_test", a.workers, a.resume)
    return 0


if __name__ == "__main__":
    from wallsweep.final_test import main as _main

    sys.exit(_main())

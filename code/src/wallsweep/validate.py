from __future__ import annotations

import argparse
import csv
import functools
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

from wallsweep.conditions import VAL_SET, val_flights
from wallsweep.controller import HAND_TUNED, Params
from wallsweep.fly import fly
from wallsweep.nsga import _git_hash, _quiet

HAND = "hand_tuned"
NAMES = Params.names()
FINAL_SEEDS = (0, 1, 2)
PICKS = {"nsga": ("fast", "thorough", "balanced"), "random": ("pick",)}

FLIGHT_FIELDS = (
    "setting", *NAMES, "wall", "gust", "coverage", "raw_coverage", "time", "T_max", "time_ratio",
    "failure", "completed", "end_state", "min_standoff", "rms_standoff", "ledges", "run_s",
)
SUMMARY_FIELDS = (
    "setting", "seed", "cand", *NAMES, "train_coverage", "train_time_ratio", "train_fail",
    "val_flights", "val_fail", "val_coverage", "val_time_ratio", "val_time", "val_done", "status",
    "meets_threshold", "distance", "picks",
)
CONFIG_KEYS = ("method", "seeds", "provisional", "pareto_sha256", "set", "walls")


@functools.cache
def _val() -> tuple:
    return tuple(val_flights())


def val_conditions(index: int):
    return _val()[index]


def fly_one(job) -> dict:
    setting, x, index = job
    cond = val_conditions(index)
    base = {"setting": setting, **dict(zip(NAMES, map(float, x))), "wall": index,
            "gust": cond.gust_level}
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


def pareto_path(results: Path, method: str, seed: int) -> Path:
    return results / f"{method}_s{seed}" / "pareto.csv"


def load_settings(results: Path, method: str, seeds: list[int]) -> dict[str, dict]:
    settings = {HAND: {"x": [float(v) for v in HAND_TUNED.to_vector()], "seed": None,
                       "cand": None, "train_coverage": None, "train_time_ratio": None,
                       "train_fail": None}}
    for seed in seeds:
        with open(pareto_path(results, method, seed), newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            settings[f"s{seed}:{r['cand']}"] = {
                "x": [float(r[n]) for n in NAMES],
                "seed": seed,
                "cand": int(r["cand"]),
                "train_coverage": float(r["mean_coverage"]),
                "train_time_ratio": float(r["f_time_ratio"]),
                "train_fail": int(r["n_fail"]),
            }
    return settings


def _float(v: str) -> float:
    return math.nan if v in ("", "None") else float(v)


def parse_flight(r: dict) -> dict | None:
    try:
        return {
            **r, "wall": int(r["wall"]), "coverage": float(r["coverage"]),
            "time": _float(r["time"]), "time_ratio": float(r["time_ratio"]),
            "completed": r["completed"] == "True", "run_s": float(r["run_s"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def read_flights(path: Path, settings: dict, walls: list[int]) -> list[dict]:
    with open(path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    seen, keep, rows = set(), [], []
    for r in raw:
        p = parse_flight(r)
        if p is None or p["setting"] not in settings or p["wall"] not in walls:
            continue
        key = (p["setting"], p["wall"])
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
        rows.append(p)
    if len(keep) != len(raw):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FLIGHT_FIELDS)
            w.writeheader()
            w.writerows(keep)
    return rows


def summarize(settings: dict, flights: list[dict], n_walls: int) -> list[dict]:
    by = defaultdict(list)
    for r in flights:
        by[r["setting"]].append(r)
    out = []
    for sid, s in settings.items():
        rows = sorted(by[sid], key=lambda r: r["wall"])
        times = [r["time"] for r in rows if not math.isnan(r["time"])]
        out.append({
            "setting": sid, "seed": s["seed"], "cand": s["cand"], **dict(zip(NAMES, s["x"])),
            "train_coverage": s["train_coverage"], "train_time_ratio": s["train_time_ratio"],
            "train_fail": s["train_fail"], "val_flights": len(rows),
            "val_fail": sum(bool(r["failure"]) for r in rows),
            "val_coverage": float(np.mean([r["coverage"] for r in rows])) if rows else math.nan,
            "val_time_ratio": float(np.mean([r["time_ratio"] for r in rows])) if rows else math.nan,
            "val_time": float(np.mean(times)) if times else math.nan,
            "val_done": sum(r["completed"] for r in rows),
            "status": "", "meets_threshold": "", "distance": "", "picks": "",
        })
    missing = [s["setting"] for s in out if s["val_flights"] != n_walls]
    if missing:
        sys.exit(f"incomplete validation flights for {', '.join(missing)}; run again with --resume")
    return out


def cand_key(s: dict) -> tuple[int, int]:
    return s["seed"], s["cand"]


def select_pool(cands: list[dict]) -> tuple[str, list[dict]]:
    zero = [s for s in cands if s["val_fail"] == 0]
    if zero:
        return "zero_failures", zero
    fewest = min(s["val_fail"] for s in cands)
    return "fewest_failures", [s for s in cands if s["val_fail"] == fewest]


def pick_thorough(pool: list[dict]) -> dict:
    return min(pool, key=lambda s: (-s["val_coverage"], s["val_time_ratio"], cand_key(s)))


def pick_fast(pool: list[dict], threshold: float, time_key: str) -> tuple[dict, str]:
    met = [s for s in pool if s["val_coverage"] >= threshold]
    if not met:
        return pick_thorough(pool), "fallback_thorough"
    return min(met, key=lambda s: (s[time_key], -s["val_coverage"], cand_key(s))), "threshold_met"


def _scale(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def distances(pool: list[dict]) -> dict[str, float]:
    cov = _scale([s["val_coverage"] for s in pool])
    tr = _scale([s["val_time_ratio"] for s in pool])
    return {s["setting"]: math.hypot(1.0 - c, t) for s, c, t in zip(pool, cov, tr)}


def pick_balanced(pool: list[dict], dist: dict[str, float], taken: set[str]) -> dict | None:
    left = [s for s in pool if s["setting"] not in taken]
    if not left:
        return None
    return min(left, key=lambda s: (dist[s["setting"]], -s["val_coverage"], cand_key(s)))


def metrics(s: dict) -> dict:
    return {
        "setting": s["setting"], "seed": s["seed"], "cand": s["cand"],
        "params": {n: s[n] for n in NAMES},
        **{k: s[k] for k in ("val_fail", "val_coverage", "val_time_ratio", "val_time",
                             "val_done")},
    }


def choose(method: str, summary: list[dict]) -> dict:
    hand = next(s for s in summary if s["setting"] == HAND)
    threshold = hand["val_coverage"]
    cands = [s for s in summary if s["setting"] != HAND]
    if not cands:
        sys.exit("no Pareto settings to choose from")
    pool_branch, pool = select_pool(cands)
    fast, fast_branch = pick_fast(pool, threshold, "val_time_ratio")
    by_seconds, _ = pick_fast(pool, threshold, "val_time")
    fast_info = {**metrics(fast), "branch": fast_branch,
                 "seconds_choice": by_seconds["setting"],
                 "seconds_agrees": by_seconds["setting"] == fast["setting"]}
    picks: dict[str, dict | None] = {}
    dist: dict[str, float] = {}
    if method == "nsga":
        thorough = pick_thorough(pool)
        dist = distances(pool)
        balanced = pick_balanced(pool, dist, {fast["setting"], thorough["setting"]})
        picks["fast"] = fast_info
        picks["thorough"] = {**metrics(thorough), "branch": "highest_coverage"}
        picks["balanced"] = (None if balanced is None else
                             {**metrics(balanced), "branch": "lowest_distance",
                              "distance": dist[balanced["setting"]]})
    else:
        picks["pick"] = fast_info

    in_pool = {s["setting"] for s in pool}
    for s in summary:
        if s["setting"] == HAND:
            s["status"] = "reference"
        else:
            s["status"] = "pool" if s["setting"] in in_pool else "failures"
            s["meets_threshold"] = s["val_coverage"] >= threshold
            s["distance"] = dist.get(s["setting"], "")
        s["picks"] = "+".join(k for k, p in picks.items() if p and p["setting"] == s["setting"])

    return {
        "threshold": threshold,
        "threshold_source": "hand_tuned mean val2 coverage in this run",
        "n_candidates": len(cands),
        "pool": {"branch": pool_branch, "size": len(pool), "val_fail": pool[0]["val_fail"]},
        "picks": picks,
        "hand_tuned": {k: hand[k] for k in ("val_fail", "val_coverage", "val_time_ratio",
                                            "val_time", "val_done")},
    }


def _fmt(v, spec: str) -> str:
    return "-" if v in (None, "") or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def describe(p: dict) -> str:
    return (f"{p['setting']} ({p['val_fail']} failures, coverage {p['val_coverage']:.2%}, "
            f"time ratio {p['val_time_ratio']:.4f}, {_fmt(p['val_time'], '.1f')} s)")


def report(summary: list[dict], choice: dict, n_walls: int) -> None:
    order = sorted(summary, key=lambda s: (s["setting"] != HAND, s["val_time_ratio"]))
    print(f"{'setting':<12}{'speed':>7}{'vstep':>7}{'tr cov':>8}{'tr t/T':>8}{'fail':>6}"
          f"{'val cov':>9}{'val t/T':>9}{'val s':>8}{'done':>6}{'dist':>7}  status")
    for s in order:
        print(f"{s['setting']:<12}{s['sweep_speed']:>7.3f}{s['vertical_step']:>7.3f}"
              f"{_fmt(s['train_coverage'], '.3f'):>8}{_fmt(s['train_time_ratio'], '.3f'):>8}"
              f"{s['val_fail']:>3}/{n_walls:<2}{s['val_coverage']:>9.2%}{s['val_time_ratio']:>9.4f}"
              f"{_fmt(s['val_time'], '.1f'):>8}{s['val_done']:>6}{_fmt(s['distance'], '.3f'):>7}  "
              f"{s['status']}{'  <- ' + s['picks'] if s['picks'] else ''}")
    print()
    pool = choice["pool"]
    print(f"threshold: hand-tuned val2 coverage {choice['threshold']:.4%}")
    print(f"pool: {pool['size']}/{choice['n_candidates']} settings ({pool['branch']}, "
          f"{pool['val_fail']} failures each)")
    for name, p in choice["picks"].items():
        if p is None:
            print(f"{name}: none (no pool setting left after fast and thorough)")
            continue
        print(f"{name}: {describe(p)} [{p['branch']}]")
        if "seconds_agrees" in p and not p["seconds_agrees"]:
            print(f"  note: ranking by mean seconds instead of time ratio would pick "
                  f"{p['seconds_choice']}")
    if choice["provisional"]:
        print(f"note: provisional, seeds {choice['seeds']} are not all of {list(FINAL_SEEDS)}")
    print()
    hand = next(s for s in summary if s["setting"] == HAND)
    cands = [s for s in summary if s["setting"] != HAND]
    crashed = [s for s in cands if s["val_fail"] > 0]
    total = sum(s["val_fail"] for s in cands)
    print(f"train vs val failures: every Pareto setting had 0 failures on its 10 train flights; "
          f"on val2 {len(crashed)}/{len(cands)} settings failed at least once, "
          f"{total}/{len(cands) * n_walls} flights ({total / (len(cands) * n_walls):.1%}); "
          f"hand-tuned {hand['val_fail']}/{n_walls} ({hand['val_fail'] / n_walls:.1%})")
    for s in sorted(crashed, key=lambda s: -s["val_fail"]):
        print(f"  {s['setting']}: 0/10 train, {s['val_fail']}/{n_walls} val2 "
              f"({s['val_fail'] / n_walls:.1%}), sweep_speed {s['sweep_speed']:.3f}, "
              f"vertical_step {s['vertical_step']:.3f}")


def write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def is_provisional(seeds: list[int], walls: list[int] | None = None) -> bool:
    full = list(range(len(_val())))
    return sorted(set(seeds)) != list(FINAL_SEEDS) or (walls is not None and list(walls) != full)


def default_out(results: Path, method: str, seeds: list[int]) -> Path:
    tag = "val2_provisional" if is_provisional(seeds) else "val2"
    return results / f"{tag}_{method}_s{'-'.join(map(str, sorted(set(seeds))))}"


def run(method: str, seeds: list[int], out: Path, workers: int | None, resume: bool,
        results: Path = Path("results"), walls: list[int] | None = None) -> dict:
    seeds = sorted(set(seeds))
    paretos = {seed: pareto_path(results, method, seed) for seed in seeds}
    missing = [str(p) for p in paretos.values() if not p.exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    walls = list(range(len(_val()))) if walls is None else list(walls)
    provisional = is_provisional(seeds, walls)
    if provisional != ("provisional" in out.name):
        sys.exit(f"{out}: a {'provisional' if provisional else 'final'} run must write to a folder "
                 f"whose name {'contains' if provisional else 'does not contain'} 'provisional'")
    settings = load_settings(results, method, seeds)
    config = {
        "method": method, "seeds": seeds, "provisional": provisional,
        "sources": {str(k): str(p) for k, p in paretos.items()},
        "pareto_sha256": {str(k): hashlib.sha256(p.read_bytes()).hexdigest()
                          for k, p in paretos.items()},
        "set": VAL_SET, "walls": [[i, val_conditions(i).gust_level] for i in walls],
        "settings": list(settings), "git": _git_hash(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    out.mkdir(parents=True, exist_ok=True)
    flights_path = out / "flights.csv"
    cfg_path = out / "config.json"
    flights: list[dict] = []
    if flights_path.exists():
        if not resume:
            sys.exit(f"{out} already has flights; pass --resume to continue it")
        saved = json.loads(cfg_path.read_text())
        diff = [k for k in CONFIG_KEYS if saved.get(k) != config[k]]
        if diff:
            sys.exit(f"{out} holds a different validation run (differs in {', '.join(diff)})")
        flights = read_flights(flights_path, settings, walls)
        print(f"resuming {out}: {len(flights)} flights already done")
    else:
        cfg_path.write_text(json.dumps(config, indent=2))

    done = {(r["setting"], r["wall"]) for r in flights}
    jobs = [(sid, s["x"], i) for sid, s in settings.items() for i in walls
            if (sid, i) not in done]
    if jobs:
        print(f"flying {len(jobs)} {VAL_SET} flights ({len(settings)} settings x {len(walls)} walls)",
              flush=True)
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
                row = fut.result()
                w.writerow(row)
                fh.flush()
                if k % 50 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} flights, {time.perf_counter() - t0:.0f} s", flush=True)
        flights = read_flights(flights_path, settings, walls)

    summary = summarize(settings, flights, len(walls))
    choice = {"method": method, "seeds": seeds, "provisional": provisional,
              **choose(method, summary)}
    write_csv(out / "summary.csv", SUMMARY_FIELDS, summary)
    (out / "choice.json").write_text(json.dumps(choice, indent=2))
    report(summary, choice, len(walls))
    print(f"\nresults in {out}")
    return choice


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fly the pooled Pareto settings of one method and HAND_TUNED on the 50 val2 "
                    "walls and make the picks by the rule in docs/DECISIONS.md.")
    ap.add_argument("--method", choices=list(PICKS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS),
                    help="search seeds whose Pareto fronts are pooled (default: 0 1 2)")
    ap.add_argument("--workers", type=int, default=None, help="parallel flights (default: all cores)")
    ap.add_argument("--resume", action="store_true", help="continue a folder that already has flights")
    ap.add_argument("--results", default="results", help="folder holding the search runs")
    ap.add_argument("--out", help="output folder (default: <results>/val2_<method>_s0-1-2, or "
                                  "<results>/val2_provisional_<method>_s<seeds> for fewer seeds)")
    a = ap.parse_args(argv)
    results = Path(a.results)
    seeds = sorted(set(a.seeds))
    out = Path(a.out) if a.out else default_out(results, a.method, seeds)
    run(a.method, seeds, out, a.workers, a.resume, results)
    return 0


if __name__ == "__main__":
    from wallsweep.validate import main as _main

    sys.exit(_main())

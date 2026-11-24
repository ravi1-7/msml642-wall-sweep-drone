"""NSGA-II and random search over the 9 controller settings, on the same 10 training flights.

Run from the code/ folder:
    uv run python -m wallsweep.nsga --algo nsga --seed 0
    uv run python -m wallsweep.nsga --algo random --seed 0
    uv run python -m wallsweep.nsga --algo nsga --pop 6 --gens 2 --out results/smoke

Objectives, both minimized: minus the mean coverage, and the mean of time / T_max.
Constraint: the number of failed flights must be 0.

Each run folder holds config.json, evaluations.csv (one row per flight), candidates.csv
(one row per candidate), generations.csv (one row per generation or batch), pareto.csv
(non-dominated candidates with no failures, over every evaluation so far) and checkpoint.pkl.
Running the same command again resumes from the last finished generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import pybullet

from wallsweep.conditions import TRAINING_WALLS, conditions
from wallsweep.controller import BOUNDS, Params
from wallsweep.fly import fly

EXTRA_WALLS = ((5, "strong"), (6, "strong"), (7, "strong"), (8, "moderate"), (9, "none"))
SEARCH_WALLS = TRAINING_WALLS + EXTRA_WALLS
HV_REF = (0.0, 1.0)
NAMES = Params.names()

FLIGHT_FIELDS = (
    "gen", "cand", *NAMES, "wall", "gust", "coverage", "raw_coverage", "time", "elapsed", "T_max",
    "time_ratio", "failure", "completed", "end_state", "min_standoff", "rms_standoff",
    "ledges", "run_s",
)
CAND_FIELDS = (
    "gen", "cand", *NAMES, "f_coverage", "f_time_ratio", "mean_coverage", "mean_time",
    "n_fail", "n_done", "constraint", "feasible",
)
GEN_FIELDS = (
    "gen", "flights", "candidates", "feasible_gen", "feasible_total", "front_size",
    "hypervolume", "best_coverage", "best_time_ratio", "mean_time_s", "gen_s", "s_per_flight",
)


@cache
def _wall(index: int, gust: str):
    return conditions("train", index, gust)


def _quiet():
    warnings.filterwarnings("ignore", module="gymnasium")
    warnings.filterwarnings("ignore", message=".*precision lowered.*")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)


def fly_one(job) -> dict:
    x, index, gust = job
    t0 = time.perf_counter()
    try:
        f = fly(Params.from_vector(x), _wall(index, gust))
    except (pybullet.error, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        return {
            "wall": index, "gust": gust, "coverage": 0.0, "raw_coverage": 0.0, "time": np.nan,
            "elapsed": np.nan, "T_max": np.nan, "time_ratio": 1.0,
            "failure": f"error:{type(e).__name__}:{e}", "completed": False, "end_state": "error",
            "min_standoff": None, "rms_standoff": None, "ledges": 0,
            "run_s": time.perf_counter() - t0,
        }
    s = f.score
    return {
        "wall": index, "gust": gust, "coverage": s.coverage, "raw_coverage": s.raw_coverage,
        "time": s.time, "elapsed": s.elapsed, "T_max": s.T_max, "time_ratio": s.time / s.T_max,
        "failure": s.failure or "", "completed": f.completed, "end_state": f.end_state,
        "min_standoff": f.min_standoff, "rms_standoff": f.rms_standoff_error,
        "ledges": f.ledges_logged, "run_s": time.perf_counter() - t0,
    }


def evaluate(X: np.ndarray, pool, walls=SEARCH_WALLS):
    jobs = [(x, i, g) for x in X for i, g in walls]
    flights = list(pool.map(fly_one, jobs, chunksize=1))
    n = len(walls)
    F = np.zeros((len(X), 2))
    G = np.zeros((len(X), 1))
    summaries = []
    for k in range(len(X)):
        rows = flights[k * n:(k + 1) * n]
        cov = float(np.mean([r["coverage"] for r in rows]))
        ratio = float(np.mean([r["time_ratio"] for r in rows]))
        n_fail = sum(bool(r["failure"]) for r in rows)
        F[k] = (-cov, ratio)
        G[k, 0] = n_fail
        summaries.append({
            "f_coverage": -cov, "f_time_ratio": ratio, "mean_coverage": cov,
            "mean_time": float(np.nanmean([r["time"] for r in rows])),
            "n_fail": n_fail, "n_done": sum(bool(r["completed"]) for r in rows),
            "constraint": float(n_fail), "feasible": n_fail == 0,
        })
    return F, G, summaries, flights


def pareto_mask(F: np.ndarray) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    mask = np.zeros(len(F), dtype=bool)
    if len(F):
        mask[NonDominatedSorting().do(F, only_non_dominated_front=True)] = True
    return mask


def hypervolume(F: np.ndarray) -> float:
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=np.array(HV_REF))(F)) if len(F) else 0.0


@dataclass
class State:
    next_gen: int = 0
    next_cand: int = 0
    flights: int = 0
    archive: list[dict] = field(default_factory=list)
    algorithm: object = None
    random_X: np.ndarray | None = None


class Run:
    def __init__(self, out: Path, config: dict):
        self.out = out
        self.config = config
        self.ckpt = out / "checkpoint.pkl"
        self.files = {
            "evaluations.csv": FLIGHT_FIELDS,
            "candidates.csv": CAND_FIELDS,
            "generations.csv": GEN_FIELDS,
        }

    def open(self) -> State:
        self.out.mkdir(parents=True, exist_ok=True)
        cfg_path = self.out / "config.json"
        if self.ckpt.exists():
            saved = json.loads(cfg_path.read_text())
            keys = ("algo", "seed", "pop", "gens", "n", "walls")
            diff = [k for k in keys if saved.get(k) != self.config.get(k)]
            if diff:
                sys.exit(f"{self.out} holds a different run (differs in {', '.join(diff)})")
            with open(self.ckpt, "rb") as fh:
                state = pickle.load(fh)
            for name in self.files:
                self._truncate(self.out / name, state.next_gen)
            print(f"resuming {self.out} at generation {state.next_gen}")
            return state
        if any((self.out / name).exists() for name in self.files):
            sys.exit(f"{self.out} has results but no checkpoint; use a new --out folder")
        cfg_path.write_text(json.dumps(self.config, indent=2))
        return State()

    def _truncate(self, path: Path, next_gen: int) -> None:
        if not path.exists():
            return
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        keep = [r for r in rows if int(r["gen"]) < next_gen]
        if len(keep) != len(rows):
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.files[path.name])
                w.writeheader()
                w.writerows(keep)

    def _append(self, name: str, rows: list[dict]) -> None:
        path = self.out / name
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.files[name])
            if new:
                w.writeheader()
            w.writerows(rows)

    def record(self, state: State, X, summaries, flights, gen_s: float) -> None:
        gen = state.next_gen
        n = len(self.config["walls"])
        cand_rows, flight_rows = [], []
        for k, x in enumerate(X):
            cid = state.next_cand + k
            params = dict(zip(NAMES, map(float, x)))
            cand_rows.append({"gen": gen, "cand": cid, **params, **summaries[k]})
            for r in flights[k * n:(k + 1) * n]:
                flight_rows.append({"gen": gen, "cand": cid, **params, **r})
        state.archive.extend(cand_rows)
        state.next_cand += len(X)
        state.flights += len(flight_rows)

        feas = [c for c in state.archive if c["feasible"]]
        Ff = np.array([[c["f_coverage"], c["f_time_ratio"]] for c in feas]).reshape(-1, 2)
        front = [c for c, m in zip(feas, pareto_mask(Ff)) if m]
        front.sort(key=lambda c: c["f_time_ratio"])
        gen_row = {
            "gen": gen, "flights": state.flights, "candidates": state.next_cand,
            "feasible_gen": int(sum(s["feasible"] for s in summaries)),
            "feasible_total": len(feas), "front_size": len(front),
            "hypervolume": hypervolume(Ff),
            "best_coverage": max((c["mean_coverage"] for c in feas), default=np.nan),
            "best_time_ratio": min((c["f_time_ratio"] for c in feas), default=np.nan),
            "mean_time_s": float(np.nanmean([s["mean_time"] for s in summaries])),
            "gen_s": gen_s, "s_per_flight": gen_s / len(flight_rows),
        }
        self._append("evaluations.csv", flight_rows)
        self._append("candidates.csv", cand_rows)
        self._append("generations.csv", [gen_row])
        with open(self.out / "pareto.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CAND_FIELDS)
            w.writeheader()
            w.writerows(front)
        state.next_gen = gen + 1
        tmp = self.ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
        os.replace(tmp, self.ckpt)
        print(f"gen {gen:>3}  feasible {gen_row['feasible_gen']:>2}/{len(X)}  "
              f"front {len(front):>3}  hv {gen_row['hypervolume']:.4f}  "
              f"best cov {gen_row['best_coverage']:.3f}  best t/T {gen_row['best_time_ratio']:.3f}  "
              f"{gen_s:.0f} s ({gen_row['s_per_flight']:.2f} s/flight)", flush=True)


def make_nsga(pop: int, seed: int):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    problem = Problem(n_var=len(NAMES), n_obj=2, n_ieq_constr=1,
                      xl=Params.lower(), xu=Params.upper())
    algorithm = NSGA2(pop_size=pop, sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                      eliminate_duplicates=True)
    algorithm.setup(problem, seed=seed, verbose=False)
    return algorithm


def run_nsga(run: Run, state: State, pool, pop: int, gens: int, seed: int) -> None:
    from pymoo.core.evaluator import Evaluator
    from pymoo.problems.static import StaticProblem

    if state.algorithm is None:
        state.algorithm = make_nsga(pop, seed)
    algorithm = state.algorithm
    while state.next_gen < gens:
        t0 = time.perf_counter()
        offspring = algorithm.ask()
        X = offspring.get("X")
        F, G, summaries, flights = evaluate(X, pool)
        Evaluator().eval(StaticProblem(algorithm.problem, F=F, G=G), offspring)
        algorithm.tell(infills=offspring)
        run.record(state, X, summaries, flights, time.perf_counter() - t0)


def run_random(run: Run, state: State, pool, batch: int, n: int, seed: int) -> None:
    if state.random_X is None:
        rng = np.random.default_rng(seed)
        lo, hi = Params.lower(), Params.upper()
        state.random_X = lo + rng.random((n, len(NAMES))) * (hi - lo)
    n_batches = -(-n // batch)
    while state.next_gen < n_batches:
        t0 = time.perf_counter()
        X = state.random_X[state.next_gen * batch:(state.next_gen + 1) * batch]
        _, _, summaries, flights = evaluate(X, pool)
        run.record(state, X, summaries, flights, time.perf_counter() - t0)


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    from pymoo.config import Config

    Config.warnings["not_compiled"] = False
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["nsga", "random"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=40,
                    help="NSGA-II population, or random search batch size")
    ap.add_argument("--gens", type=int, default=50, help="NSGA-II generations")
    ap.add_argument("--n", type=int, default=2000, help="random search candidates")
    ap.add_argument("--workers", type=int, default=None, help="parallel flights (default: all cores)")
    ap.add_argument("--out", help="run folder (default: results/<algo>_s<seed>)")
    a = ap.parse_args(argv)

    out = Path(a.out or f"results/{a.algo}_s{a.seed}")
    config = {
        "algo": a.algo, "seed": a.seed, "pop": a.pop,
        "gens": a.gens if a.algo == "nsga" else None,
        "n": a.n if a.algo == "random" else a.pop * a.gens,
        "walls": [list(w) for w in SEARCH_WALLS],
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "hv_ref": list(HV_REF),
        "git": _git_hash(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    run = Run(out, config)
    state = run.open()
    with ProcessPoolExecutor(a.workers, initializer=_quiet) as pool:
        if a.algo == "nsga":
            run_nsga(run, state, pool, a.pop, a.gens, a.seed)
        else:
            run_random(run, state, pool, a.pop, a.n, a.seed)
    print(f"done: {state.next_cand} candidates, {state.flights} flights, "
          f"{sum(c['feasible'] for c in state.archive)} with no failures; "
          f"results in {out}")
    return 0


if __name__ == "__main__":
    from wallsweep.nsga import main as _main

    sys.exit(_main())

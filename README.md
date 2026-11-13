# Sweep Controller for Wall Cleaning (MSML642, Group 10)

Ravichandra Parvatham and Akhil Kambhatla.

A Crazyflie 2.1 sweeps a building face while holding a 30 cm standoff from the wall, handling sills, ledges, wind gusts and the growing load of a cleaning hose fed from the ground. NSGA-II, a multi-objective genetic algorithm, tunes the 9 settings of the sweep controller to maximize wall coverage and minimize sweep time with no wall contact. Results are compared against hand tuning and random search on walls not seen during tuning, in simulation (gym-pybullet-drones).

## Quick start

Needs [uv](https://docs.astral.sh/uv/). Works on macOS (Apple Silicon) and Linux.

```bash
git clone git@github.com:ravi1-7/msml642-wall-sweep-drone.git
cd msml642-wall-sweep-drone/code
uv sync
uv run python scripts/smoke_test.py   # a simulated Crazyflie hovers
```

More in [code/README.md](code/README.md).

## Layout

| Folder | Contents |
|---|---|
| `code/` | Python project (uv): simulator wrapper, controller, NSGA-II, scripts. |
| `.github/workflows/` | Linux check: installs the code and runs a short simulated flight on Ubuntu on every push. |
| `archive/` | Earlier drafts, kept locally only (not in git). |


## Project Description

The goal is to create an autonomous drone that can clean windows of buildings. Our idea is to simulate a Crazyflie 2.1 nano drone that sweeps a wall in back-and-forth passes, holding about 30 cm from the surface (accepted band 25 to 35 cm) without touching it, while handling sills, ledges and wind gusts. A small, readable sweep controller decides where the drone moves. Its behavior depends on 9 numbers. We use NSGA-II, a multi-objective genetic algorithm, to find good values for those 9 numbers in simulation, scoring each candidate on wall coverage and sweep time, with wall contact forbidden. We then compare the evolved controllers against hand tuning and random search on walls they have never seen.

The question we answer is: can evolved settings clean unseen walls better than hand tuning, without more contacts?

We model contact-free spray cleaning, the way commercial cleaning drones work, i.e., the drone sprays from a fixed distance and never wipes. "Coverage" of the spray is synonymous to cleaning.

The drone's load changes during the flight. Like commercial cleaning drones (e.g. Lucid Bots Sherpa), our drone is fed cleaning liquid through a hose from a pump on the ground, so it never carries a tank or runs dry, and doesn't spend battery lifting liquid it hasn't sprayed yet. The cost is that the higher it climbs, the more hose hangs below it, so the load it lifts, and how it responds to the same commands, keeps changing with height. The controller must work from the bottom of the wall (little hose lifted) to the top (the most hose lifted).


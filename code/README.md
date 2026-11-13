# wallsweep

Python code for the wall-cleaning sweep controller.

## Setup

Needs [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync                    # simulation only (Mac or Linux)
```

uv installs Python 3.12 and the exact package versions in `uv.lock`. On a Mac the first install compiles PyBullet (under a minute).

## Check it works

```bash
uv run python scripts/smoke_test.py        # a 5 s simulated hover; prints how well it held 1 m
uv run python scripts/time_one_flight.py   # how long flights take, and the full NSGA-II run estimate
```

## Watch the simulated world

```bash
uv run python scripts/demo.py                          # random wall (seed 0), moderate gusts, hose
uv run python scripts/demo.py --seed 5 --harder --gust strong
uv run python scripts/demo.py --flat --gust none       # plain wall
uv run python scripts/demo.py --glass-dropout 0.4      # glass readings often lost
uv run python scripts/demo.py --seed 3 --snapshot results/wall.png   # save a picture, no window
```

The drone follows a fixed scripted path 30 cm from the wall (not the sweep controller yet). Orange lines are the 4 range sensors; the trail is green while the drone is inside the 25-35 cm band and red outside it. The label shows the raw front reading next to the Kalman filter's estimate (raw readings jump to 400 cm when glass is missed), the drone's mass (grows with the hose) and contacts. If you activated the venv (`source .venv/bin/activate`), drop the `uv run`.

## Linux (VM or cluster)

Same commands: `git clone`, then `uv sync` and `uv run ...` from `code/`. No admin rights, display or GPU needed. For a Slurm cluster, edit and submit `scripts/hpc/run_nsga.sbatch`.

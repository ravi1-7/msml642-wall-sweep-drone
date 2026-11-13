#!/bin/bash
# Slurm job template for a full NSGA-II run on a Linux cluster (e.g. UMD Zaratan).
# Only needed if laptop runs are too slow. Edit the account/partition lines first.
# Submit from the code/ folder:  mkdir -p results && sbatch scripts/hpc/run_nsga.sbatch
#SBATCH --job-name=wallsweep-nsga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
##SBATCH --account=YOUR_ALLOCATION
##SBATCH --partition=standard

set -euo pipefail
mkdir -p results

# uv installs into your home folder; no admin rights needed.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked
uv run python scripts/smoke_test.py

# Placeholder until nsga.py exists; for now, measure speed on this node.
uv run python scripts/time_one_flight.py --workers "${SLURM_CPUS_PER_TASK}"

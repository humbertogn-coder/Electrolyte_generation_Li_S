#!/bin/bash
# Tier 0 on Grace (HPRC, TAMU) as ONE serial job with block-level checkpointing.
# No job array -> no MaxArraySize / QOSMaxSubmit limits. If the job is cut,
# resubmit the same command: blocks whose CSV already exists are skipped.
#
# Usage:
#   sbatch workflows/slurm/tier0_xtb_serial.sbatch data/campaigns/AL-01/candidates_stage1.csv
#
# Cost: ~13 s per molecule with 4 threads (measured on a laptop) -> 53k molecules
# ~ 8 days serial. Split the range across a few jobs with START/STOP, e.g.
#   START=0     STOP=20000 sbatch ... ; START=20000 STOP=40000 sbatch ... ; ...
# Afterwards:  python -m pace_s.descriptors.merge_tier0 --candidates <csv> \
#                  --blocks data/campaigns/AL-01/tier0 --out data/campaigns/AL-01/candidates_tier0.csv
#
#SBATCH --job-name=paces_t0
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/user/%u/logs/paces_t0_%j.out
#SBATCH --error=/scratch/user/%u/logs/paces_t0_%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
CANDIDATES=${1:?candidate table csv}
CHUNK=${CHUNK:-50}
START=${START:-0}
STOP=${STOP:-}                      # empty = until the end of the table
CONFIG=${CONFIG:-workflows/configs/campaign_LiS.yaml}

module purge
module load GCC/13.2.0
module load Python/3.11.5
module load xtb/6.7.1
module load FlexiBLAS/3.3.1
source /scratch/user/$USER/pace_env/bin/activate   # venv with `pip install -e .` of this repo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_STACKSIZE=4G
ulimit -s unlimited
mkdir -p /scratch/user/$USER/logs

CAMPAIGN=$(basename "$(dirname "$CANDIDATES")")
OUT_DIR="data/campaigns/$CAMPAIGN/tier0"
WORK_DIR="/scratch/user/$USER/pace-s/$CAMPAIGN/tier0"
mkdir -p "$OUT_DIR" "$WORK_DIR"

# number of eligible rows (status == generated); the CLI indexes only those
N=$(python - "$CANDIDATES" <<'EOF'
import sys
from pace_s.generate.table import read_table
df = read_table(sys.argv[1])
print(int((df["status"] == "generated").sum()) if "status" in df else len(df))
EOF
)
STOP=${STOP:-$N}
echo "[$(date)] tier 0 serial: rows $START-$STOP of $N eligible, chunk $CHUNK"

for (( s=START; s<STOP; s+=CHUNK )); do
    out="$OUT_DIR/tier0_$(printf '%06d' "$s").csv"
    if [[ -s "$out" ]]; then
        echo "[$(date)] block $s exists, skipping"
        continue
    fi
    echo "[$(date)] block $s"
    python -m pace_s.descriptors.run_tier0 \
        --config "$CONFIG" --candidates "$CANDIDATES" \
        --start "$s" --n "$CHUNK" \
        --workdir "$WORK_DIR" --threads "$SLURM_CPUS_PER_TASK" \
        --out "$out.tmp" && mv "$out.tmp" "$out"
done
echo "[$(date)] done"

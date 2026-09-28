#!/bin/bash
# Tier 0 on Grace (HPRC, TAMU): xtb job array over the candidate table.
# Usage:  sbatch --array=0-1199%200 tier0_xtb_array.sbatch data/campaigns/AL-01/candidates_stage1.csv
# NOTE: Grace limits array size and submissions per QOS; if that bites, use tier0_xtb_serial.sbatch.
# Each task processes a block of CHUNK molecules (50 x 1200 tasks = 60k). With 4 threads,
# ~5-20 s per molecule measured on a laptop; a block of 50 fits in < 1 h.
# Output: one CSV per block in data/campaigns/<campaign>/tier0/; merged afterwards with pandas.
#SBATCH --job-name=paces_t0
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/user/%u/logs/paces_t0_%A_%a.out

set -euo pipefail
MANIFEST=${1:?candidate table csv}
CHUNK=${CHUNK:-50}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_STACKSIZE=4G
ulimit -s unlimited

module purge
module load GCC/13.2.0
module load Python/3.11.5
module load xtb/6.7.1
module load FlexiBLAS/3.3.1
source /scratch/user/$USER/pace_env/bin/activate   # venv with `pip install -e .` of this repo

CAMPAIGN=$(basename "$(dirname "$MANIFEST")")
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
mkdir -p "data/campaigns/$CAMPAIGN/tier0"
python -m pace_s.descriptors.run_tier0 \
    --config "workflows/configs/campaign_LiS.yaml" \
    --candidates "$MANIFEST" --start "$START" --n "$CHUNK" \
    --workdir "$SCRATCH/pace-s/$CAMPAIGN/tier0" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --out "data/campaigns/$CAMPAIGN/tier0/tier0_$(printf '%06d' "$START").csv"

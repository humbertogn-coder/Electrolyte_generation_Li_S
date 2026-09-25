#!/bin/bash
# Tier 0 en Grace (HPRC, TAMU): job array de xtb/CREST sobre un manifiesto de SMILES.
# Uso:  sbatch --array=0-999%200 tier0_xtb_array.sbatch data/campaigns/GEN-01/tier0_manifest.csv
# Cada tarea procesa un bloque de CHUNK moléculas para no saturar el scheduler.
#SBATCH --job-name=paces_t0
#SBATCH --account=CHANGE_ME            # cuenta de HPRC
#SBATCH --partition=cpu                # ajustar al nombre real de la cola CPU de Grace
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
MANIFEST=${1:?manifest csv}
CHUNK=${CHUNK:-50}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_STACKSIZE=4G
ulimit -s unlimited

module purge
# module load ...   # o: source activate pace-s
source "$HOME/.bashrc"
conda activate pace-s

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
python -m pace_s.descriptors.run_tier0 \
    --manifest "$MANIFEST" --start "$START" --n "$CHUNK" \
    --workdir "$SCRATCH/pace-s/$(basename "$(dirname "$MANIFEST")")/tier0" \
    --campaign-id "$(basename "$(dirname "$MANIFEST")")"

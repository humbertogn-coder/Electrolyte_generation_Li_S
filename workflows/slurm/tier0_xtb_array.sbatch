#!/bin/bash
# Tier 0 en Grace (HPRC, TAMU): job array de xtb sobre la tabla de candidatos.
# Uso:  sbatch --array=0-1199%200 tier0_xtb_array.sbatch data/campaigns/AL-01/candidates_stage1.csv
# Cada tarea procesa un bloque de CHUNK moléculas (50 x 1200 tareas = 60k). Con 4 hilos,
# ~5-20 s por molécula medido en laptop; un bloque de 50 cabe en < 1 h.
# Salida: un CSV por bloque en data/campaigns/<campaña>/tier0/; se unen después con pandas.
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
conda activate pace-s-tier0

CAMPAIGN=$(basename "$(dirname "$MANIFEST")")
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
mkdir -p "data/campaigns/$CAMPAIGN/tier0"
python -m pace_s.descriptors.run_tier0 \
    --config "workflows/configs/campaign_LiS.yaml" \
    --candidates "$MANIFEST" --start "$START" --n "$CHUNK" \
    --workdir "$SCRATCH/pace-s/$CAMPAIGN/tier0" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --out "data/campaigns/$CAMPAIGN/tier0/tier0_$(printf '%06d' "$START").csv"

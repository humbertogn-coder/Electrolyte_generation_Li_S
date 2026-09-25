#!/bin/bash
# Tier 1 en Grace: MD con MLIP (LAMMPS + plugin MACE/DeePMD) para una formulación.
# Uso:  sbatch tier1_mlip_md_gpu.sbatch PACES-0134 workflows/configs/campaign_LiS.yaml
# Presupuesto: 2-4 GPU-h por sistema (2000-4000 átomos, 2 ns).
#SBATCH --job-name=paces_t1
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu                # ajustar a la cola GPU de Grace
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
ELECTROLYTE_ID=${1:?electrolyte id}
CONFIG=${2:?campaign yaml}

module purge
# module load CUDA/12.1 ...           # según la compilación de LAMMPS con plugin del grupo
source "$HOME/.bashrc"
conda activate pace-s-mlip
export LAMMPS_BIN=${LAMMPS_BIN:-lmp}   # binario compilado con plugin de MACE o DeePMD

python -m pace_s.oracles.tier1_mlip_md \
    --electrolyte-id "$ELECTROLYTE_ID" --config "$CONFIG" \
    --workdir "$SCRATCH/pace-s/tier1/$ELECTROLYTE_ID" \
    --commit "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"

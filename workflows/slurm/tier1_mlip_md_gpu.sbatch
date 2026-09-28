#!/bin/bash
# Tier 1 on Grace: MLIP molecular dynamics (LAMMPS + MACE/DeePMD plugin) for one formulation.
# Usage:  sbatch tier1_mlip_md_gpu.sbatch PACES-0134 workflows/configs/campaign_LiS.yaml
# Budget: 2-4 GPU-h per system (2000-4000 atoms, 2 ns).
#SBATCH --job-name=paces_t1
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu                # adjust to Grace's GPU queue
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
# module load CUDA/12.1 ...           # according to the group's LAMMPS+plugin build
source "$HOME/.bashrc"
conda activate pace-s-mlip
export LAMMPS_BIN=${LAMMPS_BIN:-lmp}   # binary compiled with the MACE or DeePMD plugin

python -m pace_s.oracles.tier1_mlip_md \
    --electrolyte-id "$ELECTROLYTE_ID" --config "$CONFIG" \
    --workdir "$SCRATCH/pace-s/tier1/$ELECTROLYTE_ID" \
    --commit "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)"

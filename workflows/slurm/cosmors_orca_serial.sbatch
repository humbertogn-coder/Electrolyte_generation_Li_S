#!/bin/bash
# openCOSMO-RS surfaces (ORCA 6.1, BP86/def2-TZVP(D)/CPCM) on Grace as ONE serial
# job over a structures file, molecule by molecule, resumable. The vendored
# pipeline (`third_party/openCOSMO-RS_conformer_pipeline`) skips finished steps of
# a molecule when rerun in the same directory (`--continue_calculation`), so if
# the job is cut, resubmit the same command.
#
# Usage (from the repo root, after `prepare`):
#   python -m pace_s.oracles.cosmors prepare --set gate --out data/campaigns/cosmors/gate
#   sbatch workflows/slurm/cosmors_orca_serial.sbatch data/campaigns/cosmors/gate/structures.inp
#
# Outputs: /scratch/user/$USER/pace-s/cosmors/<set>/runs/<name>/COSMO_TZVPD/<name>_c000.orcacosmo
# and a copy of every finished .orcacosmo under data/campaigns/cosmors/<set>/runs/ in the repo
# (small text files, ~100 kB each). Then:
#   python -m pace_s.oracles.cosmors evaluate --root data/campaigns/cosmors/gate/runs \
#       --structures data/campaigns/cosmors/gate/structures.csv --out data/campaigns/cosmors/gate/gate_h3_cosmors.csv
#
# Requirements on Grace (once):
#   module load GCC/13.2.0 OpenMPI/4.1.8 ORCA/6.1.1-avx2   (from `module spider ORCA/6.1.1-avx2`)
#   pip install plotly "opencosmorspy @ git+https://github.com/TUHH-TVT/openCOSMO-RS_py"
#
# Cost (rough, 8 cores): 10-40 min per solvent molecule (5-20 heavy atoms), longer
# for Li2S8/S8. The 21-molecule gate set fits in a day; the 81-molecule known set
# in a few days. Split with START/STOP (line numbers of the structures file, 0-based).
#
#SBATCH --job-name=paces_cosmors
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/%u/logs/paces_cosmors_%j.out
#SBATCH --error=/scratch/user/%u/logs/paces_cosmors_%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
STRUCTURES=${1:?structures.inp written by `python -m pace_s.oracles.cosmors prepare`}
START=${START:-0}
STOP=${STOP:-}
NCORES=${NCORES:-${SLURM_CPUS_PER_TASK:-1}}   # set NCORES=1 if ORCA's MPI does not start

REPO=$(pwd)
SET=$(basename "$(dirname "$STRUCTURES")")
WORK="/scratch/user/$USER/pace-s/cosmors/$SET/runs"
OUT="$REPO/data/campaigns/cosmors/$SET/runs"
mkdir -p "$WORK" "$OUT" /scratch/user/$USER/logs

module purge
module load GCC/13.2.0
module load Python/3.11.5
module load xtb/6.7.1
module load FlexiBLAS/3.3.1
module load OpenMPI/4.1.8            # required by ORCA/6.1.1-avx2 (module spider)
module load ORCA/6.1.1-avx2
source /scratch/user/$USER/pace_env/bin/activate

# ORCA looks for helper programs (otool_*) next to its own executable, and the
# module directory is read-only. A private bin with symlinks to every ORCA file
# plus `otool_xtb -> xtb` solves it; the pipeline finds both on PATH.
ORCA_DIR=$(dirname "$(which orca)")
BIN="/scratch/user/$USER/orca_bin"
mkdir -p "$BIN"
for f in "$ORCA_DIR"/*; do ln -sf "$f" "$BIN/$(basename "$f")"; done
ln -sf "$(which xtb)" "$BIN/otool_xtb"
export PATH="$BIN:$PATH"
export OMP_NUM_THREADS=1
ulimit -s unlimited

orca --version | grep -m1 "Program Version" || true
echo "[$(date)] set=$SET structures=$STRUCTURES lines $START-${STOP:-end} ncores=$NCORES"

N=$(grep -c . "$STRUCTURES")
STOP=${STOP:-$N}
cp "$REPO/third_party/openCOSMO-RS_conformer_pipeline/cpcm_radii.inp" "$WORK/cpcm_radii.inp"
cd "$WORK"

i=-1
while IFS= read -r line; do
  i=$((i + 1))
  [ "$i" -lt "$START" ] && continue
  [ "$i" -ge "$STOP" ] && break
  [ -z "$line" ] && continue
  name=$(printf '%s' "$line" | cut -f1)
  if [ -f "$name/COSMO_TZVPD/${name}_c000.orcacosmo" ]; then
    echo "[$(date)] $i $name: done, skipping"
  else
    echo "[$(date)] $i $name: start"
    printf '%s\n' "$line" > "one_$name.inp"
    PYTHONPATH="$REPO/third_party/openCOSMO-RS_conformer_pipeline" python "$REPO/third_party/openCOSMO-RS_conformer_pipeline/ConformerGenerator.py" \
        --structures_file "one_$name.inp" --cpcm_radii_file "cpcm_radii.inp" --n_cores "$NCORES" \
        > "$name.log" 2>&1 || echo "[$(date)] $i $name: FAILED (see $WORK/$name.log)"
    rm -f "one_$name.inp"
  fi
  if [ -f "$name/COSMO_TZVPD/${name}_c000.orcacosmo" ]; then
    mkdir -p "$OUT/$name/COSMO_TZVPD"
    cp "$name/COSMO_TZVPD/${name}_c000.orcacosmo" "$OUT/$name/COSMO_TZVPD/"
  fi
done < "$STRUCTURES"

echo "[$(date)] finished: $(ls -d "$OUT"/*/COSMO_TZVPD 2>/dev/null | wc -l) molecules with .orcacosmo in $OUT"

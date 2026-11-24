#!/bin/bash
# CREST conformer search (GFN2-xTB, ALPB ether, NCI mode) of the Li+ clusters whose
# xtb pose sampling did not converge. One array task per cluster folder.
#
# Steps (from the repo root on Grace):
#   1. once:   bash scripts/grace/install_crest.sh
#   2. laptop: python -m pace_s.solvation.merge_runs <run dirs> --out <merged> --crest-export <merged>/crest_inputs
#              then copy the folder to Grace (data/campaigns is not in git), from PowerShell:
#              scp -r data\campaigns\<merged>\crest_inputs <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/
#   3. Grace:  mkdir -p /scratch/user/$USER/logs     # Slurm cannot start the job if this folder is missing
#              chmod -R u+w /scratch/user/$USER/crest_inputs   # scp from Windows may copy it read-only
#              N=$(ls -d /scratch/user/$USER/crest_inputs/*/ | wc -l)
#              sbatch --array=0-$((N-1)) workflows/slurm/crest_clusters.sbatch /scratch/user/$USER/crest_inputs
#   4. copy the folder back over the laptop one (scp -r <netid>@grace...:/scratch/user/<netid>/crest_inputs
#              data\campaigns\<merged>\) and on the laptop (xtb there gives the same level):
#              python -m pace_s.solvation.crest_import data/campaigns/<merged>/crest_inputs --out data/campaigns/<crest run>
#              python -m pace_s.solvation.merge_runs <run dirs> data/campaigns/<crest run> --out <merged again>
#
# CREST runs inside each cluster folder and leaves crest_conformers.xyz, crest_best.xyz and
# crest.out there (scratch files are deleted at the end). A folder that already has
# crest_best.xyz is skipped, so resubmitting the same command only redoes the failed ones.
#
# MODE: "" (default, full iMTD-GC search) or "--quick". Measured on Grace (8 cores, full
# mode, 2026-11-23): Li+/FSI-/DME 26 atoms 5 min, Li+/FSI-/TTE 28 atoms 13 min,
# Li(TTE)2+ 37 atoms 14 min, Li(DEE)2+ 45 atoms 22 min (about 0.7-3 SU each).
#
#SBATCH --job-name=paces_crest
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/user/%u/logs/paces_crest_%A_%a.out

set -euo pipefail
INPUTS=${1:?folder written by merge_runs --crest-export}
MODE=${MODE:-}
CREST=${CREST:-$(command -v crest || echo /scratch/user/$USER/bin/crest)}
[ -x "$CREST" ] || { echo "CREST not found ($CREST); run scripts/grace/install_crest.sh"; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_STACKSIZE=4G
ulimit -s unlimited || true

mapfile -t DIRS < <(find "$INPUTS" -mindepth 1 -maxdepth 1 -type d | sort)
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt "${#DIRS[@]}" ] || { echo "task $TASK: only ${#DIRS[@]} cluster folders"; exit 0; }
D=${DIRS[$TASK]}
cd "$D"
if [ -s crest_best.xyz ]; then echo "$D: already done"; exit 0; fi
# scp from Windows can deliver read-only folders; CREST must write here
if [ ! -w . ]; then
    echo "$D is not writable. Run once on Grace:  chmod -R u+w $INPUTS   and resubmit."
    exit 1
fi
# leave a trace in the cluster folder itself, so a failed task is visible after copying it back
STATUS=crest.status
echo "started $(date) job ${SLURM_ARRAY_JOB_ID:-?}_${TASK} on $(hostname) crest=$CREST" > "$STATUS"
trap 'echo "FAILED (exit $?) $(date); see /scratch/user/$USER/logs/paces_crest_${SLURM_ARRAY_JOB_ID:-?}_${TASK}.out" >> "$STATUS"' ERR
CHRG=$(cat .CHRG)
UHF=$(cat .UHF)
# no python needed: read "alpb": "ether" (or null) from meta.json
ALPB=$(grep -o '"alpb": *"[^"]*"' meta.json | sed 's/.*"\([^"]*\)"$/\1/' || true)
SOLV=""
[ -n "$ALPB" ] && SOLV="--alpb $ALPB"
echo "$(date) $D chrg=$CHRG uhf=$UHF $SOLV $MODE threads=$OMP_NUM_THREADS"  # --nci added below for Li+ clusters
# NCI mode (wall potential that keeps the complex together) only for Li+ clusters
NCI=""
grep -qiE '^[[:space:]]*li[[:space:]]' input.xyz && NCI="--nci"
"$CREST" input.xyz --gfn2 $SOLV --chrg "$CHRG" --uhf "$UHF" $NCI $MODE -T "$OMP_NUM_THREADS" > crest.out 2>&1
grep -q "CREST terminated normally" crest.out || { echo "$D: CREST did not terminate normally"; tail -20 crest.out;
    echo "FAILED: CREST did not terminate normally $(date)" >> "$STATUS"; exit 1; }
# keep only what crest_import and a reader need
find . -maxdepth 1 -type f ! -name 'input.xyz' ! -name 'meta.json' ! -name '.CHRG' ! -name '.UHF' \
     ! -name 'crest.out' ! -name 'crest_best.xyz' ! -name 'crest_conformers.xyz' ! -name 'crest.energies' \
     ! -name 'crest.status' -delete
find . -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo "done $(date)" >> "$STATUS"
echo "$(date) $D done: $(head -2 crest_best.xyz | tail -1)"

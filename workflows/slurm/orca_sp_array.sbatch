#!/bin/bash
# ORCA single points (wB97X-D3/def2-TZVPD, RIJCOSX, CPCM ether) on the xtb minima of the
# solvation clusters. One array task per folder written by `dft_sp export`.
#
# Steps:
#   1. laptop (Anaconda Prompt, repo root):
#        set PYTHONPATH=%CD%\src
#        python -m pace_s.solvation.dft_sp export <run dirs> --out data\campaigns\<name>\dft
#        scp -r data\campaigns\<name>\dft <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/<name>_dft
#   2. Grace (repo root):
#        chmod -R u+w /scratch/user/$USER/<name>_dft      # scp from Windows may copy it read-only
#        mkdir -p /scratch/user/$USER/logs                 # Slurm cannot start without it
#        N=$(ls -d /scratch/user/$USER/<name>_dft/*/ | wc -l)
#        sbatch --array=0-$((N-1)) workflows/slurm/orca_sp_array.sbatch /scratch/user/$USER/<name>_dft
#        cat /scratch/user/$USER/<name>_dft/*/orca.status
#   3. laptop: copy the folder back over the local one with scp, then
#        python -m pace_s.solvation.dft_sp import data\campaigns\<name>\dft
#
# Each folder keeps input.inp, run.inp (input + %pal), orca.out, orca.status, meta.json and
# geometry.xyz; ORCA scratch files are deleted. A folder whose orca.out already ends normally
# is skipped, so resubmitting the same command only redoes the failed ones.
#
# Cost (orca_check on Grace, 4 MPI tasks): DME 92 s, Li+-DME 76 s. The 45-atom clusters
# should take minutes to well under an hour with 8 tasks.
#
# ORCA parallelizes with MPI processes: --ntasks (not --cpus-per-task), full path, no mpirun.
#SBATCH --job-name=paces_orca_sp
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/%u/logs/paces_orca_sp_%A_%a.out

# modules first: Lmod's shell functions are not safe under `set -u`
module purge
module load GCC/13.2.0 OpenMPI/4.1.8 ORCA/6.1.1-avx2
ORCA=$(which orca)

set -euo pipefail
INPUTS=${1:?folder written by dft_sp export}
NPROCS=${SLURM_NTASKS:-8}

mapfile -t DIRS < <(find "$INPUTS" -mindepth 1 -maxdepth 1 -type d | sort)
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt "${#DIRS[@]}" ] || { echo "task $TASK: only ${#DIRS[@]} folders"; exit 0; }
D=${DIRS[$TASK]}
cd "$D"
if [ -f orca.out ] && grep -q "ORCA TERMINATED NORMALLY" orca.out; then echo "$D: already done"; exit 0; fi
if [ ! -w . ]; then
    echo "$D is not writable. Run once on Grace:  chmod -R u+w $INPUTS   and resubmit."
    exit 1
fi
STATUS=orca.status
echo "started $(date) job ${SLURM_ARRAY_JOB_ID:-?}_${TASK} on $(hostname) nprocs=$NPROCS orca=$ORCA" > "$STATUS"
trap 'echo "FAILED (exit $?) $(date); see /scratch/user/$USER/logs/paces_orca_sp_${SLURM_ARRAY_JOB_ID:-?}_${TASK}.out" >> "$STATUS"' ERR

sed "1a %pal nprocs $NPROCS end" input.inp > run.inp
echo "$(date) $D: $ORCA run.inp with $NPROCS MPI processes"
"$ORCA" run.inp > orca.out 2>&1 || true
if ! grep -q "ORCA TERMINATED NORMALLY" orca.out; then
    echo "$D: ORCA did not terminate normally"; tail -30 orca.out
    echo "FAILED: ORCA did not terminate normally $(date)" >> "$STATUS"
    exit 1
fi
find . -maxdepth 1 -type f ! -name 'input.inp' ! -name 'run.inp' ! -name 'orca.out' ! -name 'orca.status' \
     ! -name 'meta.json' ! -name 'geometry.xyz' -delete
echo "done $(date) $(grep 'FINAL SINGLE POINT ENERGY' orca.out | tail -1)" >> "$STATUS"
echo "$(date) $D done: $(grep 'FINAL SINGLE POINT ENERGY' orca.out | tail -1)"

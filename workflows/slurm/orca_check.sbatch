#!/bin/bash
# Smoke test of ORCA on Grace with the level planned for the DFT step of PACE-S 2.0:
# wB97X-D3/def2-TZVPD, RIJCOSX, CPCM (ether-like epsilon), 4 MPI processes.
# Computes DME, Li+ and an UNOPTIMIZED Li+-DME geometry and prints a binding energy.
# It checks access, method keywords, charges, implicit solvation and parallel start;
# the energy is not a result.
#
# Usage on Grace:  mkdir -p /scratch/user/$USER/orca_test && cd $_
#                  sbatch /path/to/pace-s/workflows/slurm/orca_check.sbatch
#
# Note: ORCA runs with MPI processes, so the job asks for --ntasks (not --cpus-per-task)
# and ORCA is called with its full path (never through mpirun).
#SBATCH --job-name=orca_check
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=orca_check_%j.out

module purge
module load GCC/13.2.0 OpenMPI/4.1.8 ORCA/6.1.1-avx2
ORCA=$(which orca)
cd "$SLURM_SUBMIT_DIR"
echo "ORCA: $ORCA"

HDR="! wB97X-D3 def2-TZVPD def2/J RIJCOSX CPCM TightSCF
%pal nprocs ${SLURM_NTASKS:-4} end
%maxcore 3000
%cpcm epsilon 4.33 refrac 1.3526 end"

DME='C   2.55296  0.32010 -0.35774
O   1.37601 -0.46923 -0.45274
C   0.45693 -0.18280  0.60454
C  -0.79264 -1.05706  0.45826
O  -0.51939 -2.46137  0.50556
C  -1.67572 -3.27298  0.37023
H   2.30859  1.38359 -0.43928
H   3.21661  0.05040 -1.18361
H   3.07121  0.12370  0.58577
H   0.94144 -0.40818  1.56131
H   0.19914  0.88032  0.57976
H  -1.32221 -0.82021 -0.47147
H  -1.47301 -0.85011  1.29133
H  -1.38279 -4.31363  0.53500
H  -2.42938 -3.00387  1.11641
H  -2.09418 -3.19438 -0.63625'

printf '%s\n* xyz 0 1\n%s\n*\n' "$HDR" "$DME" > dme.inp
printf '%s\n* xyz 1 1\nLi 0.0 0.0 0.0\n*\n' "$HDR" > li.inp
printf '%s\n* xyz 1 1\nLi  1.13844 -2.47228 -0.57511\n%s\n*\n' "$HDR" "$DME" > lidme.inp

for f in dme li lidme; do
  start=$(date +%s)
  $ORCA $f.inp > $f.out 2>&1
  ok=$(grep -c "ORCA TERMINATED NORMALLY" $f.out)
  echo "$f: terminated_normally=$ok  time=$(( $(date +%s) - start )) s  E=$(grep 'FINAL SINGLE POINT ENERGY' $f.out | tail -1 | awk '{print $5}')"
done

Ed=$(grep 'FINAL SINGLE POINT ENERGY' dme.out   | tail -1 | awk '{print $5}')
El=$(grep 'FINAL SINGLE POINT ENERGY' li.out    | tail -1 | awk '{print $5}')
Ec=$(grep 'FINAL SINGLE POINT ENERGY' lidme.out | tail -1 | awk '{print $5}')
awk -v c="$Ec" -v d="$Ed" -v l="$El" 'BEGIN{printf "E_bind(Li+ - DME) = %.3f eV  (unoptimized test geometry)\n", (c-d-l)*27.211386}'

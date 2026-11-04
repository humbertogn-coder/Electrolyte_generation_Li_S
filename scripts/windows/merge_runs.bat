@echo off
rem Merge exchange runs and judge sampling convergence (Anaconda Prompt, from anywhere).
rem Usage:  scripts\windows\merge_runs.bat RUN_DIR [RUN_DIR ...] --out OUT_DIR [--crest-export OUT_DIR\crest_inputs]
rem Example (laptop 24 poses + cloud 24 poses):
rem   scripts\windows\merge_runs.bat data\campaigns\SPV2-DEMO-P24 data\campaigns\SPV2-DEMO-P24-CLOUD ^
rem       --out data\campaigns\SPV2-DEMO-MERGED --crest-export data\campaigns\SPV2-DEMO-MERGED\crest_inputs
cd /d "%~dp0..\.."
call conda activate pace-s
if errorlevel 1 (
  echo Could not activate the conda environment pace-s.
  exit /b 1
)
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python -m pace_s.solvation.merge_runs %*

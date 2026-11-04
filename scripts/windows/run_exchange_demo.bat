@echo off
REM PACE-S 2.0 - first Tier 0.5 demonstration on a Windows laptop (Anaconda).
REM Exchange and binding energies of DME, DEE and TTE with GFN2-xTB/ALPB(ether).
REM
REM Run it from Anaconda Prompt (not by double-click), from any folder:
REM     C:\Users\humbe\OneDrive\Documents\Claude\PACE-S\pace-s\scripts\windows\run_exchange_demo.bat
REM Extra arguments go to the program and override the defaults, e.g.:
REM     run_exchange_demo.bat --poses 8 --threads 6
REM     run_exchange_demo.bat --solvents DME THP F5DEE
REM Running it again skips the poses that already finished.

setlocal
cd /d "%~dp0..\.."
echo [0/3] Activating the conda environment pace-s
call conda activate pace-s
if errorlevel 1 (
    echo Could not activate pace-s. Open "Anaconda Prompt" and run this script from there.
    exit /b 1
)
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
where xtb >nul 2>nul
if errorlevel 1 (
    echo xtb is not installed in pace-s. Install it with:  conda install -c conda-forge xtb
    exit /b 1
)

echo [1/3] Quick tests of the Tier 0.5 code
python -m pytest -q -p no:cacheprovider tests\test_solvation_species.py tests\test_solvation_reactions.py tests\test_solvation_contracts.py tests\test_solvation_builder.py tests\test_solvation_runner.py tests\test_solvation_convergence.py
if errorlevel 1 (
    echo Tests failed: send the output above before running the calculations.
    exit /b 1
)

echo [2/3] Exchange energies of DME, DEE, TTE (about 15-30 minutes)
python -m pace_s.solvation.run_exchange --solvents DME DEE TTE --poses 6 --threads 4 --out data\campaigns\SPV2-DEMO %*
if errorlevel 1 (
    echo The calculation stopped with an error: send the output above.
    exit /b 1
)

echo [3/3] Done. Results: %CD%\data\campaigns\SPV2-DEMO\reactions.csv
endlocal

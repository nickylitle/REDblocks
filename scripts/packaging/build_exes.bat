@echo off
REM Builds both programs from this folder. Needs Python 3.10-3.12 from python.org on PATH. Run once per update.
REM   dist\Simblocks\Simblocks.exe  desktop node editor with the solver built in (Windows 10/11 WebView2 draws the window).
REM                                folder build: starts in about 2 s (a one-file exe unpacks itself on every launch)
REM   dist\simblocks_runner.exe     batch studies: put a *_job.yaml + its sim .py next to it, double-click,
REM                                send back the *_results.zip it writes
REM At the end it offers to install Simblocks for you: Start Menu (Windows search), optional desktop shortcut,
REM Open with for .yaml models, and Settings > Apps uninstall. No administrator rights needed.
REM Neither needs the internet or Claude to run.
setlocal
cd /d "%~dp0\.."
python -m venv .build_venv || goto :err
call .build_venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
if not exist simblocks\app\examples mkdir simblocks\app\examples
copy /y examples\learn\*.yaml simblocks\app\examples\ >nul
if not exist simblocks\app\examples\data mkdir simblocks\app\examples\data
copy /y examples\learn\data\*.csv simblocks\app\examples\data\ >nul
if not exist simblocks\app\examples\showcase mkdir simblocks\app\examples\showcase
copy /y examples\showcase\apollo11\*.yaml simblocks\app\examples\showcase\ >nul
copy /y examples\showcase\apollo11\apollo11.html simblocks\app\examples\showcase\ >nul
python -m pip install ".[app]" pyinstaller || goto :err
REM Help shows every block's Python source; the one-file exe has no .py files, so save the sources first
python -c "from simblocks.app.api import write_block_sources; print(write_block_sources('simblocks/app/block_sources.json'))" || goto :err
python -m pip install --no-deps --force-reinstall . || goto :err
set COMMON=--noconfirm --collect-all simblocks --collect-submodules numba --collect-data llvmlite --hidden-import scipy.stats._qmc --hidden-import yaml
pyinstaller %COMMON% --onedir --windowed --icon packaging\simblocks.ico --collect-all webview --name Simblocks packaging\simblocks_runner.py || goto :err
pyinstaller %COMMON% --onefile --icon packaging\simblocks.ico --name simblocks_runner packaging\simblocks_runner.py || goto :err
echo.
echo Built: %cd%\dist\Simblocks\Simblocks.exe
echo        %cd%\dist\simblocks_runner.exe
echo.
choice /C YN /M "Install Simblocks for this Windows user now (Start Menu search, uninstall entry)"
if errorlevel 2 goto :done
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\install.ps1 || goto :err
:done
pause
exit /b 0
:err
echo BUILD FAILED. Check that Python 3.10-3.12 is installed and on PATH.
pause
exit /b 1

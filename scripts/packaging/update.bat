@echo off
REM Update REDblocks to the newest version: git pull, then build and install it (build_exes.bat).
REM Needs this folder to be a git clone (git clone https://github.com/nickylitle/REDblocks.git), not a ZIP download.
cd /d "%~dp0\.."
git pull --ff-only || goto :err
call packaging\build_exes.bat
exit /b
:err
echo UPDATE FAILED: git pull did not work (the reason is above). Is this folder a git clone, and is Git installed?
pause
exit /b 1

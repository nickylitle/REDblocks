@echo off
REM Run REDblocks straight from this git folder: no exe build, no new Python environment each time.
REM   dev.bat           git pull, then start the editor (picks up the new code at once)
REM   dev.bat nopull    start the editor on the code you have
REM The first run makes .dev_venv here and installs what REDblocks needs (a few minutes, once). After that it only
REM reinstalls when pyproject.toml (the list of what REDblocks needs) changes. The installed REDblocks.exe and
REM Claude's connection to it keep the version you last built with build_exes.bat; rebuild only to update those.
setlocal
cd /d "%~dp0\.."
if /i not "%~1"=="nopull" (
  git pull --ff-only || echo git pull did not work: starting the code you have.
)
set PY=.dev_venv\Scripts\python.exe
if not exist %PY% (
  echo First run: setting up .dev_venv. This takes a few minutes, once.
  python -m venv .dev_venv || goto :err
  %PY% -m pip install --upgrade pip >nul
)
fc /b pyproject.toml .dev_venv\pyproject.stamp >nul 2>&1
if errorlevel 1 (
  echo Installing what REDblocks needs...
  %PY% -m pip install -e ".[app,mcp]" "mcp[cli]" || goto :err
  copy /y pyproject.toml .dev_venv\pyproject.stamp >nul
)
REM a build leaves copies of the examples and block sources inside the package; from source the app reads the
REM live ones in this folder instead, so clear the copies (build_exes.bat makes them again)
if exist simblocks\app\examples rmdir /s /q simblocks\app\examples
if exist simblocks\app\block_sources.json del /q simblocks\app\block_sources.json
start "" .dev_venv\Scripts\pythonw.exe -m simblocks.app.main
exit /b 0
:err
echo SETUP FAILED: the real error is in the lines above. (It needs git and Python 3.10-3.12 from python.org on PATH.)
pause
exit /b 1

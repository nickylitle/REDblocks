# simblocks

Building blocks, a system-file compiler and a verification harness for fixed-time-step explicit dynamics sims.
Runs on its own: no Claude needed to run, verify or report a system file.

```
pip install .                    # from this folder (scripts/)
simblocks check  model.yaml      # validate, compile, draw the schematic
simblocks run    model.yaml      # nominal outputs + energy ledger
simblocks set    model.yaml d="12 mm" shaft_mat=7075
simblocks sweep  model.yaml h "0.5 m" "1 m" "2 m"
simblocks report model.yaml      # full verification battery + PDF/HTML report
```

The file format is documented in `../references/system_file.md`. Models are 1D (one coordinate per body),
2D (rigid bodies in x, y and spin with points, circles and lines; contacts, pins, contact friction, gravity) or 3D
(z up; bodies slide in x, y, z and keep their orientation, spin about any axis, or rotate freely with an inertia
tensor; points, spheres and plane patches; ball joints). Examples: `examples/systems/` (1D), `examples/systems2d/`
and `examples/systems3d/`.

## Big studies on your own machine (all cores)

```
simblocks batch examples/case_studies/combat_robot/spinner_vs_spinner_job.yaml
simblocks batch job.yaml --workers 12 --n 2000     # pick cores and sample count
simblocks batch-info spinner_vs_spinner_results.zip
```

- Uses every CPU core by default (one process per core, numba compiled once and cached).
- Writes progress to `<results>.zip.partial.jsonl` as it goes. Interrupt it any time; running the same command
  again resumes where it stopped (`--fresh` starts over).
- The finished `<name>_results.zip` holds plain JSON: machine info, the job file, every run's inputs and
  outputs, the raw contact episodes, the finer-step reruns (dt check) and the verification report. Send that one
  file back for interpretation and graphing.
- A job file names the sim, the configurations to compare and the random inputs; see the header of
  `simblocks/batch.py`.

## Desktop node editor (no Claude, no internet)

`pip install ".[app]"` then `simblocks app`. On Windows, run `packaging/build_exes.bat` once (Python 3.10-3.12): it
builds `dist/Simblocks/Simblocks.exe` and offers to install it for you (no admin rights). After that, press the Windows
key and type Simblocks; right-click it > Pin to taskbar to keep it there. Model files can open with it (right-click a
.yaml > Open with > Simblocks), and it uninstalls from Settings > Apps. Re-run the .bat to update. Blender-style nodes read and write system files; Run, Study (Range and
Monte Carlo modifiers, all cores), Verify and Report work offline, with live results while you edit, templates,
undo, copy/paste, groups and a personal group library, CSV export nodes, notes with LaTeX, custom materials and custom
block folders, forgiving block search and a full in-app Help (F1). Settings live in `~/.simblocks/app.json`.

## One-click runner (no Python needed after the build)

Build once on Windows: double-click `packaging/build_exes.bat` (needs Python 3.10-3.12 from python.org).
It writes `dist/simblocks_runner.exe`.

Use: put `<name>_job.yaml` and the sim `.py` it names in the same folder as the exe, double-click it (or drag a job
file onto it). It uses every core, resumes if interrupted, and writes `<name>_results.zip` next to the job.
Send that zip back. New sims do not need a new exe: only a new job file and sim file.

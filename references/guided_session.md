# Guided session

Run this whenever the skill is invoked without a complete request (for example a bare `/explicit-sim-builder`,
"help me simulate something", or a request that leaves the question, geometry or failure definition open).
The user should only need to call the skill; every later step is offered to them as a short choice.

## Ground rules

- **Skip what is already known.** Read the request, attachments, project docs and earlier turns first. Start at
  the first step whose answer is still missing. Never re-ask something the user already said.
- **One question round per turn.** Ask with `AskUserQuestion`, at most 3 questions per round, 2 to 4 options
  each. Put the recommended option first and mark it "(Recommended)". The user can always type "Other".
- **Defaults beat waiting.** If a step is not blocking, state the default you are using and move on.
  "Just run it" or "you pick" means: take every default, list them as assumptions, continue.
- **Unattended sessions** (scheduled, or the user said they will check back): skip all questions, take
  defaults, and put the assumption list at the top of the report.
- Keep every message short: the setup card and the result, nothing in between except the task list.

## Step 1: what are we doing?

Ask one question, "What do you want to do?":

| Option | Leads to |
|---|---|
| Simulate a hit, impact or maneuver (new sim) | Step 2 |
| Re-run or change an existing sim | List `scripts/examples/case_studies/*/` and the user's earlier models; ask which one and what to change; go to Step 4 |
| Turn sim results into loads for FE (Ansys, MATLAB) | Ask which sim and which hit (worst overall / worst per direction / a given one); follow the load-export rule in SKILL.md; deliver force + moment tables in the user's frame |
| Build a material or composite card | Ask for the material text or layup; `load_curve` / `laminate_card`; deliver the card and its plot |

"Other" covers: run a big study on my machine (Step 4, option 3), diagram or verify an existing sim,
debug a sim, explain how a block works.

## Step 2: the setup card

Fill in a setup card from everything known plus defaults, and send it (SendUserMessage) as one table:

| Item | Value | Source |
|---|---|---|
| Question to answer | e.g. "does the wedge survive a 12k rpm spinner hit" | user |
| Bodies and DOFs | e.g. wedge (hinged, 1 rotation), spinner (spin + translation) | assumed |
| Geometry | key dimensions in mm | user / drawing / assumed |
| Materials | catalog names (`resolve_material`) | user / assumed |
| Speeds and initial state | rpm, closing speed, angles | user / assumed |
| Contacts | which surface hits which, friction, restitution range | assumed |
| Supports | wheels, floor, mounts | assumed |
| Outputs | what the report will show | derived from the question |

Mark every assumed row. Then ask: "Correct anything, or say go." Accept numbers in any units, a sketch, a CAD
screenshot or a drawing (`drawing_input.md`). Ask for a missing number only if a wrong guess would change the
answer; otherwise keep the assumption.

## Step 3: what counts as failure?

Call `candidate_failure_modes(part_type)` and ask one multi-select question, "Which of these count as failure
for this part?", each option with a proposed limit in its label (for example "Permanent tip drop > 5 mm").
The user can type their own limits. If they skip it, use the proposed limits and flag them as placeholders.
Nothing is ever judged against an implicit "over yield".

## Step 4: how big a run?

Ask one question, "How much should I run?":

| Option | What happens |
|---|---|
| One nominal run, fully verified (Recommended) | dt convergence, knockouts, small Monte Carlo (`mc_n` <= 20), energy ledger; report |
| A small sweep here | up to ~64 Sobol samples over the uncertain inputs, wide ranges; report adds probabilities |
| A big study on your machine | write `<name>_job.yaml` with proposed dimensions and n, then ask the user to confirm or edit them; deliver the job file, the sim `.py` and `simblocks_runner.exe` (or the build script); results come back as `<name>_results.zip` |

For the sweep and the big study, propose the sampled dimensions and ranges as a short table and let the user
change them before anything runs. Study size is always the user's call.

## Step 5: build, verify, report

Follow the main workflow (SKILL.md steps 4 to 10) with a task list the user can watch. No questions here unless
verification fails in a way that needs a decision (then show the failing check and the two options).

Deliver, in this order:
1. the answer in one line (pass/fail per failure mode, or the number asked for);
2. the report PDF (answer page, histories, design-around page, verification page) with the Schematic embedded,
   plus a sequence diagram for event results;
3. the YAML system file or sim `.py`, so the user can rerun it without Claude.

## Step 6: what next?

End with one question offering the natural follow-ups that apply, for example:

- Change a parameter and rerun
- Export the worst load as FE tables (force + moment, user's axes)
- Run a bigger study on my machine
- Done

## Phrases that skip straight to a step

Tell the user these work at any time:

| Say | Goes to |
|---|---|
| "simulate <thing>" | Step 2 with the card pre-filled |
| "rerun <sim> with <change>" | Step 4 |
| "export the worst hit for Ansys" | load export |
| "make me the exe job for <sim>" | Step 4, big study |
| "just run it" | defaults everywhere |
| "what can you do?" | Step 1 menu |

# SDL5 formulation lab tasks

Three problems from the SDL5 formulation lab (Acceleration Consortium, University of
Toronto), registered as `sdl5_blend7`, `sdl5_viscometer26` and `sdl5_robot_cell15`
in [`tasks/sdl5.py`](../../src/aobench/tasks/sdl5.py). Ported unchanged from the
hackathon repository's case study 4
(`case_studies/04_sdl_formulation_lab/`), where each has a full `PROBLEM.md` with
the variables, equations, data provenance and baseline results. Problems, data and
baselines: Mahdi Rastegar.

| Task | Dim | What it is | Data | Landscape | Best fixed method (hackathon sweep) |
|---|---:|---|---|---|---|
| `sdl5_blend7` | 7 | Blend 4 thickeners, glycerol, pH and mixing time to hit a target flow curve | Physics model calibrated on AutoVisco and PEPV measurements | Nonlinear, 3 basins, flat pH regions, active bounds | CMA-ES |
| `sdl5_viscometer26` | 26 | Joint calibration of the gap-stepping cone-plate viscometer | 100% measured: 448 points on 23 certified silicone standards | Smooth; narrow curved valley; very flat optimum | L-BFGS-B |
| `sdl5_robot_cell15` | 15 | Lay out a UR3e arm, a CNC and the grasp to minimize the mixer-loading cycle time | Cell geometry and UR3e kinematics from the `sdl_dt` digital twin | 9 continuous + 6 integer IK branch flags; infeasible regions; non-smooth penalties | CMA-ES |

## What is measured and what is modeled

Every objective value is computed in software. No recipe, calibration or layout
found by these tasks has been tested in the lab.

- **Blend:** ingredient viscosities and flow indices are measured; pH response,
  PEG–Carbopol interaction, hydration times, prices and the reference shear rate are
  literature-based assumptions (`model.ASSUMED`). The target flow curve is
  illustrative.
- **Viscometer:** all data are measured; the objective is the lab's own calibration
  equation.
- **Robot cell:** cycle times and clearances are model estimates from a reduced
  kinematic model, not timed on the robot. The collision model is conservative by
  about 1–2 cm.

## Files

| Folder | Files |
|---|---|
| `blend/` | `model.py` (the physics model), `ingredient_calibration.json` (read by the model), the measurement files it was calibrated on, `target_flow_curve.csv`, `reference_optimum.json` |
| `viscometer/` | `height_normalized.csv` (the measured torque data), `reference_optimum.json` |
| `robot_cell/` | `model.py` (kinematics, timing and collision model), `cell_geometry.json` (from `sdl_dt`), `reference_optimum.json` |

The only edits to the two `model.py` files are their data paths, which now point to
this flat layout.

## Reference optima

Each `f_opt` is a best-known value from an offline search, not a proven optimum
(the search method is in each `reference_optimum.json`). The robot cell's best-known
layout is **not collision-free**: its minimum clearance is −25 mm, and no fully
collision-free layout has been found yet. Report it as an open problem or with
feasibility as a separate metric.

## Notes for the agent runner

These tasks name their science in this README and in the model files. Keep both out
of any agent workspace; the agent should see only the anonymous `problem.yaml`
(`x0..`, `y`), as for the other benchmark tasks. The data files are listed in each
task's `private_data_paths`, so the isolation preflight checks that the agent cannot
read them.

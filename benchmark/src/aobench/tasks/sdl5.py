"""Three optimization problems from the SDL5 formulation lab (Acceleration Consortium,
University of Toronto), ported from the hackathon repository's case study 4.

- `sdl5_blend7`: blend four thickeners, glycerol, pH and mixing time to hit a target
  flow curve. A physics model calibrated on AutoVisco and PEPV measurements;
  nonlinear, three basins, flat pH regions, active bounds. CMA-ES wins.
- `sdl5_viscometer26`: joint calibration of the gap-stepping cone-plate viscometer
  on 448 measured points from 23 certified silicone standards. Smooth, but a narrow
  curved valley couples two parameters; L-BFGS-B wins, CMA-ES is far behind.
- `sdl5_robot_cell15`: lay out a UR3e arm, a CNC and the grasp to minimize the
  mixer-loading cycle time, from the `sdl_dt` digital twin. Discrete inverse-
  kinematics branch flags, infeasible regions and non-smooth penalties.

Every objective is computed in software from lab data; no recipe, calibration or
layout found here has been tested in the lab. Each `f_opt` is a best-known value
from an offline search, not a proven optimum. The robot cell's is not collision-free.
Full problem definitions, equations and data provenance:
https://github.com/Mahyar-rajabi94/Cheapest_opt_lunch (case_studies/04_sdl_formulation_lab).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from .base import Task

SDL5 = Path(__file__).resolve().parents[3] / "data" / "sdl5"


def _load_model(folder: str):
    """Import data/sdl5/<folder>/model.py under a unique module name."""
    path = SDL5 / folder / "model.py"
    spec = importlib.util.spec_from_file_location(f"aobench_sdl5_{folder}_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f_opt(folder: str) -> float:
    return float(json.loads((SDL5 / folder / "reference_optimum.json").read_text())["f_opt"])


class SDL5BlendTask(Task):
    """7-D blend: x = [c_carbopol, c_sepineo, c_solagum, c_peg600k (wt%), w_glycerol
    (wt%), pH, t_mix (s)]. Minimize the log10-RMSE to the target flow curve plus
    ingredient-cost, mixing-time and pH-spec penalties."""

    def __init__(self):
        model = _load_model("blend")
        self._model = model.BlendModel()
        self.dim = len(model.VARIABLES)
        self.name = f"sdl5_blend_d{self.dim}"
        self.bounds = np.array([[v[2], v[3]] for v in model.VARIABLES], dtype=float)
        self.f_opt = _f_opt("blend")
        self.private_data_paths = tuple(sorted((SDL5 / "blend").glob("*.*")))

    def _f(self, x: np.ndarray) -> float:
        return float(self._model.objective(x))


_COLUMN = re.compile(r"([\d.]+)kcp_([\d.]+)_torque_%_rpm_([\d.]+)")


class SDL5ViscometerTask(Task):
    """26-D calibration: x = [log10 k, p, h_c, beta_1 .. beta_23]. The model is
    D^_ij = k * mu_j**p / (h_i + h_c) + beta_j * mean_i(D_ij), fitted to the observed
    drag D_ij = torque_ij / RPM_j. Minimize the RMS relative drag error in percent."""

    def __init__(self):
        path = SDL5 / "viscometer" / "height_normalized.csv"
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        header, body = rows[0], rows[1:]
        h = np.array([float(r[0]) for r in body])
        fluid, heights, drag, mu, mean_drag = [], [], [], [], []
        for j, column in enumerate(header[1:], start=1):
            _, certified_kcp, rpm = (float(g) for g in _COLUMN.fullmatch(column).groups())
            torque = np.array([float(r[j]) if r[j] else np.nan for r in body])
            ok = np.isfinite(torque)
            d = torque[ok] / rpm
            fluid.append(np.full(ok.sum(), j - 1))
            heights.append(h[ok])
            drag.append(d)
            mu.append(certified_kcp * 1000.0)
            mean_drag.append(d.mean())
        m = len(mu)
        self._fluid = np.concatenate(fluid)
        self._h = np.concatenate(heights)
        self._drag = np.concatenate(drag)
        self._mu = np.array(mu)[self._fluid]
        self._mean_drag = np.array(mean_drag)

        self.dim = 3 + m
        self.name = f"sdl5_viscometer_d{self.dim}"
        self.bounds = np.array(
            [[-11.0, -6.0],  # log10 k
             [1.0, 3.0],  # p, viscosity exponent
             [0.01, 1.0]]  # h_c, mm
            + [[-1.0, 1.0]] * m  # beta_j, offset as a fraction of the mean drag
        )
        self.f_opt = _f_opt("viscometer")
        self.private_data_paths = (path, SDL5 / "viscometer" / "reference_optimum.json")

    def _f(self, x: np.ndarray) -> float:
        log_k, p, h_c = x[:3]
        predicted = 10.0**log_k * self._mu**p / (self._h + h_c) + (x[3:] * self._mean_drag)[self._fluid]
        rel = (predicted - self._drag) / self._drag
        return 100.0 * float(np.sqrt(np.mean(rel**2)))


class SDL5RobotCellTask(Task):
    """15-D layout: x = [base_x, base_y, base_yaw, cnc_x, cnc_y, lid_open, tilt_x,
    tilt_y, pick_azimuth, shoulder_A, elbow_A, wrist_A, shoulder_B, elbow_B, wrist_B].
    The last six are inverse-kinematics branch flags, declared here as 0/1 integers
    (the hackathon version took them as reals in [0, 1] read as >= 0.5). Minimize the
    penalized cycle time in seconds."""

    N_FLAGS = 6

    def __init__(self):
        self._model = _load_model("robot_cell")
        self.dim = len(self._model.NAMES)
        self.name = f"sdl5_robot_cell_d{self.dim}"
        self.bounds = np.asarray(self._model.BOUNDS, dtype=float).copy()
        n_cont = self.dim - self.N_FLAGS
        self.bounds[n_cont:] = [0.0, 1.0]
        self.variables = (
            [{"type": "continuous", "low": float(lo), "high": float(hi)} for lo, hi in self.bounds[:n_cont]]
            + [{"type": "integer", "low": 0, "high": 1}] * self.N_FLAGS
        )
        self.f_opt = _f_opt("robot_cell")
        self.private_data_paths = tuple(sorted((SDL5 / "robot_cell").glob("*.*")))

    def _f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).copy()
        x[-self.N_FLAGS:] = x[-self.N_FLAGS:] >= 0.5  # as the original model reads a flag
        return float(self._model.objective(x))


class SDL5RobotCellFacingTask(SDL5RobotCellTask):
    """16-D layout: the 15 variables above plus `mixer_facing`, a categorical choice of
    which way the mixer's front (its handle side) points: 0, 90, 180 or 270 degrees,
    turned about the centre of its footprint. The mixer stays in place; the deck, the
    CNC and the robot base keep their cell-frame bounds. Facing 0 is the 15-D task."""

    FACING = "mixer_facing"

    def __init__(self):
        super().__init__()
        self.dim = len(self._model.NAMES_WITH_FACING)
        self.name = f"sdl5_robot_cell_facing_d{self.dim}"
        # Baselines see a continuous box; [0, 4) is read as floor -> quarter turns.
        self.bounds = np.vstack([self.bounds, [0.0, 3.999]])
        self.variables = self.variables + [
            {"type": "categorical", "choices": [str(d) for d in self._model.FACINGS_DEG]}
        ]
        self.f_opt = _f_opt_facing()
        self.private_data_paths = tuple(sorted((SDL5 / "robot_cell").glob("*.*")))

    def _f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).copy()
        x[9:15] = x[9:15] >= 0.5
        return float(self._model.objective(x))


def _f_opt_facing() -> float:
    path = SDL5 / "robot_cell" / "reference_optimum_facing.json"
    return float(json.loads(path.read_text())["f_opt"])

"""Core tests: encoding, data.csv, and whole blocks driven through `aopt` the way a user would."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aoptcore import cli, observations  # noqa: E402
from aoptcore.encoding import Encoder  # noqa: E402
from aoptcore.problem import ProblemError, parse  # noqa: E402

MIXED = {
    "name": "mixed",
    "objective": {"name": "yield", "direction": "maximize"},
    "variables": [
        {"name": "temp", "type": "continuous", "low": 40, "high": 120, "resolution": 0.5},
        {"name": "pressure", "type": "continuous", "low": 0.5, "high": 50, "log": True},
        {"name": "cycles", "type": "integer", "low": 1, "high": 12},
        {"name": "solvent", "type": "categorical", "choices": ["water", "ethanol", "dmso"]},
        {"name": "rpm", "type": "ordinal", "choices": [200, 400, 800, 1600]},
    ],
    "budget": 40,
}
BOX = {"name": "box", "objective": {"name": "loss"},
       "variables": [{"name": f"x{i}", "type": "continuous", "low": -5, "high": 5} for i in range(3)],
       "budget": 40}


def yield_of(c):
    """A synthetic objective to maximize over MIXED."""
    s = {"water": 0.0, "ethanol": 1.0, "dmso": 0.4}[c["solvent"]]
    return (s - ((c["temp"] - 90) / 40) ** 2 - (math.log10(c["pressure"]) - 0.7) ** 2
            - ((c["cycles"] - 7) / 6) ** 2 - abs(c["rpm"] - 800) / 1600)


def loss_of(c):
    return sum((c[f"x{i}"] - 1) ** 2 for i in range(3))


def make(tmp_path, problem, rows=()):
    root = tmp_path / ".agentoptimizer"
    root.mkdir()
    (root / "problem.yaml").write_text(yaml.safe_dump(problem))
    args = ["--project", str(root), "init"]
    if rows:
        src = tmp_path / "initial.csv"
        with open(src, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        args += ["--import", str(src)]
    assert cli.main(args) == 0
    return root


def aopt(root, *args):
    return cli.main(["--project", str(root), *args])


def propose(root, decision="**Decision:** test", **p):
    (root / "agent/proposal.json").write_text(json.dumps(p))
    (root / "agent/decision.md").write_text(decision)
    return aopt(root, "commit")


def read(root):
    with open(root / "data.csv", newline="") as f:
        return list(csv.DictReader(f))


def measure(root, f, objective, fail=()):
    """Fill every pending row like a user would: edit the value cell, leave status alone."""
    rows = read(root)
    for r in rows:
        if r["status"] == "pending":
            if r["id"] in fail:
                r["status"] = "failed"
                continue
            c = {k: r[k] for k in r}
            for k in ("temp", "pressure", "x0", "x1", "x2"):
                if k in c:
                    c[k] = float(c[k])
            if "cycles" in c:
                c["cycles"], c["rpm"] = int(c["cycles"]), int(c["rpm"])
            r[objective] = repr(f(c))
    with open(root / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_block(root, f, objective, rounds=100):
    for _ in range(rounds):
        assert aopt(root, "suggest") == 0
        if not any(r["status"] == "pending" for r in read(root)):
            return
        measure(root, f, objective)
    raise AssertionError("block did not finish")


# ---- problem and encoding ----------------------------------------------------------------------
def test_problem_rejects_bad_names_and_types():
    bad = dict(BOX, variables=[{"name": "id", "type": "continuous", "low": 0, "high": 1}])
    with pytest.raises(ProblemError):
        parse(bad)
    with pytest.raises(ProblemError):
        parse(dict(BOX, variables=[{"name": "a", "type": "categorical", "choices": ["x"]}]))


def test_encoding_round_trip_and_box_identity():
    enc = Encoder(parse(MIXED))
    assert enc.dim == 1 + 1 + 1 + 3 + 1
    c = {"temp": 87.5, "pressure": 3.2, "cycles": 4, "solvent": "dmso", "rpm": 800}
    assert enc.decode(enc.encode(c)) == c
    box = Encoder(parse(BOX))
    assert box.space()["bounds"] == [[-5.0, 5.0]] * 3 and box.all_continuous


def test_resolution_and_rounding():
    enc = Encoder(parse(MIXED))
    x = enc.encode({"temp": 87.5, "pressure": 3.2, "cycles": 4, "solvent": "dmso", "rpm": 800})
    x[0], x[2], x[6] = 87.74, 4.4, 2.6  # temp, cycles, rpm index
    c = enc.decode(x)
    assert c["temp"] == 87.5 and c["cycles"] == 4 and c["rpm"] == 1600


# ---- data ---------------------------------------------------------------------------------------
def test_import_and_maximize(tmp_path):
    rows = [{"temp": 60, "pressure": 1, "cycles": 2, "solvent": "water", "rpm": 400, "yield": 3.0},
            {"temp": 100, "pressure": 10, "cycles": 8, "solvent": "ethanol", "rpm": 800, "yield": 7.0}]
    root = make(tmp_path, MIXED, rows)
    data = read(root)
    assert [r["status"] for r in data] == ["measured", "measured"] and data[0]["block"] == "0"
    X, y, _ = observations(root)
    assert X.shape == (2, 7) and list(y) == [-3.0, -7.0]  # maximized -> minimized for optimizers


def test_import_rejects_out_of_range(tmp_path):
    rows = [{"temp": 300, "pressure": 1, "cycles": 2, "solvent": "water", "rpm": 400, "yield": 3.0}]
    with pytest.raises(AssertionError):  # init refuses the row
        make(tmp_path, MIXED, rows)


# ---- blocks -------------------------------------------------------------------------------------
@pytest.mark.parametrize("script,args", [
    ("lib/sobol.py", {}), ("lib/random_search.py", {}), ("lib/cma_es.py", {"sigma0": 0.3}),
    ("lib/de.py", {"pop_size": 6}), ("lib/pso.py", {"swarm": 5}), ("lib/nelder_mead.py", {}),
    ("lib/turbo.py", {"n_init": 4}), ("lib/lbfgsb.py", {}), ("lib/powell.py", {}),
    ("lib/bobyqa.py", {}), ("lib/cobyla.py", {}),
])
def test_library_blocks_on_mixed_space(tmp_path, script, args):
    root = make(tmp_path, MIXED)
    assert propose(root, script=script, label="main", args=args, n_points=8, state="restart") == 0
    run_block(root, yield_of, "yield")
    rows = read(root)
    assert 1 <= len(rows) <= 8 and all(r["status"] == "measured" for r in rows)


@pytest.mark.parametrize("script", ["lib/gp_bo.py", "lib/mixed_bo.py"])
def test_bo_one_point_at_a_time(tmp_path, script):
    root = make(tmp_path, MIXED)
    args = {"n_init": 3, "num_restarts": 2, "raw_samples": 32}
    assert propose(root, script=script, label="bo", args=args, n_points=5, state="restart") == 0
    for _ in range(5):
        assert aopt(root, "suggest") == 0
        assert sum(r["status"] == "pending" for r in read(root)) == 1
        measure(root, yield_of, "yield")
    assert aopt(root, "suggest") == 0
    assert len(read(root)) == 5


def test_continue_resumes_and_warm_start(tmp_path):
    root = make(tmp_path, BOX)
    assert propose(root, script="lib/cma_es.py", label="cma", args={"sigma0": 0.3}, n_points=6, state="restart") == 0
    run_block(root, loss_of, "loss")
    assert propose(root, script="lib/cma_es.py", label="cma", n_points=6, state="continue") == 0
    run_block(root, loss_of, "loss")
    assert propose(root, script="lib/turbo.py", label="tr", args={"n_init": 2}, n_points=4,
                   state="restart", warm_start=True) == 0
    run_block(root, loss_of, "loss")
    assert len(read(root)) == 16
    st = json.loads((root / "state/tr.json").read_text())
    assert len(st["told"]) == 16  # 12 warm-start rows + 4 of its own


def test_local_solver_resumes_after_each_reload(tmp_path):
    root = make(tmp_path, BOX)
    assert propose(root, script="lib/nelder_mead.py", label="nm", n_points=12, state="restart") == 0
    run_block(root, loss_of, "loss")
    assert propose(root, script="lib/powell.py", label="pw", n_points=12, state="restart", warm_start=True) == 0
    run_block(root, loss_of, "loss")
    rows = read(root)
    keys = [tuple(r[f"x{i}"] for i in range(3)) for r in rows]
    assert len(set(keys)) == len(keys), "a reloaded solver asked for a point it had already been told"


def test_parallel_requests(tmp_path):
    root = make(tmp_path, dict(BOX, parallel=4))
    assert propose(root, script="lib/cma_es.py", label="cma", args={"sigma0": 0.3}, n_points=8, state="restart") == 0
    assert aopt(root, "suggest") == 0
    assert sum(r["status"] == "pending" for r in read(root)) == 4


def test_batch_method_lists_whole_block(tmp_path):
    root = make(tmp_path, BOX)
    assert propose(root, script="lib/sobol.py", label="design", n_points=6, state="restart") == 0
    assert aopt(root, "suggest") == 0
    assert sum(r["status"] == "pending" for r in read(root)) == 6


def test_replicates_on_noisy_problem(tmp_path):
    rows = [{"x0": 1, "x1": 1, "x2": 1, "loss": 0.1}]
    root = make(tmp_path, dict(BOX, noise={"present": True}), rows)
    pts = [{"x0": 1.0, "x1": 1.0, "x2": 1.0}] * 2
    assert propose(root, script="lib/fixed_points.py", label="rep", args={"points": pts}, n_points=2,
                   state="restart") == 0
    assert aopt(root, "suggest") == 0
    data = read(root)
    assert [r["replicate_of"] for r in data[1:]] == ["1", "1"]


def test_deterministic_repeat_is_not_measured_again(tmp_path):
    rows = [{"x0": 1, "x1": 1, "x2": 1, "loss": 0.1}]
    root = make(tmp_path, BOX, rows)
    pts = [{"x0": 1.0, "x1": 1.0, "x2": 1.0}, {"x0": 2.0, "x1": 1.0, "x2": 1.0}]
    assert propose(root, script="lib/fixed_points.py", label="probe", args={"points": pts}, n_points=2,
                   state="restart") == 0
    assert aopt(root, "suggest") == 0
    data = read(root)
    assert len(data) == 2 and data[1]["x0"] == "2.0"


def test_failed_rows_and_cancel_on_new_block(tmp_path):
    root = make(tmp_path, BOX)
    assert propose(root, script="lib/random_search.py", label="rs", n_points=4, state="restart") == 0
    assert aopt(root, "suggest") == 0
    measure(root, loss_of, "loss", fail={"2"})
    assert aopt(root, "suggest") == 0
    assert propose(root, script="lib/sobol.py", label="s", n_points=3, state="restart") == 0
    assert aopt(root, "suggest") == 0
    assert propose(root, script="lib/sobol.py", label="s2", n_points=2, state="restart") == 0
    data = read(root)
    assert [r["status"] for r in data].count("cancelled") == 3
    assert data[1]["status"] == "failed"


def test_rejections(tmp_path):
    root = make(tmp_path, BOX)
    assert propose(root, script="lib/nope.py", label="a", n_points=2) == 1
    assert propose(root, script="lib/sobol.py", label="Bad", n_points=2) == 1
    assert propose(root, script="lib/sobol.py", label="a", n_points=99) == 1
    assert propose(root, script="lib/sobol.py", label="a", n_points=2, state="continue") == 1
    assert propose(root, script="../x.py", label="a", n_points=2) == 1


def test_custom_optimizer_in_work(tmp_path):
    root = make(tmp_path, BOX)
    (root / "work/mine.py").write_text(
        "import sys\nfrom random_search import Optimizer as R\n\nclass Optimizer(R):\n    pass\n")
    assert propose(root, script="work/mine.py", label="mine", n_points=3, state="restart") == 0
    run_block(root, loss_of, "loss")
    assert len(read(root)) == 3

"""python -m aobench: run the benchmark and print who is good and who is bad on each task.

    python -m aobench --out benchmark/runs/main                    # everything, 10 seeds
    python -m aobench --out benchmark/runs/try --seeds 2 --solvers sobol lbfgsb cobyla cma_es
    python -m aobench --out benchmark/runs/main --report-only

The report is printed and written to <out>/report.md.
"""

import argparse
from pathlib import Path

from .run import SOLVERS, default_workers, report, run_all
from .tasks import DIM, SPECS


def main():
    p = argparse.ArgumentParser(prog="python -m aobench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="*", default=list(SPECS), choices=list(SPECS))
    p.add_argument("--solvers", nargs="*", default=SOLVERS, choices=SOLVERS)
    p.add_argument("--seeds", type=int, default=10, help="seeds 0..n-1 (BBOB instances 1..n)")
    p.add_argument("--dim", type=int, default=DIM)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    out = a.out / f"d{a.dim}"
    if not a.report_only:
        run_all(out, a.tasks, a.solvers, range(a.seeds), a.dim, a.workers)
    solvers = ["sobol"] + [s for s in a.solvers if s != "sobol"]
    text = report(out, a.tasks, solvers)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

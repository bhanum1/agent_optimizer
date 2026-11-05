"""Export every mixture's excess properties from sdagent (run inside sdagent, with its .venv).

Excess = mixture value minus the volume-fraction chord through the pure components
(sdagent src/utils/excess.py), with viscosity mixed as ln(cP). Pure bases follow
sdagent: campaigns 1-10 use the basis recovered exactly from EV_lab_results.csv
(benchmarking/pures.py); campaigns 11+ use their measured pure rows.

    cd ~/Documents/sdagent && .venv/bin/python <this file> > excess_rows.csv
"""
import csv
import math
import sys

from benchmarking.pures import recover_campaign_pure_basis

rows = list(csv.DictReader(open("data/processed/measurements.csv")))
recovered, measured = recover_campaign_pure_basis(), {}
for r in rows:
    ids = r["fluid_ids"].split(";")
    if len(ids) == 1 and int(r["measured_in_campaign"]) >= 11:
        for p in ("cp", "rho", "miu"):
            if r[p]:
                measured.setdefault(ids[0], {}).setdefault(p, float(r[p]))
scale = lambda p, v: math.log(v) if p == "miu" else v
out = csv.writer(sys.stdout)
out.writerow(["campaign", "fluid_ids", "volume_fractions", "excess_cp", "excess_rho", "excess_ln_miu"])
for r in rows:
    campaign, ids = int(r["measured_in_campaign"]), r["fluid_ids"].split(";")
    if len(ids) == 1:
        continue
    basis = recovered if campaign <= 10 else measured
    f = [float(v) for v in r["volume_fractions"].split(";")]
    f = [v / sum(f) for v in f]
    out.writerow([campaign, r["fluid_ids"], r["volume_fractions"],
                  *[repr(scale(p, float(r[p])) - sum(w * scale(p, basis[i][p]) for i, w in zip(ids, f)))
                    if r[p] else "" for p in ("cp", "rho", "miu")]])

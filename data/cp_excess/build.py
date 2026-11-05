"""Build the private heat-capacity-excess pool: campaign 11's mixtures (the near-exhaustive
cohort), excess cp above ideal volume-fraction mixing, with outlier rows removed.

Cleaning: a row is dropped when its excess cp differs from the median of the other
compositions of the same fluids by more than K robust sigmas, where sigma is 1.4826 *
the median absolute deviation of those leave-one-out residuals over the campaign.

Features: the volume-fraction-weighted mean of each component's RDKit 2D descriptors
(descriptors constant or non-finite across the components dropped).

Inputs (proprietary, kept in git-ignored external/fluid_mixtures/):
    excess_rows.csv    from ../viscosity_excess/export_excess.py, run inside sdagent
    fluid_library.csv  sdagent data/processed/fluid_library.csv (SMILES by CID)
Run with an interpreter that has RDKit (e.g. sdagent's .venv), from the repository root:
    python data/cp_excess/build.py
Writes external/fluid_mixtures/cp_excess.csv and data/cp_excess/provenance.json.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

CAMPAIGN, K = "11", 5.0
RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parents[1] / "external/fluid_mixtures"
OUT = PRIVATE / "cp_excess.csv"

rows = [r for r in csv.DictReader((PRIVATE / "excess_rows.csv").open(newline=""))
        if r["campaign"] == CAMPAIGN and r["excess_cp"]]
x = np.array([float(r["excess_cp"]) for r in rows])
siblings = defaultdict(list)
for k, r in enumerate(rows):
    siblings[r["fluid_ids"]].append(k)
residual = np.array([x[k] - np.median(x[[j for j in siblings[r["fluid_ids"]] if j != k]])
                     if len(siblings[r["fluid_ids"]]) > 1 else 0.0 for k, r in enumerate(rows)])
sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
keep = np.abs(residual) <= K * sigma
kept = [r for r, k in zip(rows, keep) if k]

smiles = {r["id"]: r["smiles"] for r in csv.DictReader((PRIVATE / "fluid_library.csv").open(newline=""))}
fluids = sorted({i for r in kept for i in r["fluid_ids"].split(";")})
D = np.array([[fn(Chem.MolFromSmiles(smiles[i])) for _, fn in Descriptors._descList] for i in fluids], float)
use = np.isfinite(D).all(0) & (np.ptp(np.nan_to_num(D), 0) > 0)
D, index = D[:, use], {f: k for k, f in enumerate(fluids)}

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"x{i}" for i in range(D.shape[1])] + ["y"])
    for n, r in enumerate(kept):
        ids, v = r["fluid_ids"].split(";"), np.array(r["volume_fractions"].split(";"), float)
        w.writerow([f"m{n}"] + [repr(float(t)) for t in (v / v.sum()) @ D[[index[i] for i in ids]]]
                   + [r["excess_cp"]])

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = [n for n, _ in Descriptors._descList]
(HERE / "provenance.json").write_text(json.dumps({
    "source": f"sdagent data/processed/measurements.csv via export_excess.py, campaign {CAMPAIGN}",
    "excess_rows_sha256": sha(PRIVATE / "excess_rows.csv"),
    "fluid_library_sha256": sha(PRIVATE / "fluid_library.csv"),
    "data_sha256": sha(OUT), "rdkit": rdkit.__version__,
    "cleaning": {"k_sigma": K, "sigma_j_per_kg_k": float(sigma), "rows_before": len(rows),
                 "rows_dropped": int((~keep).sum())},
    "rows": len(kept), "fluids": len(fluids), "features": int(use.sum()),
    "descriptors": [n for n, u in zip(names, use) if u],
    "objective": "maximize y = cp_mix - sum_i phi_i cp_i, volume fractions phi, cp in J/(kg K)",
}, indent=1) + "\n")
print(f"{len(kept)} mixtures ({int((~keep).sum())} dropped, sigma {sigma:.1f}), {len(fluids)} fluids, "
      f"{int(use.sum())} descriptors")

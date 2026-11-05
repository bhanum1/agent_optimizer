"""Build the private mixture pool from sdagent's lab measurements: every row with
measured mu, cp and rho (campaigns 1-11; campaign 12 has no cp).

Features: the volume-fraction-weighted mean of each component's RDKit 2D
descriptors (descriptors constant or non-finite across the components dropped).
y: log10 laminar pump work mu / (cp * rho)^2 in m3 s K2 / J; lower is better.

Inputs, from sdagent's data/processed/ (proprietary, so kept in git-ignored external/):
    external/fluid_mixtures/measurements.csv, external/fluid_mixtures/fluid_library.csv
Run with an interpreter that has RDKit (e.g. sdagent's .venv), from the repository root:
    python data/fluid_mixtures/build.py
Writes external/fluid_mixtures/data.csv and data/fluid_mixtures/provenance.json.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parents[1] / "external/fluid_mixtures"

rows = [r for r in csv.DictReader((PRIVATE / "measurements.csv").open(newline=""))
        if r["cp"] and r["miu"] and r["rho"]]
smiles = {r["id"]: r["smiles"] for r in csv.DictReader((PRIVATE / "fluid_library.csv").open(newline=""))}
fluids = sorted({i for r in rows for i in r["fluid_ids"].split(";")})
D = np.array([[fn(Chem.MolFromSmiles(smiles[i])) for _, fn in Descriptors._descList] for i in fluids], float)
use = np.isfinite(D).all(0) & (np.ptp(np.nan_to_num(D), 0) > 0)
D, index = D[:, use], {f: k for k, f in enumerate(fluids)}

with open(PRIVATE / "data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"x{i}" for i in range(D.shape[1])] + ["y"])
    for n, r in enumerate(rows):
        ids, v = r["fluid_ids"].split(";"), np.array(r["volume_fractions"].split(";"), float)
        x = (v / v.sum()) @ D[[index[i] for i in ids]]
        mu, cp, rho = float(r["miu"]) * 1e-3, float(r["cp"]), float(r["rho"]) * 1e3  # Pa s, J/(kg K), kg/m3
        w.writerow([f"m{n}"] + [repr(float(t)) for t in x] + [repr(math.log10(mu / (cp * rho) ** 2))])

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = [n for n, _ in Descriptors._descList]
(HERE / "provenance.json").write_text(json.dumps({
    "source": "sdagent data/processed/measurements.csv, every row with cp, miu and rho (campaigns 1-11)",
    "measurements_sha256": sha(PRIVATE / "measurements.csv"),
    "fluid_library_sha256": sha(PRIVATE / "fluid_library.csv"),
    "data_sha256": sha(PRIVATE / "data.csv"), "rdkit": rdkit.__version__,
    "rows": len(rows), "campaigns": sorted({r["measured_in_campaign"] for r in rows}, key=int), "fluids": len(fluids), "features": int(use.sum()),
    "descriptors": [n for n, u in zip(names, use) if u],
    "objective": "minimize y = log10(mu / (cp * rho)^2), mu in Pa s, cp in J/(kg K), rho in kg/m3",
}, indent=1) + "\n")
print(len(rows), "mixtures,", len(fluids), "fluids,", int(use.sum()), "descriptors")

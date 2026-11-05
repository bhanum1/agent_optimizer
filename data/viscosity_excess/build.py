"""Build the private viscosity-excess pool: every measured mixture (campaigns 1-12)
with its excess ln(mu) below ideal volume-fraction mixing; lower is better.

Features: the volume-fraction-weighted mean of each component's RDKit 2D descriptors
(descriptors constant or non-finite across the components dropped).

Inputs (proprietary, kept in git-ignored external/fluid_mixtures/):
    excess_rows.csv    from export_excess.py, run inside sdagent
    fluid_library.csv  sdagent data/processed/fluid_library.csv (SMILES by CID)
Run with an interpreter that has RDKit (e.g. sdagent's .venv), from the repository root:
    python data/viscosity_excess/build.py
Writes external/fluid_mixtures/viscosity_excess.csv and data/viscosity_excess/provenance.json.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parents[1] / "external/fluid_mixtures"
OUT = PRIVATE / "viscosity_excess.csv"

rows = [r for r in csv.DictReader((PRIVATE / "excess_rows.csv").open(newline="")) if r["excess_ln_miu"]]
smiles = {r["id"]: r["smiles"] for r in csv.DictReader((PRIVATE / "fluid_library.csv").open(newline=""))}
fluids = sorted({i for r in rows for i in r["fluid_ids"].split(";")})
D = np.array([[fn(Chem.MolFromSmiles(smiles[i])) for _, fn in Descriptors._descList] for i in fluids], float)
use = np.isfinite(D).all(0) & (np.ptp(np.nan_to_num(D), 0) > 0)
D, index = D[:, use], {f: k for k, f in enumerate(fluids)}

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id"] + [f"x{i}" for i in range(D.shape[1])] + ["y"])
    for n, r in enumerate(rows):
        ids, v = r["fluid_ids"].split(";"), np.array(r["volume_fractions"].split(";"), float)
        w.writerow([f"m{n}"] + [repr(float(t)) for t in (v / v.sum()) @ D[[index[i] for i in ids]]]
                   + [r["excess_ln_miu"]])

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = [n for n, _ in Descriptors._descList]
(HERE / "provenance.json").write_text(json.dumps({
    "source": "sdagent data/processed/measurements.csv via export_excess.py, every mixture with miu",
    "excess_rows_sha256": sha(PRIVATE / "excess_rows.csv"),
    "fluid_library_sha256": sha(PRIVATE / "fluid_library.csv"),
    "data_sha256": sha(OUT), "rdkit": rdkit.__version__, "rows": len(rows),
    "campaigns": sorted({r["campaign"] for r in rows}, key=int), "fluids": len(fluids),
    "features": int(use.sum()), "descriptors": [n for n, u in zip(names, use) if u],
    "objective": "minimize y = ln(mu_mix) - sum_i phi_i ln(mu_i), volume fractions phi, mu in cP",
}, indent=1) + "\n")
print(len(rows), "mixtures,", len(fluids), "fluids,", int(use.sum()), "descriptors")

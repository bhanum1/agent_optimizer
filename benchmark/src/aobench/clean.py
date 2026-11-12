"""Readable summaries of agent runs: a small folder per finished campaign, from the full run logs.

    python -m aobench.clean                                   every run folder in benchmark/runs
    python -m aobench.clean --runs benchmark/runs/opus_high   one run

Writes, for each campaign that ended at its budget:

    <out>/<run>/README.md                 one row per campaign: final regret, calls, cost, methods used
    <out>/<run>/<task>/seed_<s>/data.csv  the campaign's data.csv, every requested and measured point
    <out>/<run>/<task>/seed_<s>/trace.md  call by call: the agent's reasoning, the block it proposed and
                                          why, and how far the regret got; tool calls in one line each
    <out>/<run>/<task>/seed_<s>/scripts/  the source of every optimizer it ran, each version once

<out> is benchmark/clean_runs by default, which is kept in git, unlike the run folders. Absolute paths
are written as <workspace> and ~, since they name this machine's user.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

RUNS = Path("benchmark/runs")
OUT = Path("benchmark/clean_runs")  # kept in git, unlike the run folders
# Absolute paths in the logs name this machine's user: a workspace becomes <workspace>, any other home ~.
WORKSPACE = re.compile(r"(?:/(?:Users|home)/[^/\s]+)?/\S*?/workspaces/[0-9a-f]{16}(?:\.home)?")
HOME = re.compile(r"/(?:Users|home)/[^/\s`'\"]+")
DASHED = re.compile(r"-(?:Users|home)-[^-/\s]+-")  # the same path in a folder name, as claude's temp folders spell it


def scrub(text: str) -> str:
    return DASHED.sub("-~-", HOME.sub("~", WORKSPACE.sub("<workspace>", text)))
FLOOR = 1e-8


def events(path: Path):
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def call_content(folder: Path) -> tuple[list[tuple[str, str]], str | None]:
    """One call's reasoning, messages and tool calls in order, as (kind, text), and the model it ran on."""
    out, model = [], None
    for e in events(folder / "events.jsonl"):
        if e.get("type") == "system" and e.get("subtype") == "init":
            model = e.get("model")
        if e.get("type") != "assistant":
            continue
        for b in e.get("message", {}).get("content") or []:
            kind = b.get("type")
            if kind == "thinking" and b.get("thinking", "").strip():
                out.append(("reasoning", b["thinking"].strip()))
            elif kind == "text" and b.get("text", "").strip():
                out.append(("message", b["text"].strip()))
            elif kind == "tool_use":
                args = b.get("input", {})
                what = (args.get("command") or args.get("file_path") or args.get("pattern") or
                        args.get("description") or "")
                what = scrub(" ".join(str(what).split()))  # before cutting, which would split a path
                out.append(("tool", f"{b.get('name')}: {what[:110]}" + ("…" if len(what) > 110 else "")))
    return out, model


def section(text: str, name: str) -> str:
    """The `**Name:** ...` paragraph of a decision.md, if any."""
    for para in text.split("\n"):
        if para.startswith(f"**{name}:**"):
            return para[len(name) + 5:].strip()
    return ""


def describe(p: dict) -> str:
    """A proposal in one line: the method, its settings and the block's size."""
    if "points" in p or p.get("script") == "points":
        pts = p.get("points") or p.get("args", {}).get("points") or []
        return f"**{p.get('label')}**: a list of {len(pts)} points chosen by the agent"
    args = ", ".join(f"{k}={v}" for k, v in (p.get("args") or {}).items())
    return (f"**{p.get('label')}**: optimizer `{p.get('script')}`" + (f" ({args})" if args else "") +
            f", {p.get('n_points')} points, state {p.get('state')}, " +
            ("warm-started on all data" if p.get("warm_start") else "not warm-started"))


def lg(r: float) -> str:
    return f"{math.log10(max(r, FLOOR)):.2f}"


def campaign(run: Path, task: str, seed_dir: Path, out: Path) -> dict | None:
    end = json.loads((seed_dir / "end.json").read_text()) if (seed_dir / "end.json").is_file() else {}
    if end.get("reason") != "budget":
        return None
    seed = int(seed_dir.name.split("_")[1])
    ws = run / "workspaces" / Path((seed_dir / "workspace.txt").read_text().strip()).name / ".agentoptimizer"
    regret = [float(r["regret"]) for r in csv.DictReader(open(seed_dir.parent / f"seed_{seed}.csv"))]
    rows = list(csv.DictReader(open(ws / "data.csv")))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ws / "data.csv", out / "data.csv")

    # Measurements per block, in block order: where each block's evaluations fall in the regret curve.
    per_block: dict[int, int] = {}
    for r in rows:
        if r["status"] == "measured":
            per_block[int(r["block"])] = per_block.get(int(r["block"]), 0) + 1
    spans, used = {}, 0
    for b in sorted(per_block):
        spans[b] = (used + 1, used + per_block[b])
        used += per_block[b]

    # Each block's optimizer source, under the block's number and label.
    # Each distinct optimizer source once, named after the first block that ran it.
    blocks, saved = {}, {}
    for f in sorted((ws / "blocks").glob("[0-9]*.json")):
        meta = json.loads(f.read_text())
        blocks[int(meta["block"])] = meta
        src = ws / meta["source_file"] if meta.get("source_file") else None
        if meta.get("script") != "points" and src and src.is_file():
            text = src.read_text()
            if text not in saved:
                saved[text] = f"{int(meta['block']):03d}_{meta['label']}.py"
                (out / "scripts").mkdir(exist_ok=True)
                (out / "scripts" / saved[text]).write_text(scrub(text))
            meta["saved"] = saved[text]

    entries = {e["call"]: e for e in events(seed_dir / "calls.jsonl") if "call" in e}
    resumed_before = [e for e in events(seed_dir / "calls.jsonl") if e.get("resumed")]
    calls = sorted(int(d.name) for d in (seed_dir / "calls").iterdir() if d.name.isdigit())
    resume_points = []  # the call number each resumption came before
    lines = list(events(seed_dir / "calls.jsonl"))
    for i, e in enumerate(lines):
        if e.get("resumed"):
            nxt = next((x["call"] for x in lines[i + 1:] if "call" in x), None)
            resume_points.append(nxt)

    model, body, table, block_no, cost, methods = None, [], [], 0, 0.0, []
    for n in calls:
        folder = seed_dir / "calls" / f"{n:03d}"
        content, m = call_content(folder)
        model = model or m
        e = entries.get(n)
        cost += (e or {}).get("total_cost_usd") or 0
        if n in resume_points:
            body.append("*The harness was stopped here and the campaign resumed from its workspace.*\n")
        head = f"## Call {n}"
        if e is None:
            body.append(head + ": interrupted\n\nThe harness was stopped during this call; it left no proposal.\n")
            continue
        if e.get("api_error") or (e.get("is_error") and not e.get("total_cost_usd")):
            body.append(head + ": the API could not be reached\n\nThe harness waited and called again.\n")
            continue
        info = f"{e.get('num_turns')} turns, {e.get('seconds', 0):.0f} s, ${e.get('total_cost_usd') or 0:.2f}"
        if not e.get("committed"):
            body.append(head + f": no block ({info})\n\nThe call ended without a proposal that validated" +
                        (f": {e.get('error')}" if e.get("error") else "") + ".\n")
            body += render(content)
            continue
        block_no += 1
        p = e.get("proposal") or {}
        a, b = spans.get(block_no, (None, None))
        before = regret[a - 2] if a and a > 1 else None
        after = regret[b - 1] if b else None
        label = p.get("label") or blocks.get(block_no, {}).get("label")
        methods.append(label)
        span = f"evaluations {a}–{b}" if a else "no evaluations"
        body.append(f"{head}: block {block_no}, {label} ({span})\n\n*{info}*\n")
        body += render(content)
        decision = (folder / "decision.md").read_text() if (folder / "decision.md").is_file() else ""
        body.append("**Proposed:** " + describe(p) + "\n")
        for name in ("Rationale", "Review"):
            if section(decision, name):
                body.append(f"**{name}:** {section(decision, name)}\n")
        if e.get("messages"):
            body.append("**aopt said:** " + " / ".join(str(x) for x in e["messages"]) + "\n")
        if after is not None:
            body.append(f"**Result:** log10 regret {lg(before) + ' → ' if before is not None else ''}{lg(after)}\n")
        table.append((block_no, n, label, "points" if "points" in p else
                      f"[{p.get('script')}](scripts/{blocks.get(block_no, {}).get('saved')})", f"{a}–{b}" if a else "–",
                      lg(after) if after is not None else "–"))

    head = [f"# {task}, seed {seed}\n",
            f"Run `{run.name}`, model {model or '?'}. Final log10 regret **{lg(regret[-1])}** after "
            f"{len(regret)} evaluations; {len(calls)} calls, ${cost:.2f}." +
            (f" Resumed {len(resumed_before)} time(s)." if resumed_before else "") + "\n",
            "Regret is the noise-free gap to the optimum of the best point so far (1e-8 counts as solved).\n",
            "| Block | Call | Method | Script | Evaluations | log10 regret after |", "|---|---|---|---|---|---|"]
    head += [f"| {b} | {c} | {lab} | {s} | {sp} | {r} |" for b, c, lab, s, sp, r in table]
    (out / "trace.md").write_text(scrub("\n".join(head) + "\n\n" + "\n".join(body)))
    return {"task": task, "seed": seed, "regret": lg(regret[-1]), "calls": len(calls), "cost": cost,
            "methods": methods, "model": model}


def render(content: list[tuple[str, str]]) -> list[str]:
    out, tools = [], []

    def flush():
        if tools:
            out.append(f"<details><summary>{len(tools)} tool call{'s' if len(tools) > 1 else ''}</summary>\n\n" +
                       "\n".join(f"- `{t.replace('`', chr(39))}`" for t in tools) + "\n\n</details>\n")
            tools.clear()

    for kind, text in content:
        if kind == "tool":
            tools.append(text)
            continue
        flush()
        if kind == "reasoning":
            out.append("**Reasoning:**\n\n" + "\n".join("> " + l if l else ">" for l in text.splitlines()) + "\n")
        else:
            out.append("**Message to the user:**\n\n" + text + "\n")
    flush()
    return out


def clean(run: Path, out: Path) -> int:
    done = []
    for seed_dir in sorted(run.glob("*/agent/seed_*/"), key=lambda p: (p.parts[-3], int(p.name.split("_")[1]))):
        task = seed_dir.parts[-3]
        r = campaign(run, task, seed_dir, out / run.name / task / seed_dir.name)
        if r:
            done.append(r)
    if not done:
        return 0
    lines = [f"# {run.name}\n", f"Model {done[0]['model']}. One folder per task and seed; each has `trace.md` "
             "(the agent's reasoning and decisions, call by call), `data.csv` and `scripts/` (its optimizers).\n",
             "| Task | Seed | Final log10 regret | Calls | Cost | Methods, block by block |", "|---|---|---|---|---|---|"]
    lines += [f"| [{d['task']}]({d['task']}/seed_{d['seed']}/trace.md) | {d['seed']} | {d['regret']} | {d['calls']} | "
              f"${d['cost']:.2f} | {' → '.join(map(str, d['methods']))} |" for d in done]
    lines.append(f"\nTotal ${sum(d['cost'] for d in done):.2f} over {len(done)} campaigns.")
    (out / run.name / "README.md").write_text("\n".join(lines) + "\n")
    return len(done)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", type=Path, help="run folders (default: every one in benchmark/runs)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    runs = a.runs or sorted(p for p in RUNS.iterdir() if p.is_dir() and any(p.glob("*/agent/seed_*/end.json")))
    for run in runs:
        n = clean(run, a.out)
        print(f"{run.name}: {n} finished campaigns" + (f" -> {a.out / run.name}" if n else ", skipped"))


if __name__ == "__main__":
    main()

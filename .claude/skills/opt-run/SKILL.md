---
name: opt-run
description: Advise a real optimization campaign one block at a time. Reads a campaign folder (problem.toml, data.csv, decisions.md), decides the next block's method, settings and size, writes a reproducible propose.py, runs it, and adds the proposals to data.csv as pending experiments. Also starts new campaigns, records results and reports status. Use when asked what experiments to run next, to start an optimization campaign, or to record results.
argument-hint: "<campaign folder> | init <folder> | record <folder> <results.csv> | status <folder>"
allowed-tools: Bash(.venv/bin/python scripts/opt_run.py:*)
---

# Advise the next block

Arguments: $ARGUMENTS

Work from the repository root. The decision policy is
[prompts/advisor.md](../../../prompts/advisor.md); read it in full before
deciding anything. `scripts/opt_run.py` does all the bookkeeping (run it with
`.venv/bin/python`), and the steps below say when to call each command.

## 1. Read the request

- **`init <folder>`:** start a campaign (section 2).
- **`record <folder> <file>`:** run `opt_run.py record <folder> <file>`, then
  `opt_run.py summarize <folder>`, and report the new results and the
  incumbent in a few lines. Stop there.
- **`status <folder>`:** run `opt_run.py summarize <folder>` and report. Stop there.
- **`<folder>`**, or words naming one: propose the next block (section 3).
- **Nothing:** ask for a campaign folder, or offer to start one.

## 2. Start a campaign

1. If `<folder>/problem.toml` exists already, say so and stop.
2. Ask in one message for what is missing: what is optimized and in which
   direction; each parameter's range and units, whether it is whole-number,
   and whether it spans orders of magnitude (log scale); the total budget
   and how many experiments run in parallel; known failure modes and noise;
   and whether experiments are chosen from a fixed candidate list.
3. Run `opt_run.py init <folder>`, then rewrite `problem.toml` from the
   answers; the template's comments show every key. For a candidate list,
   add `[pool]` with `file = "candidates.csv"` (columns `candidate_id` and
   one per parameter).
4. Run `opt_run.py check <folder>` until it passes, show the parameter table,
   and ask the user to confirm it.
5. If they have earlier results in a CSV, run `opt_run.py import <folder> <file>`.
6. Offer to propose the first block.

## 3. Propose the next block

1. **Check.** Run `opt_run.py check <folder>`. On errors, show them and stop:
   the user fixes `data.csv` or `problem.toml`.
2. **Pending and budget.** Run `opt_run.py summarize <folder>`.
   - If experiments are pending, say how many and ask whether their results
     are in (record them first) or whether to propose alongside them. Stop
     unless the request already said to go ahead.
   - If no budget remains, append the final entry to `decisions.md`
     (policy, section 3) and report the outcome. Stop there.
3. **Open the block.** Run `opt_run.py new-block <folder>`. It prints the
   block number N and its folder, `blocks/NNN/`.
4. **Read.** Read `prompts/advisor.md`, `problem.toml`,
   `blocks/NNN/summary.json`, `decisions.md` if it exists, the previous
   block's `propose.py`, and `data.csv` when the summary is not enough. Read
   nothing else about the objective. The `agentoptimizer` source is fine for
   API details.
5. **Decide and record.** Follow the policy, section 2. Append the
   `## Block N` entry to `decisions.md`, creating the file with a
   `# Decisions` heading if needed.
6. **Write `blocks/NNN/propose.py`** to the contract in the policy, section 3.
7. **Run.** Run `opt_run.py run <folder>`. It runs the script twice, refuses
   proposals that differ, and validates them. If it fails, fix the script
   and run again. If the fix changes the decision (method, size or
   settings), append a `**Revision:**` line under this block's entry saying
   what changed and why. After three failed attempts, stop and report.
8. **Accept.** Run `opt_run.py accept <folder>`.
9. **Report:**
   - the block number;
   - the proposals as a table in the user's units, with their new ids;
   - the decision in two or three sentences: the method, why, and what this
     block should show;
   - how to send results back. Either fill in the objective (and status
     `done`, or `failed` with an empty value) for those ids in `data.csv`, or
     put `id,<objective>[,status,notes]` rows in a CSV and call
     `/opt-run record <folder> <file>`.

## Rules

- Never exceed the budget, and never edit, invent or estimate a result.
- Do not read the objective's implementation or any simulator. For the
  bundled example, `examples/reaction/lab.py` is off limits.
- To run different points than proposed, the user can edit the pending rows
  in `data.csv` before running them (`check` validates them). Note the change
  in `decisions.md`.

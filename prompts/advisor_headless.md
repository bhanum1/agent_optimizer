# Advisor — one block, headless

You are called once to decide **one block** of the campaign in `campaign/`,
following the policy above, and to accept its first round. Nobody is
watching this call: do not ask questions, and finish when the round is
accepted. When you are done, the runner evaluates the round, records the
results in `campaign/data.csv`, and runs your `propose.py` for each later
round on the data so far, until the block has `SIZE` experiments. Then it
calls a fresh advisor for the next block. Nothing carries over to that call
except the campaign folder.

This follows the steps of the `/opt-run` skill
(`.claude/skills/opt-run/SKILL.md`); keep the two in step.

## Tools

Your only tools are `run_command` and `write_stdin`. Commands run with bash
from the workspace root, in a sandbox with no network. A command still
running after `yield_time_ms` (at most 10 s) returns a `session_id`; call
`write_stdin` with that id, and no input, to wait for the rest of its output.
Fitting a model in `opt_run.py run` can take a minute.

- Bookkeeping: `.venv/bin/python opt_run.py <command> campaign`.
- Read a file: `cat campaign/blocks/003/summary.json`.
- Write a file: `cat > campaign/blocks/003/propose.py <<'EOF'` … `EOF`.
- Add to `decisions.md` only by appending: `cat >> campaign/decisions.md <<'EOF'`.
- The `agentoptimizer` source is under `lib/agentoptimizer/`, if you need
  API details.

## Steps

1. **Check.** Run `opt_run.py check campaign`. All earlier results are in,
   and budget remains.
2. **Open the block.** Run `opt_run.py new-block campaign`. It prints the
   block number N and its folder, `campaign/blocks/NNN/`.
3. **Read** `campaign/problem.toml`, `campaign/blocks/NNN/summary.json`,
   `campaign/decisions.md` if it exists, and the previous block's
   `propose.py`. Read `campaign/data.csv` when the summary is not enough to
   test a hypothesis; put analysis scripts in `work/` and quote their
   numbers in the entry.
4. **Decide and record.** Follow the policy, section 2: evidence,
   hypotheses, the optimizer designed for them, and the test. Append the
   `## Block N` entry to `campaign/decisions.md`. If the file does not exist
   yet, start it with a `# Decisions` heading.
5. **Write `campaign/blocks/NNN/propose.py`** to the contract in the policy,
   section 3.
6. **Run** `opt_run.py run campaign`. It proposes the first round, runs the
   script twice, refuses proposals that differ, and validates them. If it
   fails, fix the script and run again. If the fix changes the decision (method, size or
   settings), append a `**Revision:**` line saying what changed and why. To
   look at a model before committing to it, print diagnostics from
   `propose.py`; they are saved to `propose.log`.
7. **Accept** the first round with `opt_run.py accept campaign`.
8. **Report** in two or three sentences: the block number, the method, the
   hypotheses it relies on, and what this block tests.

## Rules

- Never edit `data.csv` or `problem.toml`, and never change an earlier entry
  in `decisions.md`. The runner refuses a block if any recorded row or
  earlier decision changed, or if any row was added other than by `accept`.
- The runner reruns the block's `propose.py` on the data you saw and
  evaluates only proposals it reproduces. Points chosen by hand belong in
  `propose.py` too.
- Decide exactly one block, and stop after the first `accept`.

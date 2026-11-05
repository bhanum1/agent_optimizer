"""Campaign loop checks with a scripted fake agent. The last test also runs
the optimizer host inside the real sandbox when the Codex CLI is installed."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from agentoptimizer.loop import campaign, results, sandbox
from test_loop_block import COUNTER, Quadratic



class FakeAgent:
    """Plays back one scripted reply per call: a proposal dict, raw text, or None (no files)."""

    def __init__(self, run, replies):
        self.run, self.replies, self.prompts = run, list(replies), []

    def __call__(self, number, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        agent = self.run.workspace / "agent"
        if reply is not None:
            (agent / "proposal.json").write_text(reply if isinstance(reply, str) else json.dumps(reply))
            (agent / "decision.md").write_text(f"Decision {number}.")
            (agent / "context.md").write_text(f"Context after call {number}.")
        return 0


def counter(n, state="restart", label="counter", **args):
    return {"script": "work/counter.py", "label": label, "args": args, "n_points": n, "state": state}


class LoopCampaignTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "run"
        self.run = campaign.Campaign.prepare(Quadratic(), 6, 0, self.root)
        (self.run.workspace / "work/counter.py").write_text(COUNTER)

    def play(self, replies):
        agent = FakeAgent(self.run, replies)
        return self.run.run(agent), agent, (self.run.workspace / "decisions.md").read_text()

    def test_blocks_run_until_budget(self):
        reason, agent, decisions = self.play([counter(4), counter(2, "continue")])
        self.assertEqual(reason, "budget")
        X, _, _ = self.run.store.history()
        self.assertEqual([x[0] for x in X], [1, 2, 3, 4, 5, 6])
        self.assertIn("## Decision after evaluation 0 (call 1)\n\nDecision 1.", decisions)
        self.assertIn("## Decision after evaluation 4 (call 2)", decisions)
        self.assertIn("Budget exhausted", decisions)
        self.assertIn("4 of 6 evaluations used; 2 remaining", agent.prompts[1])
        self.assertEqual((self.run.private / "calls/002/context.md").read_text(), "Context after call 2.")

    def test_early_stop_is_rejected(self):
        reason, _, decisions = self.play([counter(2), {"n_points": 0}, counter(4, "continue")])
        self.assertEqual((reason, len(self.run.store)), ("budget", 6))
        self.assertIn("n_points must be an integer from 1 to 4", decisions)

    def test_invalid_proposals_are_noted_and_retried(self):
        reason, _, decisions = self.play([
            counter(7),                                   # over budget
            counter(1, "continue"),                       # no saved state
            counter(1),
            "{not json",
            counter(5, "continue")])
        self.assertEqual((reason, len(self.run.store)), ("budget", 6))
        self.assertIn("n_points must be an integer from 1 to 6", decisions)
        self.assertIn('no saved state for label "counter"', decisions)
        self.assertIn("not valid JSON", decisions)

    def test_repeated_failures_stop(self):
        reason, _, decisions = self.play([None, None, None])
        self.assertEqual(reason, "failures")
        self.assertEqual(decisions.count("decision.md is missing")
                         + decisions.count("proposal.json is missing"), 3)

    def test_script_outside_lib_or_work_is_rejected(self):
        bad = dict(counter(1), script="../private/journal.csv")
        _, _, decisions = self.play([bad, dict(counter(1), script="host.py"), counter(6)])
        self.assertEqual(decisions.count("must be a .py file under lib/ or work/"), 2)

    def test_crash_is_recorded_and_forces_restart(self):
        reason, _, decisions = self.play([counter(4, crash_at=3), counter(1, "continue"), counter(4)])
        self.assertIn("Block 1 (counter) stopped after 2 of 4 evaluations", decisions)
        self.assertIn("RuntimeError: boom", decisions)
        self.assertIn('no saved state for label "counter"', decisions)

    def test_edited_script_cannot_continue(self):
        def edit_then_continue(number, prompt):
            if number == 2:
                (self.run.workspace / "work/counter.py").write_text(COUNTER + "\n# edited\n")
            return agent(number, prompt)
        agent = FakeAgent(self.run, [counter(1), counter(1, "continue"), counter(5)])
        self.run.run(edit_then_continue)
        self.assertIn("changed since", (self.run.workspace / "decisions.md").read_text())

    def test_audit_passes_and_results_are_written(self):
        self.play([counter(3, crash_at=3), "{bad", counter(2), counter(1, "continue"), counter(1, "continue")])
        report = results.audit(self.run, None)
        self.assertTrue(report["passed"], report)
        cfg = {"task": "quad", "agent": "fake", "model": "none", "effort": "none"}
        results.write(self.run, cfg, "budget", report, "start")
        with open(self.root / "results/trace.csv") as f:
            trace = list(__import__("csv").DictReader(f))
        self.assertEqual([r["optimizer"] for r in trace], ["counter"] * 6)
        self.assertEqual([float(r["y_best_so_far"]) for r in trace], [1] * 6)
        meta = json.loads((self.root / "results/meta.json").read_text())
        self.assertEqual((meta["evaluations_used"], meta["blocks"], meta["best_y"]), (6, 4, 1.0))

    def test_audit_detects_tampering(self):
        self.play([counter(2), counter(4, "continue")])
        decisions = self.run.workspace / "decisions.md"
        decisions.write_text(decisions.read_text().replace("Decision 1.", "Edited."))
        with self.assertRaises(RuntimeError):
            results.audit(self.run, None)
        decisions.write_text(self.run.written)
        data = self.run.workspace / "data.csv"
        data.write_text(data.read_text().replace(",1.0,", ",0.5,", 1))
        with self.assertRaises(RuntimeError):
            results.audit(self.run, None)

    @unittest.skipUnless(shutil.which("codex") and (REPO / ".agent-venv/bin/python").exists(),
                         "needs the Codex CLI and .agent-venv")
    def test_host_runs_in_sandbox(self):
        root = REPO / "runs" / ("test-" + Path(tempfile.mktemp()).name)
        try:
            run = campaign.Campaign.prepare(Quadratic(), 3, 0, root)
            runtime = sandbox.runtime_info(REPO / ".agent-venv/bin/python")
            sandbox.link_venv(run.workspace, runtime)
            roots = sandbox.runtime_roots(runtime, shutil.which("codex"))
            policy = {"codex": shutil.which("codex"), "workspace": str(run.workspace),
                      "host": sandbox.permissions(run.workspace, roots, sandbox.HOST_WRITES)}
            run.prefix = {"agent": [], "host": sandbox.prefix(policy, "host")}
            (run.workspace / "work/counter.py").write_text(COUNTER)
            reason = run.run(FakeAgent(run, [counter(2), counter(1, "continue")]))
            self.assertEqual(reason, "budget", (run.workspace / "decisions.md").read_text())
            self.assertEqual(len(run.store), 3)
            self.assertTrue((run.workspace / "state/counter.state").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

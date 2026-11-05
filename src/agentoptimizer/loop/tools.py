"""MCP stdio server giving the agent one tool: run a command in the sandbox.

    python tools.py POLICY_JSON

Commands block until they finish or time out; nothing persists between
calls except files, since optimizer state lives on disk.
"""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agentoptimizer.loop.sandbox import environment, prefix  # noqa: E402

MAX_SECONDS = 600
TOOL = {"name": "run_command", "description":
        "Run a bash command in your sandboxed workspace and return its output. "
        "Use .venv/bin/python; put scripts in work/. No network. "
        f"Commands are killed after timeout_s seconds (default 120, max {MAX_SECONDS}).",
        "inputSchema": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_SECONDS}},
            "required": ["command"], "additionalProperties": False}}


def run(policy: dict, script: str, timeout: int) -> dict:
    workspace = Path(policy["workspace"])
    process = subprocess.Popen([*prefix(policy, "agent"), "/bin/bash", "-c", script], cwd=workspace,
                               env=environment(workspace), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        timed_out = True
    text = output.decode(errors="replace")
    if len(text) > 40000:
        text = "[... output truncated ...]\n" + text[-40000:]
    return {"output": text, "exit_code": process.returncode, "timed_out": timed_out}


def main():
    policy = json.loads(Path(sys.argv[1]).read_text())
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        try:
            method = request["method"]
            if method == "initialize":
                result = {"protocolVersion": request["params"]["protocolVersion"],
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "sandbox", "version": "1.0"}}
            elif method == "tools/list":
                result = {"tools": [TOOL]}
            elif method == "tools/call" and request["params"]["name"] == "run_command":
                args = request["params"].get("arguments", {})
                if not isinstance(args.get("command"), str) or len(args["command"]) > 250000:
                    raise ValueError("Invalid command")
                timeout = min(max(int(args.get("timeout_s", 120)), 1), MAX_SECONDS)
                result = {"content": [{"type": "text", "text": json.dumps(run(policy, args["command"], timeout))}]}
            elif method in ("resources/list", "resources/templates/list"):
                result = {"resources" if method == "resources/list" else "resourceTemplates": []}
            elif method == "ping":
                result = {}
            else:
                raise ValueError("Unsupported RPC")
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32603, "message": str(exc)}}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()

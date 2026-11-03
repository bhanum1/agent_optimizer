"""The headless agent: one CLI call (Claude Code or Codex) whose only tools are
the sandboxed executor's run_command and write_stdin, and what its events say.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .execution import overrides
from .loop import USAGE, LoopError

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]

# MCP server name per agent CLI; Claude Code reserves "workspace".
SERVER = {'codex': 'workspace', 'claude': 'isolated'}

TOOLS = ['run_command', 'write_stdin']


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def cli_environment():
    # Launched from inside a Claude Code session (the /optimize skill, say), the
    # agent CLI would inherit that session's proxy URL, messaging socket, effort
    # and mode settings. Dropping them starts it as it starts from a terminal.
    if 'CLAUDECODE' not in os.environ:
        return dict(os.environ)
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('CLAUDE', 'MCP_')) and k != 'ANTHROPIC_BASE_URL'}


def executor_args(private):
    return ['-B', str(Path(__file__).with_name('execution.py')), str(private / 'execution.json')]


def codex_command(args, workspace, private, policy, system, logs):
    config = dict(policy['permissions'])
    config.update({
        'developer_instructions': system,
        'model_reasoning_effort': args.effort,
        'mcp_servers': {'workspace': {'command': sys.executable, 'args': executor_args(private),
            'enabled': True, 'required': True, 'startup_timeout_sec': 30, 'tool_timeout_sec': 45,
            'enabled_tools': TOOLS, 'default_tools_approval_mode': 'approve'}},
    })
    for feature in ['shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins', 'remote_plugin',
                    'memories', 'hooks', 'shell_snapshot', 'browser_use', 'browser_use_external',
                    'computer_use', 'in_app_browser', 'multi_agent', 'multi_agent_v2',
                    'image_generation', 'skill_search', 'workspace_dependencies', 'code_mode']:
        config['features.' + feature] = False
    config['features.code_mode_host'] = True
    write_json(logs / 'effective_overrides.json', config)
    return [str(args.codex), '--no-daemon', 'exec', '--ignore-user-config', '--ignore-rules',
            '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never',
            '-m', args.model, '-C', str(workspace), *overrides(config),
            '-o', str(logs / 'agent_final.txt'), '-'], cli_environment()


def claude_command(args, workspace, private, system, logs):
    # No built-in tools, settings, memory, CLAUDE.md, skills or connectors: the only
    # tools are the sandboxed MCP executor's, and anything else is denied unprompted.
    config = {'mcpServers': {SERVER['claude']: {'type': 'stdio', 'command': sys.executable,
                                                'args': executor_args(private)}}}
    write_json(logs / 'claude_mcp.json', config)
    command = [str(args.claude), '-p', '--output-format', 'stream-json', '--verbose',
               '--model', args.model, '--effort', args.effort, '--tools', '',
               '--strict-mcp-config', '--mcp-config', str(logs / 'claude_mcp.json'),
               '--allowedTools', ','.join(f"mcp__{SERVER['claude']}__{t}" for t in TOOLS),
               '--permission-mode', 'dontAsk', '--setting-sources', '', '--disable-slash-commands',
               '--no-session-persistence', '--system-prompt', system]
    # The ceiling keeps the enclosing repository's git status out of the CLI's context.
    env = {**cli_environment(), 'ENABLE_CLAUDEAI_MCP_SERVERS': 'false', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
           'GIT_CEILING_DIRECTORIES': str(workspace.parent)}
    write_json(logs / 'effective_overrides.json', {'command': command[1:], 'mcp': config})
    return command, env


def launch(args, workspace, private, policy, prompt, system, logs):
    """One CLI call with the sandboxed executor as its only tools; logs go to `logs`."""
    logs.mkdir(parents=True, exist_ok=True)
    if args.agent == 'codex':
        command, env = codex_command(args, workspace, private, policy, system, logs)
    else:
        command, env = claude_command(args, workspace, private, system, logs)
    (logs / 'initial_prompt.txt').write_text(prompt)
    with (logs / 'cli_events.jsonl').open('w') as events, (logs / 'cli_stderr.log').open('w') as errors:
        process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                   stdout=events, stderr=errors, text=True)
        try:
            process.communicate(prompt)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    write_json(logs / 'cli_exit.json', {'returncode': process.returncode})
    if args.agent == 'claude':
        results = [e for e in events_of(logs) if e.get('type') == 'result']
        (logs / 'agent_final.txt').write_text((results[-1].get('result') or '') if results else '')
    return process.returncode


def events_of(logs):
    path = logs / 'cli_events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def tool_calls(events):
    """(server, tool) for every tool the model called; anything outside MCP is 'unexpected'."""
    tools = []
    for event in events:
        # Codex reports completed items; Claude reports tool_use blocks named mcp__<server>__<tool>.
        item = event.get('item', {}) if event.get('type') == 'item.completed' else {}
        if item.get('type') == 'mcp_tool_call':
            tools.append((item.get('server'), item.get('tool')))
        elif item.get('type') in ('command_execution', 'file_change', 'web_search'):
            tools.append(('unexpected', item['type']))
        elif event.get('type') == 'assistant':
            for block in event.get('message', {}).get('content', []):
                if block.get('type') in ('tool_use', 'server_tool_use'):
                    _, server, tool = (block.get('name', '').split('__', 2) + ['', ''])[:3]
                    tools.append((server, tool) if block['type'] == 'tool_use' else ('unexpected', block['name']))
    return tools


def only_workspace_tools(tools, agent):
    return all(s == SERVER[agent] and t in TOOLS for s, t in tools)


def system_prompt(agent):
    """The advisor's instructions: the policy, then the headless procedure."""
    return ((REPO / 'prompts/advisor.md').read_text() + '\n'
            + (REPO / 'prompts/advisor_headless.md').read_text()
            + f'\nAll commands and file access use the {SERVER[agent]} MCP tools.\n')


def usage_of(events):
    """Turns, cost and tokens of one CLI call, from its event stream (Claude or Codex)."""
    usage = dict.fromkeys(USAGE, 0)
    for event in events:
        if event.get('type') == 'result':  # Claude: one per call
            tokens = event.get('usage', {})
            usage.update(turns=event.get('num_turns', 0), cost_usd=event.get('total_cost_usd', 0.),
                         input_tokens=tokens.get('input_tokens', 0),
                         cache_write_tokens=tokens.get('cache_creation_input_tokens', 0),
                         cache_read_tokens=tokens.get('cache_read_input_tokens', 0),
                         output_tokens=tokens.get('output_tokens', 0))
        elif event.get('type') == 'turn.completed':  # Codex: input tokens include cached ones
            tokens = event.get('usage', {})
            cached = tokens.get('cached_input_tokens', 0)
            usage['turns'] += 1
            usage['input_tokens'] += tokens.get('input_tokens', 0) - cached
            usage['cache_read_tokens'] += cached
            usage['output_tokens'] += tokens.get('output_tokens', 0)
    return usage


def cli_advisor(args, workspace, private, policy):
    """advise() for the loop: one fresh headless CLI call per attempt."""
    system = system_prompt(args.agent)

    def advise(block, attempt, logs):
        prompt = f'Propose block {block} of the campaign in campaign/, following your instructions.'
        start = time.monotonic()
        launch(args, workspace, private, policy, prompt, system, logs)
        events = events_of(logs)
        tools = tool_calls(events)
        usage = {**usage_of(events), 'tool_calls': len(tools), 'seconds': round(time.monotonic() - start, 1)}
        if not only_workspace_tools(tools, args.agent):
            raise LoopError(f'block {block}: the advisor used a tool outside the sandbox; see {logs}', usage)
        return usage
    return advise

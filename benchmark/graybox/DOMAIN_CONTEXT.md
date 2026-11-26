# Domain-context agent explanations and decisions

This supplement contains saved prompts and model-generated artifacts for the 12 sequential domain-context agent runs in the selected results: 98 model calls. It covers Opus/high on all three tasks and Sonnet/medium on `mix_1`, with seeds 0–2. Offline prior-ranking conditions and other agent information modes are outside this supplement.

Start with a run’s **decision journal** for its block-by-block explanations. Open a numbered call’s `transcript.json` for the recorded model commentary and tool interaction that produced that decision.

| Task | Method | Seed | Calls | Decision journal |
| --- | --- | ---: | ---: | --- |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | 10 | [Read journal](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/decisions.md) |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | 8 | [Read journal](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/decisions.md) |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | 9 | [Read journal](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/decisions.md) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 0 | 6 | [Read journal](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_0/agent_artifacts/decisions.md) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 1 | 7 | [Read journal](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_1/agent_artifacts/decisions.md) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 2 | 6 | [Read journal](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_2/agent_artifacts/decisions.md) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | 9 | [Read journal](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/decisions.md) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | 7 | [Read journal](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/decisions.md) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | 8 | [Read journal](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/decisions.md) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | 8 | [Read journal](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/decisions.md) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | 9 | [Read journal](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/decisions.md) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | 11 | [Read journal](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/decisions.md) |

## Agent-written context files

The original hidden workspace file was `workspace/.agentoptimizer/agent/context.md`. Its final contents are exported as `agent_artifacts/final_agent/context.md`. The controller also saved a copy after each call as `private/context_001.md`, `context_002.md`, and so on; these are exported as `agent_artifacts/calls/001/saved_context.md`, `calls/002/saved_context.md`, etc. These are the agent’s evolving notes, distinct from the supplied `domain-context.json`. The copies are sanitized as described below; the original local files remain untouched.

| Task | Method | Seed | Final context.md | Intermediate context files |
| --- | --- | ---: | --- | --- |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | [Read final notes](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/calls/) |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | [Read final notes](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/calls/) |
| `mix_1` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | [Read final notes](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/calls/) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 0 | [Read final notes](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_0/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_0/agent_artifacts/calls/) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 1 | [Read final notes](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_1/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_1/agent_artifacts/calls/) |
| `mix_1` | `cheapest_lunch_sonnet_5_5_medium_domain_context` | 2 | [Read final notes](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_2/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mix_1/cheapest_lunch_sonnet_5_5_medium_domain_context/seed_2/agent_artifacts/calls/) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | [Read final notes](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/calls/) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | [Read final notes](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/calls/) |
| `mat_5` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | [Read final notes](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/mat_5/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/calls/) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 0 | [Read final notes](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_0/agent_artifacts/calls/) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 1 | [Read final notes](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_1/agent_artifacts/calls/) |
| `sup` | `cheapest_lunch_opus_5_5_high_domain_context` | 2 | [Read final notes](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/final_agent/context.md) | [Browse numbered calls](tasks/sup/cheapest_lunch_opus_5_5_high_domain_context/seed_2/agent_artifacts/calls/) |

Each numbered call folder contains `saved_context.md` alongside its prompt and transcript. For example, `calls/001/saved_context.md` is the agent’s notes saved after call 1.

## Artifact layout

Each exported domain-context seed has an `agent_artifacts/` directory:

```text
agent_artifacts/
  decisions.md              # cumulative recorded block decisions
  domain-context.json       # supplied domain description and descriptor names
  PROBLEM.md
  problem.yaml              # public problem instructions
  blocks/                   # committed block proposals and script snapshots
  final_agent/
    context.md              # final persistent notes
    decision.md             # final decision explanation
    proposal.json           # final proposed block
  calls/
    001/
      system.txt            # controller-supplied system instructions
      prompt.txt            # call-specific user instructions
      saved_context.md      # saved post-call memory, when present
      transcript.json       # sanitized messages in recorded event order
    ...
  manifest.json             # original source hashes and transformation notes
```

## Interpreting the transcript

`transcript.json` contains `messages` with `role` and ordered `content` blocks. The retained types are `text`, `recorded_model_thinking`, `tool_use`, and `tool_result`. A `recorded_model_thinking` block is text already present in the saved provider event stream, not a newly generated explanation or a guarantee of access to all internal computation. Explanations are model claims and can contain errors; use the numerical evaluation trace as the evidence for measured outcomes. No missing reasoning has been reconstructed.

Tool IDs are normalized within each call so inputs can be matched to results. Tool inputs include analysis commands and files written by the model; tool results retain their saved output. `block_or_batch` in the seed’s evaluation trace connects observations to committed files in `blocks/`. A model call and an optimization block are different units: a call may analyze results or revise a proposal before a block is committed.

The journal and final notes are cumulative/final artifacts; they are not snapshots of what the model knew before every call. `saved_context.md` is the context file saved **after** the corresponding call. Use the recorded tool interactions to inspect memory reads and writes during the call. `system.txt` is the application-supplied instruction text; it is not a claim to include every provider-side instruction.

## Privacy and fidelity

The export removes provider/session/request identifiers, thinking signatures, usage/diagnostic metadata, system stream events, stderr, authentication state, and private absolute machine paths. Private paths are replaced with `[PRIVATE_MACHINE_PATH]`; credential redaction markers would appear if a credential were encountered. The content scan found no apparent credentials in the selected artifacts. Instructions, proposal scripts, and tool commands are historical records for reading, not an executable replay environment.

Original local files remain untouched. Each per-run manifest gives original relative source paths and SHA-256 digests. These hashes identify the unsanitized originals; they are not hashes of the transformed export. The top-level [artifact manifest](domain_context_artifacts_manifest.json) lists coverage and redaction counts.

The exported transcripts explain recorded actions but do not include the complete original workspaces, large feature tables, fitted state, or controller authentication/deployment files. Existing numerical traces and result metadata are unchanged.

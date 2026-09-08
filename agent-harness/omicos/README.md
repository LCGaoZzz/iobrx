# Omicos integration

The bundle follows `omicos-core`'s Agent/Skill discovery and MCP stdio
contracts, and `omicos-admin/domains/biology/skills/README.md`'s portable
resource format. It contains no host-specific interpreter paths or tokens.

## Workspace development

From the iobrx checkout:

```bash
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace
```

This adds `agents/iobrx_analyst.md` and `skills/iobrx/`. The Skill's entire
runtime lives under `scripts/` and travels with the catalog resources; it
does not refer back to the original checkout. The installer preflights both
destinations and refuses to overwrite either.

Workspace extension discovery is entitlement-gated in current Omicos. Check
the account/workspace extension capability; copying to a workspace does not
make an Agent available to every subscription. Maintainer offline development
can use the supported roots with Linux/WSL paths:

```bash
export OMICOS_TEMPLATES_DIR=/path/to/analysis-workspace
export OMICOS_SKILL_ROOTS=/path/to/analysis-workspace/skills
export OMICOS_AGENTS_OFFLINE=1
export OMICOS_SKILLS_OFFLINE=1
```

Launch a local test core with that workspace, then verify `/api/agents` contains
`iobrx_analyst` and `/api/skills` contains `iobrx`. Resolve
`scripts/run_iobrx.py` with `include_runtime_path: true` and run `doctor` with
the prepared analysis interpreter. This is a development configuration, not a
way to change the shared production catalog or bypass account permissions.

## Shared Omicos catalog

Use a clean feature branch from current omicos-admin main:

```bash
python agent-harness/install_omicos.py --destination /path/to/omicos-admin-checkout --layout catalog
cd /path/to/omicos-admin-checkout
python scripts/validate_skills.py --skill iobrx
```

The exact destination layout is:

```text
domains/biology/agents/iobrx_analyst.md
domains/biology/skills/iobrx/SKILL.md
domains/biology/skills/iobrx/scripts/...
domains/biology/skills/iobrx/references/...
```

Review and merge a separate admin PR, then follow its normal main-branch
catalog deployment and live verification procedure. The iobrx PR ships an
integration bundle; it does not itself register or deploy a production Agent.
Do not use omicos-admin's legacy top-level agents/skills directories.

## Optional stdio MCP

Install the companion `[mcp]` extra in the prepared Python environment. Add
a server through Omicos Settings → MCP servers using this config shape:

```json
{
  "id": "iobrx",
  "command": "/absolute/path/to/analysis-python",
  "args": ["-m", "iobrx_harness.mcp_server", "--workspace", "/absolute/path/to/workspace"],
  "cwd": "/absolute/path/to/workspace",
  "transport": "stdio",
  "enabled": true,
  "env": {}
}
```

Replace the illustrative paths; use paths native to the host running the
server. If Windows Omicos launches a WSL environment, use `command=wsl.exe`
and prefix the arguments with `-e` and the WSL Python path; the workspace must
be a WSL path too. Do not launch Windows Python against a Linux wheel.

Tools: `iobrx_capabilities`, `iobrx_doctor`, `iobrx_validate`, `iobrx_run`,
`iobrx_status`. Omicos prefixes these with `mcp__iobrx__`. Requests use the
same JSON contract as the CLI; paths are confined to the configured workspace.
Errors return MCP `isError=true`. Analyses use isolated worker processes so
native logs/global thread settings cannot corrupt the protocol or another
call. A cancellation attempts to stop the worker; after a forced kill,
inspect the recorded manifest and never assume completion.

The server exposes local file analysis only. It does not fetch remote URLs,
accept user pickle files, execute arbitrary commands or modify client-wide
configuration. Use the existing Omicos shell job runner for very long work.

"""Optional MCP v1 SDK stdio adapter. Each tool uses an isolated CLI process."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path


def create_server(workspace):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    workspace = Path(workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    server = FastMCP("iobrx", instructions="Use capabilities and validate before run. Paths are confined to the configured workspace.")
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    writes = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

    async def invoke(command, request=None, path=None):
        # Script-relative launch also works from a materialized Omicos Skill,
        # without installing the companion wheel or depending on the cwd.
        worker = Path(__file__).resolve().with_name("worker.py")
        args = [sys.executable, str(worker), command]
        if request is not None:
            args += ["--request", "-", "--workspace", str(workspace)]
        if path is not None:
            args += [path, "--workspace", str(workspace)]
        process = await asyncio.create_subprocess_exec(
            *args, cwd=workspace, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=None,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, _ = await process.communicate(None if request is None else json.dumps(request, allow_nan=False).encode())
        except BaseException:
            if process.returncode is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.communicate(), timeout=5)
                except asyncio.TimeoutError:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    await process.wait()
            raise
        result = json.loads(stdout)
        if process.returncode:
            # MCP uses isError=true while retaining the structured CLI error text.
            raise ValueError(json.dumps(result, ensure_ascii=False))
        return result

    @server.tool(annotations=read_only)
    async def iobrx_capabilities() -> dict:
        """List 11 analyses, supported input scales/IDs and strict JSON request schemas."""
        return await invoke("capabilities")

    @server.tool(annotations=read_only)
    async def iobrx_doctor() -> dict:
        """Check this Python environment and native availability; no analysis is run."""
        return await invoke("doctor")

    @server.tool(annotations=read_only)
    async def iobrx_validate(request: dict) -> dict:
        """Validate a request and input matrix without running an analysis or writing outputs."""
        return await invoke("validate", request=request)

    @server.tool(annotations=writes)
    async def iobrx_run(request: dict) -> dict:
        """Run one iobrx analysis in a new output directory, returning its timed result manifest."""
        return await invoke("run", request=request)

    @server.tool(annotations=read_only)
    async def iobrx_status(path: str) -> dict:
        """Read a run manifest and verify output hashes. Running does not prove a process is still alive."""
        return await invoke("status", path=path)

    return server


def main():
    parser = argparse.ArgumentParser(description="Workspace-scoped iobrx stdio MCP server")
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    try:
        server = create_server(args.workspace)
    except ImportError as exc:
        parser.exit(2, f"MCP dependency unavailable: {exc}. Install the companion package with [mcp].\n")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

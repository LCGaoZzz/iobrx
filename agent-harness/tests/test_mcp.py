import asyncio
import json
import shutil
import sys

import pytest

from conftest import request_for


def test_real_stdio_handshake_tools_run_and_boundary(fixtures, tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    shutil.copyfile(fixtures / "signature.parquet", tmp_path / "input.parquet")
    request = request_for("signature_pca", fixtures, tmp_path / "run")
    request["input"]["path"] = "input.parquet"
    request["output_dir"] = "run"

    async def check():
        config = StdioServerParameters(command=sys.executable, args=["-m", "iobrx_harness.mcp_server", "--workspace", str(tmp_path)])
        async with stdio_client(config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "iobrx_capabilities", "iobrx_doctor", "iobrx_validate", "iobrx_run", "iobrx_status"}
                cap = await session.call_tool("iobrx_capabilities", {})
                assert not cap.isError
                assert "signature_pca" in json.dumps(cap.model_dump())
                validated = await session.call_tool("iobrx_validate", {"request": request})
                assert not validated.isError
                assert not (tmp_path / "run").exists()
                result = await session.call_tool("iobrx_run", {"request": request})
                assert not result.isError, result
                manifest = json.loads((tmp_path / "run/results_manifest.json").read_text())
                assert manifest["status"] == "completed"
                state = await session.call_tool("iobrx_status", {"path": "run"})
                assert not state.isError
                bad = await session.call_tool("iobrx_status", {"path": str(fixtures)})
                assert bad.isError
                request["output_dir"] = str(fixtures / "escaped-output")
                bad = await session.call_tool("iobrx_run", {"request": request})
                assert bad.isError and not (fixtures / "escaped-output").exists()

    asyncio.run(asyncio.wait_for(check(), timeout=120))

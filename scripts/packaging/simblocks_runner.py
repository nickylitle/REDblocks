"""PyInstaller entry script for both programs:
Simblocks.exe (desktop node editor, simblocks/app/main.py; folder build, installed by install.ps1) and
simblocks_runner.exe (one-file batch runner, simblocks/runner_app.py). The executable's name picks which one starts:
anything except a name containing "runner" opens the editor."""
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":       # an AI assistant starts REDblocks as an MCP server
        from simblocks.mcp_server import main
        sys.exit(main())
    if "runner" not in os.path.basename(sys.argv[0]).lower():
        from simblocks.app.main import main
    else:
        from simblocks.runner_app import main
    sys.exit(main())

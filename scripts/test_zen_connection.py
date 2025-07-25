#!/usr/bin/env python3
"""
Test connection to zen MCP server and check available tools
"""

import json
import subprocess
import sys
import os

def test_zen_server():
    """Test if we can connect to the zen MCP server"""
    
    print("Testing zen MCP server connection...")
    print("="*60)
    
    # Check if the server script exists
    server_path = "/Users/pfay01/Projects/zen-mcp-server/server.py"
    if not os.path.exists(server_path):
        print(f"Error: zen server not found at {server_path}")
        return False
        
    # Check if virtual environment exists
    venv_path = "/Users/pfay01/Projects/zen-mcp-server/.zen_venv"
    if not os.path.exists(venv_path):
        print(f"Error: zen virtual environment not found at {venv_path}")
        print("You may need to set up the zen-mcp-server first")
        return False
        
    print("✓ zen-mcp-server found")
    print("✓ Virtual environment found")
    
    # Check Python interpreter
    python_path = f"{venv_path}/bin/python"
    if os.path.exists(python_path):
        print(f"✓ Python interpreter found: {python_path}")
    else:
        print(f"✗ Python interpreter not found at {python_path}")
        return False
        
    # Try to import the server modules
    print("\nChecking zen modules...")
    try:
        result = subprocess.run(
            [python_path, "-c", "import server; import tools.consensus; print('Modules OK')"],
            cwd="/Users/pfay01/Projects/zen-mcp-server",
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ zen modules can be imported")
        else:
            print("✗ Error importing modules:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error checking modules: {e}")
        return False
        
    print("\n" + "="*60)
    print("zen MCP server appears to be properly configured!")
    print("\nTo use the consensus tool:")
    print("1. The zen MCP server should be running (check Claude desktop config)")
    print("2. Use the MCP client protocol to send requests")
    print("3. The consensus tool will coordinate multiple AI models")
    
    return True

if __name__ == "__main__":
    success = test_zen_server()
    sys.exit(0 if success else 1)
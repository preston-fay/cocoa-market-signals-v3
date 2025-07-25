#!/usr/bin/env python3
"""
Use zen consensus tool to get multiple AI perspectives on fixing XGBoost libomp issue on Mac
"""

import json
import subprocess
import sys
import os

# Define the consensus request
consensus_request = {
    "tool": "consensus",
    "request": {
        "query": """
        I need help fixing XGBoost installation on macOS. The error is that libomp (OpenMP library) is missing.
        
        Error details:
        - XGBoost requires libomp for parallel computation
        - Mac doesn't include OpenMP by default
        - Common error: "Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib"
        
        Requirements:
        1. Must make XGBoost work properly - it's a great model and we can't take shortcuts
        2. Need a robust, production-ready solution
        3. Should handle both Intel and Apple Silicon Macs
        4. Must integrate well with Python virtual environments
        
        Please provide comprehensive analysis on:
        1. Best practices for installing XGBoost with libomp on Mac
        2. Alternative approaches if direct installation fails
        3. How to handle this dependency in production environments
        4. Common pitfalls and how to avoid them
        5. Verification steps to ensure it's working correctly
        
        Context: This is for a cocoa market signals analysis project that depends heavily on XGBoost models.
        """,
        "step": "Initial analysis of XGBoost libomp installation issue on macOS",
        "step_number": 1,
        "total_steps": 4,
        "next_step_required": True,
        "findings": "Starting analysis of XGBoost installation issues with missing libomp on macOS. This is a common issue that requires careful handling of OpenMP dependencies.",
        "models": [
            {
                "model": "o3",
                "stance": "neutral",
                "stance_prompt": "Focus on robust, production-ready solutions and best practices"
            },
            {
                "model": "flash",
                "stance": "for",
                "stance_prompt": "Advocate for the most straightforward installation approach using Homebrew"
            },
            {
                "model": "pro",
                "stance": "critical",
                "stance_prompt": "Identify potential issues and edge cases with different approaches"
            },
            {
                "model": "gemini",
                "stance": "neutral",
                "stance_prompt": "Provide alternative solutions and workarounds"
            }
        ],
        "focus_areas": [
            "Installation methods",
            "Dependency management",
            "Virtual environment compatibility",
            "Apple Silicon vs Intel",
            "Production deployment"
        ],
        "temperature": 0.2,
        "thinking_mode": "high",
        "use_websearch": True
    },
    "tool_config": {
        "provider": "anthropic"
    }
}

# Write the request to a temporary file
request_file = "/tmp/xgboost_consensus_request.json"
with open(request_file, "w") as f:
    json.dump(consensus_request, f, indent=2)

print("Created consensus request for XGBoost libomp fix")
print(f"Request saved to: {request_file}")
print("\nRequest details:")
print(json.dumps(consensus_request, indent=2))

# Instructions for running the consensus
print("\n" + "="*60)
print("To run the consensus analysis:")
print("1. Make sure the zen MCP server is running")
print("2. Use the MCP client to send this request")
print("3. The consensus tool will gather perspectives from multiple models")
print("4. You'll receive a comprehensive analysis with multiple viewpoints")
print("\nThe request file is ready at:", request_file)
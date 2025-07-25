#!/usr/bin/env python3
"""
Direct script to run zen consensus analysis for XGBoost libomp fix
"""

import json
import os
import sys
import asyncio
from pathlib import Path

# Add the zen-mcp-server to the path
zen_path = Path("/Users/pfay01/Projects/zen-mcp-server")
if zen_path.exists():
    sys.path.insert(0, str(zen_path))

async def run_consensus_analysis():
    """Run the consensus analysis using the zen MCP server"""
    
    print("Running XGBoost libomp consensus analysis...")
    print("="*60)
    
    # Import necessary modules from zen
    try:
        from tools.consensus import ConsensusTool
        from utils.model_context import ModelContext
        from providers.registry import get_provider
        
        # Initialize the consensus tool
        consensus_tool = ConsensusTool()
        
        # Create the initial request
        initial_request = {
            "prompt": """
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
        }
        
        # Get the provider
        provider_name = os.getenv("ZEN_DEFAULT_PROVIDER", "anthropic")
        api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
        
        if not api_key:
            print(f"Error: {provider_name.upper()}_API_KEY not set")
            return
            
        provider = get_provider(provider_name, api_key)
        
        # Create model context
        context = ModelContext(
            provider=provider,
            model="claude-3-5-sonnet-20241022",  # Default model for initial analysis
            temperature=0.2
        )
        
        # Run the consensus analysis
        print("\nStarting consensus analysis...")
        print("This will gather perspectives from multiple AI models...\n")
        
        # Call the consensus tool
        result = await consensus_tool.call(initial_request, context)
        
        # Print the result
        print("\nConsensus Analysis Result:")
        print("="*60)
        print(json.dumps(result, indent=2))
        
    except ImportError as e:
        print(f"Error importing zen modules: {e}")
        print("\nMake sure the zen-mcp-server is properly installed")
        print("You may need to run this from within the zen-mcp-server directory")
    except Exception as e:
        print(f"Error running consensus: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async function
    asyncio.run(run_consensus_analysis())
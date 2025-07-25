# Using Zen Consensus Tool for XGBoost libomp Fix

## Overview
The zen consensus tool is configured as an MCP (Model Context Protocol) server that can coordinate multiple AI models to provide diverse perspectives on technical problems.

## How to Use Zen Consensus

### 1. Ensure Zen MCP Server is Running
The zen server is configured in your Claude Desktop configuration at:
`/Users/pfay01/Library/Application Support/Claude/claude_desktop_config.json`

### 2. Request Format
When using Claude with the zen MCP server enabled, you can request consensus analysis by asking Claude to use the consensus tool. Here's an example request:

```
Use the zen consensus tool to analyze the XGBoost libomp installation issue on Mac. 
I want perspectives from:
- o3 model (neutral stance) - focus on best practices
- flash model (supportive stance) - advocate for Homebrew approach  
- pro model (critical stance) - identify potential issues
- gemini model (neutral stance) - provide alternatives
```

### 3. The Consensus Process
The consensus tool will:
1. First provide Claude's initial analysis
2. Consult each specified model with their assigned stance
3. Synthesize all perspectives into a comprehensive recommendation

### 4. Example Consensus Request for XGBoost
Here's the structured request that would be sent:

```json
{
  "tool": "consensus",
  "request": {
    "prompt": "Help fix XGBoost installation on macOS with missing libomp...",
    "models": [
      {
        "model": "o3",
        "stance": "neutral",
        "stance_prompt": "Focus on robust, production-ready solutions"
      },
      {
        "model": "flash", 
        "stance": "for",
        "stance_prompt": "Advocate for straightforward Homebrew installation"
      },
      {
        "model": "pro",
        "stance": "critical", 
        "stance_prompt": "Identify edge cases and potential issues"
      }
    ],
    "focus_areas": [
      "Installation methods",
      "Virtual environment compatibility",
      "Apple Silicon vs Intel differences"
    ],
    "thinking_mode": "high",
    "use_websearch": true
  }
}
```

### 5. Benefits of Consensus Analysis
- **Multiple Perspectives**: Get diverse viewpoints on the solution
- **Balanced Decision**: Supportive and critical stances provide balance
- **Expert Knowledge**: Each model brings different expertise
- **Comprehensive Coverage**: Covers edge cases and alternatives

### 6. When to Use Consensus
- Complex technical decisions (like dependency management)
- Architecture choices
- When multiple valid approaches exist
- When you need to consider various trade-offs

## Direct Usage in Claude
Simply ask Claude something like:

"Use the zen consensus tool to help me fix XGBoost libomp issues on Mac. Get perspectives from multiple models including supportive and critical viewpoints."

Claude will then coordinate with the zen MCP server to gather and synthesize the analysis.
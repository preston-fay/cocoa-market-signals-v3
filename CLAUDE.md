# CLAUDE.md - AI Assistant Instructions

## Project Overview
This is the Cocoa Market Signals v3 project - a real-data market analysis system combining:
- Daily cocoa futures prices from Yahoo Finance
- UN Comtrade export data (REAL, not synthetic)
- Weather data from Open-Meteo
- Advanced ML models for signal detection

## CRITICAL: Standards to Follow

### 1. COLOR STANDARDS (Kearney Design System)
**ONLY use these colors:**
- Primary Purple: `#6f42c1`
- Primary Charcoal: `#272b30` 
- White: `#FFFFFF`
- Light Gray: `#e9ecef`
- Medium Gray: `#999999`
- Dark Gray: `#52575c`
- Border Gray: `#7a8288`

**FORBIDDEN colors:**
- NO red (#ff0000, #ef4444, #dc3545)
- NO green (#00ff00, #10b981, #28a745)
- NO yellow/orange (#ffff00, #f59e0b, #ffa500)
- NO blue (#0000ff, #3b82f6, #007bff)

### 2. DEVELOPMENT STANDARDS
- Dark theme ONLY
- Use Feather icons
- Follow preston-dev-setup standards
- No comments in code unless asked
- Always validate with color_validator.py

### 3. DATA STANDARDS
- 100% REAL DATA ONLY
- NO synthetic/fake data
- NO hardcoded values
- NO np.random for generating fake data
- Always cite data sources

### 4. TESTING STANDARDS
Before claiming any task is complete:
- Run color validation: `python3 src/validation/color_validator.py`
- Run standards enforcement: `python3 src/validation/standards_enforcer.py`
- Test the dashboard: `python3 src/dashboard/app_timeline.py`
- Verify all charts display correctly

### 5. CRITICAL REMINDERS
- Read standards BEFORE coding
- Check colors DURING coding  
- Validate AFTER coding
- Review designs with user BEFORE claiming completion

## Project Structure
```
cocoa-market-signals-v3/
├── src/
│   ├── data_pipeline/       # Real data fetchers
│   ├── dashboard/           # FastAPI dashboards
│   ├── models/              # ML models
│   ├── backtesting/         # Time-aware backtesting
│   └── validation/          # Standards enforcement
├── templates/               # HTML templates (dark theme)
├── data/                    # Real data storage
└── CLAUDE.md               # This file
```

## Key Commands
- Lint: `ruff check src/`
- Type check: `mypy src/`
- Run dashboard: `python3 src/dashboard/app_timeline.py`
- Validate colors: `python3 src/validation/color_validator.py`

## User Preferences
- Wants month-based navigation (not phases)
- Wants to see prediction accuracy at different time horizons
- Wants beautiful, dark-themed dashboards
- Expects 100% adherence to standards WITHOUT reminders

## Current Status
- Backtesting complete with real data
- Timeline dashboard working
- Need to complete month-based navigation
- Model performance needs improvement (currently 53.5%)

Remember: The user created comprehensive standards in preston-dev-setup. Follow them ALL without being asked!
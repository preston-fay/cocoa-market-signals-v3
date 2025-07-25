# Cocoa Market Signals v3 - Production System

## Overview

Version 3 combines the **real data and advanced analytics** from v1 with the **clean, modern dashboard** from v2, creating a production-ready market intelligence system with **ZERO synthetic data**.

## Core Principles

1. **NO FAKE DATA** - Every number is traceable to a real source
2. **NO SYNTHETIC RESULTS** - All calculations from actual data
3. **NO HARDCODED METRICS** - Everything dynamically calculated
4. **FULL TRANSPARENCY** - Clear documentation of data sources
5. **REPUTATION FIRST** - Better to show "No Data" than fake data

## Project Structure

```
cocoa-market-signals-v3/
├── src/
│   ├── data_pipeline/      # Unified data ingestion
│   ├── validation/         # Data quality checks
│   ├── models/            # Statistical & ML models
│   ├── backtesting/       # Historical validation
│   ├── dashboard/         # Enhanced v2 UI
│   └── api/              # REST endpoints
├── data/
│   ├── historical/        # Validated historical data only
│   ├── real-time/        # Live API connections
│   └── validated/        # Quality-checked data only
├── tests/
│   ├── data_integrity/    # Source verification
│   ├── statistical/       # Model validation
│   └── performance/       # Backtesting accuracy
└── docs/
    ├── methodology/       # Full documentation
    ├── data_sources/      # API guides
    └── validation/        # Test results
```

## Data Sources

### Verified Real Data
- **ICCO Historical Prices**: October 2023 - January 2024
- **UN Comtrade Export Data**: Trade volumes with API access
- **Weather Station Data**: Yamoussoukro and Kumasi historical records
- **Shipping Costs**: Real container and bulk rates
- **Economic Indicators**: Inflation, currency, actual country data

## Key Features

### Data Integrity
- Every data point has a traceable source
- Verification badges: ✓ Verified, ⚠️ Estimated, ✗ No Data
- Full audit trail for all calculations
- Downloadable raw data with sources

### Advanced Analytics
- **Granger Causality Tests**: Weather → Price relationships
- **Random Forest Models**: Signal generation
- **Isolation Forest**: Anomaly detection
- **Time Series Decomposition**: Trend analysis

### Professional Dashboard
- Clean Kearney-style design from v2
- Real-time data status indicators
- Statistical significance markers
- Confidence intervals on predictions

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd cocoa-market-signals-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API credentials
```

## Quick Start

```bash
# Run data validation
python -m src.validation.check_data_integrity

# Start the dashboard
python run_dashboard.py

# Access at http://localhost:5000
```

## Development Status

- [x] Project structure created
- [ ] Data migration from v1
- [ ] Dashboard integration from v2
- [ ] Data validation layer
- [ ] API connections
- [ ] Statistical models
- [ ] Production deployment

## Contributing

Please ensure all contributions maintain our data integrity principles. See CONTRIBUTING.md for guidelines.

## License

[License details]
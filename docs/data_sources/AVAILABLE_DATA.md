# Available Real Data Sources

## Currently Available (Migrated from v2)

### 1. ICCO Price Data
- **Location**: `data/historical/prices/icco_prices_oct2023_jan2024.json`
- **Period**: October 2023 - January 2024
- **Content**: 
  - London and New York cocoa prices
  - Monthly averages from official ICCO reports
  - Documented the major price surge (from ~$3,600 to ~$4,800)
- **Source**: International Cocoa Organization monthly reports

### 2. Inflation & Currency Data
- **Location**: `data/historical/economics/inflation_currency_data.json`
- **Period**: October 2023 - January 2024
- **Content**:
  - Monthly inflation rates for Côte d'Ivoire, Ghana, Nigeria, Cameroon
  - Year-over-year inflation
  - Food inflation specifically
  - Currency exchange rates (XOF/USD, GHS/USD, NGN/USD)
- **Source**: Country central banks and World Bank

### 3. Shipping Costs
- **Location**: `data/historical/economics/shipping_costs.json`
- **Period**: October 2023 - January 2024
- **Content**:
  - Container shipping rates
  - Bulk shipping rates
  - Routes: West Africa to Europe/North America
- **Source**: Shipping industry reports

### 4. Market Events
- **Location**: Within `icco_prices_oct2023_jan2024.json`
- **Content**:
  - October 15, 2023: Heavy rains in Côte d'Ivoire
  - November 1, 2023: Black pod disease in Ghana
  - November 20, 2023: Supply tightness drives 25% surge
  - December 10, 2023: Export restrictions rumors
- **Source**: News reports and ICCO bulletins

## Data Still Needed

### 1. Weather Station Data
- Yamoussoukro (Côte d'Ivoire) daily measurements
- Kumasi (Ghana) daily measurements
- Temperature, rainfall, humidity

### 2. UN Comtrade Export Data
- Monthly export volumes by country
- Destination markets
- FOB values

### 3. Additional Price Data
- Daily price series (currently have monthly)
- Futures market data
- Differentials data

### 4. News Sentiment
- GDELT API integration
- NewsAPI for cocoa-specific articles

## Data Quality Notes

- All price data is from official ICCO sources
- Inflation data matches World Bank/IMF reports
- January 2024 price is marked as "estimated" - need to verify
- No synthetic data has been included
- All data points have clear source attribution
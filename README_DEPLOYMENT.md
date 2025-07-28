# Cocoa Market Signals Dashboard - Deployment Guide

## Overview
This dashboard shows real-world cocoa market analysis with:
- 79.8% prediction accuracy using XGBoost
- 51 significant market events with detailed triggers
- Real data from Yahoo Finance, GDELT News, Open-Meteo Weather, UN Comtrade
- Interactive timeline with clickable events
- Full methodology and model performance metrics

## Quick Start (Local Sharing)

### Option 1: Ngrok (Easiest - 5 minutes)
```bash
# 1. Start the dashboard
python3 src/dashboard/app_comprehensive.py

# 2. In another terminal, expose it publicly
ngrok http 8002

# 3. Share the https URL with colleagues
```

### Option 2: Network Sharing
```bash
# 1. Find your IP address
ifconfig | grep inet  # Mac/Linux

# 2. Start dashboard on all interfaces
python3 src/dashboard/app_comprehensive.py

# 3. Share URL: http://YOUR_IP:8002
```

## Cloud Deployment Options

### Render.com (Recommended - Free tier available)
1. Push code to GitHub
2. Connect GitHub to Render.com
3. Create new Web Service
4. Use these settings:
   - Build Command: `pip install -r requirements_minimal.txt`
   - Start Command: `python src/dashboard/app_comprehensive.py`

### Railway.app (Simple & Fast)
1. Install Railway CLI: `brew install railwayapp/railway/railway`
2. Run: `railway login`
3. Run: `railway up`
4. Get URL: `railway open`

### Google Cloud Run
```bash
# 1. Build container
docker build -t cocoa-dashboard .

# 2. Tag for GCR
docker tag cocoa-dashboard gcr.io/YOUR_PROJECT/cocoa-dashboard

# 3. Push to GCR
docker push gcr.io/YOUR_PROJECT/cocoa-dashboard

# 4. Deploy
gcloud run deploy cocoa-dashboard \
  --image gcr.io/YOUR_PROJECT/cocoa-dashboard \
  --platform managed \
  --port 8002 \
  --allow-unauthenticated
```

### AWS EC2 (Traditional)
```bash
# 1. Launch EC2 instance (t2.micro for free tier)
# 2. SSH into instance
# 3. Clone repository
git clone https://github.com/yourusername/cocoa-market-signals-v3.git
cd cocoa-market-signals-v3

# 4. Install Python 3.11
sudo apt update
sudo apt install python3.11 python3-pip

# 5. Install dependencies
pip3 install -r requirements_minimal.txt

# 6. Run with screen/tmux
screen -S dashboard
python3 src/dashboard/app_comprehensive.py

# 7. Configure security group to allow port 8002
```

## Security Considerations

### Basic Authentication (Optional)
Add to `src/dashboard/app_comprehensive.py`:
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
security = HTTPBasic()

@app.get("/")
async def dashboard(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "cocoa" or credentials.password != "signals2024":
        raise HTTPException(status_code=401)
    # ... rest of code
```

### HTTPS (Recommended)
- Use Cloudflare tunnel for instant HTTPS
- Or use Let's Encrypt with nginx reverse proxy

## Sharing with Colleagues

### What to Share:
1. **Dashboard URL**: The deployed URL
2. **Key Highlights**:
   - 79.8% prediction accuracy on real cocoa futures
   - 502 days of price data analyzed
   - 1,769 news articles processed
   - 6,520 weather observations from 6 regions
   - 51 significant market events detected

3. **How to Use**:
   - Click purple dots on timeline to see event details
   - Check Overview tab for key findings
   - Review Methodology for data sources
   - See Model Performance for accuracy metrics

### Demo Script:
"This dashboard demonstrates our ML model's ability to predict cocoa market movements using real-world data. We achieved 79.8% directional accuracy by combining price data, weather anomalies, news sentiment, and trade volumes. Click any purple dot on the timeline to explore what triggered each significant market movement."

## Troubleshooting

### Port Already in Use
```bash
lsof -i :8002
kill -9 [PID]
```

### Missing Data Files
Ensure these files exist:
- `data/cocoa_market_signals_real.db`
- `data/processed/real_dashboard_data.json`
- `data/processed/detailed_events.json`

### Performance Issues
- Use `requirements_minimal.txt` instead of full requirements
- Consider caching with Redis for production
- Use CDN for static assets

## Support
For issues or questions about the real data and methodology, refer to the dashboard's Methodology tab which explains all data sources and processing.
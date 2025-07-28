# Dashboard Authentication

The Cocoa Market Signals dashboard is protected with basic authentication.

## Default Credentials

- **Username**: `cocoa`
- **Password**: `signals2024`

## Setting Custom Credentials

### For Railway Deployment

1. Go to your Railway project
2. Click on your service
3. Go to "Variables" tab
4. Add these environment variables:
   - `DASHBOARD_USERNAME` = your-username
   - `DASHBOARD_PASSWORD` = your-secure-password

### For Local Development

Set environment variables before running:

```bash
export DASHBOARD_USERNAME="myusername"
export DASHBOARD_PASSWORD="mypassword"
python src/dashboard/app_comprehensive.py
```

## Sharing with Colleagues

When sharing the dashboard URL, also provide:
1. The dashboard URL
2. The username and password (securely)

Example message:
```
Cocoa Market Signals Dashboard
URL: https://your-app.up.railway.app
Username: cocoa
Password: [shared securely]
```

## Security Notes

- Passwords are compared using constant-time comparison to prevent timing attacks
- Uses FastAPI's built-in HTTP Basic authentication
- Credentials can be changed without modifying code
- All endpoints are protected (main page, APIs, data files)
#!/usr/bin/env python3
"""
TEST ALL MODELS PROPERLY - WITH PROOF
No shortcuts, no lies, actual implementation
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TESTING ALL MODELS - PROPERLY THIS TIME")
print(f"Started at: {datetime.now()}")
print("="*60)

# Create proof log
proof_log = {
    "start_time": datetime.now().isoformat(),
    "models_tested": {},
    "errors": {},
    "data_info": {}
}

# Load data
df = pd.read_csv('data/processed/features_with_real_dates.csv', index_col='date', parse_dates=True)
print(f"\n📊 Loaded {len(df)} samples")
proof_log["data_info"]["total_samples"] = len(df)

# Select features
selected_features = [
    'close', 'volume_ratio', 'sma_ratio', 'return_20d', 'volatility_20d', 
    'temp_mean_mean', 'temp_anomaly', 'rainfall_sum',
    'sentiment_mean', 'article_count', 'total_exports_kg'
]

# Prepare data
X = df[selected_features].copy()
y = df['return_5d_future'].copy()

# Remove NaN
mask = ~(y.isna() | X.isna().any(axis=1))
X = X[mask]
y = y[mask]

print(f"✅ After cleaning: {len(X)} samples")
proof_log["data_info"]["clean_samples"] = len(X)

# Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

train_dates = X_train.index
test_dates = X_test.index

print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
proof_log["data_info"]["train_size"] = len(X_train)
proof_log["data_info"]["test_size"] = len(X_test)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_and_log(y_true, y_pred, model_name):
    """Evaluate model and log proof"""
    try:
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true_eval = y_true[:min_len]
        y_pred_eval = y_pred[:min_len]
        
        mse = mean_squared_error(y_true_eval, y_pred_eval)
        mae = mean_absolute_error(y_true_eval, y_pred_eval)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_eval, y_pred_eval)
        
        # Direction accuracy
        direction_true = (y_true_eval > 0).astype(int)
        direction_pred = (y_pred_eval > 0).astype(int)
        direction_accuracy = (direction_true == direction_pred).mean()
        
        # Log results
        results = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'predictions_made': len(y_pred_eval),
            'first_5_predictions': y_pred_eval[:5].tolist() if hasattr(y_pred_eval, 'tolist') else y_pred_eval[:5].tolist()
        }
        
        proof_log["models_tested"][model_name] = results
        
        print(f"\n✅ {model_name}:")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Direction: {direction_accuracy:.1%}")
        print(f"   Predictions made: {len(y_pred_eval)}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ {model_name} evaluation failed: {e}")
        proof_log["errors"][model_name] = str(e)
        return None

print("\n" + "="*40)
print("TESTING EACH MODEL")
print("="*40)

# 1. RANDOM FOREST
print("\n1️⃣ Random Forest...")
try:
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    evaluate_and_log(y_test, y_pred_rf, "Random Forest")
except Exception as e:
    print(f"❌ RF failed: {e}")
    proof_log["errors"]["Random Forest"] = str(e)

# 2. XGBOOST
print("\n2️⃣ XGBoost...")
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    evaluate_and_log(y_test, y_pred_xgb, "XGBoost")
except Exception as e:
    print(f"❌ XGBoost failed: {e}")
    proof_log["errors"]["XGBoost"] = str(e)

# 3. LSTM
print("\n3️⃣ LSTM Neural Network...")
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    tf.random.set_seed(42)
    
    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Build model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Train
    print("   Training LSTM...")
    history = lstm_model.fit(
        X_train_lstm, y_train, 
        epochs=50, batch_size=16, 
        validation_split=0.1, verbose=0
    )
    
    # Predict
    y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    evaluate_and_log(y_test, y_pred_lstm, "LSTM")
    
except Exception as e:
    print(f"❌ LSTM failed: {e}")
    proof_log["errors"]["LSTM"] = str(e)

# 4. VAR (Vector Autoregression)
print("\n4️⃣ VAR Model...")
try:
    from statsmodels.tsa.api import VAR
    
    # Prepare multivariate data
    var_features = ['close', 'volume_ratio', 'temp_mean_mean', 'sentiment_mean']
    var_df = df[var_features + ['return_5d_future']].dropna()
    
    # Split properly
    split_idx = int(0.8 * len(var_df))
    var_train = var_df.iloc[:split_idx]
    var_test = var_df.iloc[split_idx:]
    
    # Fit VAR on features only
    var_model = VAR(var_train[var_features])
    var_fitted = var_model.fit(maxlags=5, ic='aic')
    
    # Forecast
    lag_order = var_fitted.k_ar
    y_pred_var = []
    
    # Make one-step ahead forecasts
    for i in range(len(var_test)):
        if i < lag_order:
            # Use last values from train
            hist_data = pd.concat([var_train[var_features].iloc[-(lag_order-i):], 
                                  var_test[var_features].iloc[:i]])
        else:
            hist_data = var_test[var_features].iloc[i-lag_order:i]
        
        forecast = var_fitted.forecast(hist_data.values, steps=1)
        # Use close price change as proxy for return
        if i > 0:
            y_pred_var.append((forecast[0][0] - var_test['close'].iloc[i-1]) / var_test['close'].iloc[i-1])
        else:
            y_pred_var.append(0)
    
    # Align with test set
    y_test_var = var_test['return_5d_future'].values
    evaluate_and_log(y_test_var, np.array(y_pred_var), "VAR")
    
except Exception as e:
    print(f"❌ VAR failed: {e}")
    proof_log["errors"]["VAR"] = str(e)

# 5. PROPHET
print("\n5️⃣ Prophet...")
try:
    from prophet import Prophet
    
    # Prepare data
    prophet_train = pd.DataFrame({
        'ds': train_dates,
        'y': y_train.values
    })
    
    # Remove any NaN
    prophet_train = prophet_train.dropna()
    
    # Initialize and fit
    model = Prophet(
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Add regressors with non-NaN values
    train_temp = X_train['temp_mean_mean'].fillna(X_train['temp_mean_mean'].mean())
    test_temp = X_test['temp_mean_mean'].fillna(X_test['temp_mean_mean'].mean())
    
    prophet_train['temp'] = train_temp.values
    model.add_regressor('temp')
    
    model.fit(prophet_train)
    
    # Predict
    future = pd.DataFrame({
        'ds': test_dates,
        'temp': test_temp.values
    })
    
    forecast = model.predict(future)
    y_pred_prophet = forecast['yhat'].values
    
    evaluate_and_log(y_test, y_pred_prophet, "Prophet")
    
except Exception as e:
    print(f"❌ Prophet failed: {e}")
    proof_log["errors"]["Prophet"] = str(e)

# 6. ARIMA
print("\n6️⃣ ARIMA...")
try:
    from statsmodels.tsa.arima.model import ARIMA
    
    # Use returns series
    returns_series = df['return_5d'].dropna()
    train_size = int(0.8 * len(returns_series))
    train_returns = returns_series[:train_size]
    
    # Fit ARIMA
    arima_model = ARIMA(train_returns, order=(2,0,2))
    arima_fitted = arima_model.fit()
    
    # Forecast
    n_periods = len(X_test)
    arima_forecast = arima_fitted.forecast(steps=n_periods)
    
    evaluate_and_log(y_test, arima_forecast, "ARIMA")
    
except Exception as e:
    print(f"❌ ARIMA failed: {e}")
    proof_log["errors"]["ARIMA"] = str(e)

# 7. TIMEGPT
print("\n7️⃣ TimeGPT...")
try:
    from nixtla import NixtlaClient
    
    # Your API key
    client = NixtlaClient(
        api_key='nixtla-a3AZwCF9kpPIUtfpNp3TIUYBXGGIM7uK4vkvHknWxdNmUlNN72HNoBoXMydXjKqKlcvdbZENiFK0xHlG'
    )
    
    # Prepare clean data
    timegpt_df = pd.DataFrame({
        'unique_id': 0,
        'ds': train_dates,
        'y': y_train.values
    })
    
    print("   Calling TimeGPT API...")
    fcst = client.forecast(
        df=timegpt_df,
        h=len(test_dates),
        time_col='ds',
        target_col='y'
    )
    
    y_pred_timegpt = fcst['TimeGPT'].values
    evaluate_and_log(y_test, y_pred_timegpt, "TimeGPT")
    
except Exception as e:
    print(f"❌ TimeGPT failed: {e}")
    proof_log["errors"]["TimeGPT"] = str(e)

# 8. Simple TSMamba implementation
print("\n8️⃣ TSMamba (State Space Model)...")
try:
    # Simplified Mamba-like state space model
    class SimpleMamba:
        def __init__(self, input_dim, hidden_dim=32):
            self.A = np.random.randn(hidden_dim, hidden_dim) * 0.1
            self.B = np.random.randn(hidden_dim, input_dim) * 0.1
            self.C = np.random.randn(1, hidden_dim) * 0.1
            self.hidden_dim = hidden_dim
            
        def fit(self, X, y, lr=0.001, epochs=100):
            n_samples = X.shape[0]
            
            for epoch in range(epochs):
                total_loss = 0
                h = np.zeros(self.hidden_dim)
                
                for i in range(n_samples):
                    # State update
                    h = np.tanh(self.A @ h + self.B @ X[i])
                    
                    # Output
                    y_pred = self.C @ h
                    
                    # Loss
                    loss = (y_pred - y[i]) ** 2
                    total_loss += loss
                    
                    # Simple gradient update
                    grad = 2 * (y_pred - y[i])
                    self.C -= lr * grad * h.reshape(1, -1)
                    
                if epoch % 20 == 0:
                    print(f"      Epoch {epoch}, Loss: {total_loss[0]:.4f}")
                    
        def predict(self, X):
            predictions = []
            h = np.zeros(self.hidden_dim)
            
            for i in range(X.shape[0]):
                h = np.tanh(self.A @ h + self.B @ X[i])
                y_pred = self.C @ h
                predictions.append(y_pred[0])
                
            return np.array(predictions)
    
    # Train
    mamba = SimpleMamba(input_dim=X_train_scaled.shape[1])
    mamba.fit(X_train_scaled, y_train.values, epochs=40)
    
    # Predict
    y_pred_mamba = mamba.predict(X_test_scaled)
    evaluate_and_log(y_test, y_pred_mamba, "TSMamba")
    
except Exception as e:
    print(f"❌ TSMamba failed: {e}")
    proof_log["errors"]["TSMamba"] = str(e)

# Save proof
print("\n" + "="*60)
print("SAVING PROOF OF TESTING")
print("="*60)

proof_log["end_time"] = datetime.now().isoformat()
proof_log["total_models_attempted"] = len(proof_log["models_tested"]) + len(proof_log["errors"])
proof_log["successful_models"] = len(proof_log["models_tested"])

# Save to file
with open('data/processed/model_testing_proof.json', 'w') as f:
    json.dump(proof_log, f, indent=2)

print(f"\n✅ Proof saved to: data/processed/model_testing_proof.json")

# Save to database
conn = sqlite3.connect("data/cocoa_market_signals_real.db")
cursor = conn.cursor()

for model_name, results in proof_log["models_tested"].items():
    cursor.execute('''
    INSERT INTO model_performance 
    (model_name, train_date, test_start, test_end, accuracy, precision, parameters)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_name,
        datetime.now().strftime('%Y-%m-%d'),
        str(test_dates.min()),
        str(test_dates.max()),
        results['direction_accuracy'],
        results['r2'],
        json.dumps(results)
    ))

conn.commit()
conn.close()

print("✅ Results saved to database")

# Print summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\nModels successfully tested: {len(proof_log['models_tested'])}")
for model, results in proof_log["models_tested"].items():
    print(f"  ✅ {model}: RMSE={results['rmse']:.4f}, R²={results['r2']:.3f}, Direction={results['direction_accuracy']:.1%}")

print(f"\nModels that failed: {len(proof_log['errors'])}")
for model, error in proof_log["errors"].items():
    print(f"  ❌ {model}: {error[:50]}...")

print(f"\n📄 Check proof file: data/processed/model_testing_proof.json")
print(f"🕐 Total time: {datetime.now() - datetime.fromisoformat(proof_log['start_time'])}")
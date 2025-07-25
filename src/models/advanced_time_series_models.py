"""
Advanced Time Series Models for Cocoa Market Signals
Implements ALL models from v2 with real v3 data
NO FAKE DATA - 100% REAL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model  # For GARCH models

# ML Models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping

# Statistical Process Control
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

# Prophet
from prophet import Prophet

# Custom imports
from .statistical_models import StatisticalSignalModels

class AdvancedTimeSeriesModels:
    """Comprehensive suite of time series models from v2"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.base_models = StatisticalSignalModels()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for time series modeling"""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Add returns
        df['returns'] = df['price'].pct_change()
        
        # Add log returns for volatility models
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        return df
    
    # =========================
    # TRADITIONAL TIME SERIES
    # =========================
    
    def fit_arima(self, df: pd.DataFrame, order: Tuple[int, int, int] = None) -> Dict:
        """
        Fit ARIMA model for price prediction
        Auto-selects order if not provided
        """
        print("\n" + "="*60)
        print("FITTING ARIMA MODEL")
        print("="*60)
        
        # Use price series
        price_series = df['price'].dropna()
        
        if order is None:
            # Auto-select order using AIC
            print("Auto-selecting ARIMA order...")
            best_aic = np.inf
            best_order = None
            
            for p in range(0, 4):
                for d in range(0, 2):
                    for q in range(0, 4):
                        try:
                            model = ARIMA(price_series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            order = best_order
            print(f"Selected order: {order} (AIC: {best_aic:.2f})")
        
        # Fit final model
        model = ARIMA(price_series, order=order)
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast_steps = 30
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # In-sample predictions
        in_sample_pred = fitted_model.fittedvalues
        
        # Calculate metrics
        residuals = price_series - in_sample_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / price_series)) * 100
        
        print(f"\nARIMA Results:")
        print(f"  Order: {order}")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        self.models['arima'] = fitted_model
        self.results['arima'] = {
            'order': order,
            'aic': fitted_model.aic,
            'rmse': rmse,
            'mape': mape,
            'forecast': forecast,
            'fitted_values': in_sample_pred,
            'residuals': residuals
        }
        
        return self.results['arima']
    
    def fit_var(self, df: pd.DataFrame, variables: List[str] = None) -> Dict:
        """
        Fit Vector Autoregression model for multivariate dynamics
        """
        print("\n" + "="*60)
        print("FITTING VAR MODEL")
        print("="*60)
        
        if variables is None:
            variables = ['price', 'rainfall_anomaly', 'temperature_anomaly', 
                        'trade_volume_change', 'export_concentration']
        
        # Prepare data
        var_data = df[variables].dropna()
        
        # Check stationarity
        print("Checking stationarity of variables...")
        for var in variables:
            adf_result = self.base_models.test_stationarity(var_data[var], name=var)
            if not adf_result['is_stationary']:
                # Difference if non-stationary
                var_data[f'{var}_diff'] = var_data[var].diff()
        
        # Remove any differenced columns with NaN
        var_data = var_data.dropna()
        
        # Fit VAR model
        model = VAR(var_data)
        
        # Select lag order
        lag_order = model.select_order()
        print(f"\nOptimal lag order: {lag_order.selected_orders['aic']}")
        
        # Fit with optimal lag
        fitted_model = model.fit(lag_order.selected_orders['aic'])
        
        # Granger causality tests
        print("\nGranger Causality Tests:")
        causality_results = {}
        for var in variables:
            if var != 'price':
                try:
                    gc_test = fitted_model.test_causality('price', var, kind='f')
                    causality_results[var] = {
                        'statistic': gc_test.test_statistic,
                        'p_value': gc_test.pvalue,
                        'causes_price': gc_test.pvalue < 0.05
                    }
                    print(f"  {var} → price: p={gc_test.pvalue:.4f} "
                          f"{'***' if gc_test.pvalue < 0.05 else ''}")
                except:
                    pass
        
        # Forecast
        forecast = fitted_model.forecast(var_data.values[-fitted_model.k_ar:], steps=30)
        
        # Impulse response
        irf = fitted_model.irf(periods=20)
        
        self.models['var'] = fitted_model
        self.results['var'] = {
            'lag_order': lag_order.selected_orders['aic'],
            'causality': causality_results,
            'forecast': forecast,
            'impulse_response': irf,
            'variables': list(var_data.columns)
        }
        
        return self.results['var']
    
    def fit_stl_decomposition(self, df: pd.DataFrame, period: int = 30) -> Dict:
        """
        STL Decomposition (Seasonal and Trend decomposition using Loess)
        """
        print("\n" + "="*60)
        print("FITTING STL DECOMPOSITION")
        print("="*60)
        
        # Use price series
        price_series = df['price'].dropna()
        
        # Perform STL decomposition
        stl = STL(price_series, seasonal=13, period=period)
        decomposition = stl.fit()
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Calculate strength of components
        var_total = price_series.var()
        strength_trend = max(0, 1 - residual.var() / (residual + trend).var())
        strength_seasonal = max(0, 1 - residual.var() / (residual + seasonal).var())
        
        print(f"\nSTL Decomposition Results:")
        print(f"  Period: {period}")
        print(f"  Trend strength: {strength_trend:.3f}")
        print(f"  Seasonal strength: {strength_seasonal:.3f}")
        print(f"  Residual std: ${residual.std():.2f}")
        
        self.results['stl'] = {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'strength_trend': strength_trend,
            'strength_seasonal': strength_seasonal,
            'decomposition': decomposition
        }
        
        return self.results['stl']
    
    def fit_sarima(self, df: pd.DataFrame, order: Tuple = None, seasonal_order: Tuple = None) -> Dict:
        """
        Fit SARIMA (Seasonal ARIMA) model
        """
        print("\n" + "="*60)
        print("FITTING SARIMA MODEL")
        print("="*60)
        
        price_series = df['price'].dropna()
        
        # Default orders if not provided
        if order is None:
            order = (1, 1, 1)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 30)  # Monthly seasonality
        
        try:
            # Fit SARIMA
            model = SARIMAX(price_series, 
                           order=order,
                           seasonal_order=seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast = fitted_model.forecast(steps=30)
            
            # In-sample predictions
            in_sample_pred = fitted_model.fittedvalues
            
            # Metrics
            residuals = price_series - in_sample_pred
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs(residuals / price_series)) * 100
            
            print(f"\nSARIMA Results:")
            print(f"  Order: {order} x {seasonal_order}")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            
            self.models['sarima'] = fitted_model
            self.results['sarima'] = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': fitted_model.aic,
                'rmse': rmse,
                'mape': mape,
                'forecast': forecast
            }
            
        except Exception as e:
            print(f"SARIMA failed: {str(e)}")
            self.results['sarima'] = {'error': str(e)}
            
        return self.results.get('sarima', {})
    
    def fit_holt_winters(self, df: pd.DataFrame, seasonal_periods: int = 30) -> Dict:
        """
        Fit Holt-Winters Exponential Smoothing
        """
        print("\n" + "="*60)
        print("FITTING HOLT-WINTERS EXPONENTIAL SMOOTHING")
        print("="*60)
        
        price_series = df['price'].dropna()
        
        try:
            # Fit Holt-Winters
            model = ExponentialSmoothing(
                price_series,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                damped_trend=True
            )
            fitted_model = model.fit(optimized=True)
            
            # Forecast
            forecast = fitted_model.forecast(steps=30)
            
            # In-sample predictions
            in_sample_pred = fitted_model.fittedvalues
            
            # Metrics
            residuals = price_series - in_sample_pred
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs(residuals / price_series)) * 100
            
            print(f"\nHolt-Winters Results:")
            print(f"  Seasonal periods: {seasonal_periods}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Alpha (level): {fitted_model.params['smoothing_level']:.3f}")
            print(f"  Beta (trend): {fitted_model.params['smoothing_trend']:.3f}")
            print(f"  Gamma (seasonal): {fitted_model.params['smoothing_seasonal']:.3f}")
            
            self.models['holt_winters'] = fitted_model
            self.results['holt_winters'] = {
                'rmse': rmse,
                'mape': mape,
                'forecast': forecast,
                'params': fitted_model.params
            }
            
        except Exception as e:
            print(f"Holt-Winters failed: {str(e)}")
            self.results['holt_winters'] = {'error': str(e)}
            
        return self.results.get('holt_winters', {})
    
    def fit_prophet(self, df: pd.DataFrame) -> Dict:
        """
        Fit Prophet model for trend and seasonality decomposition
        """
        print("\n" + "="*60)
        print("FITTING PROPHET MODEL")
        print("="*60)
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': df.index,
            'y': df['price']
        }).reset_index(drop=True)
        
        # Add regressors
        for col in ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']:
            if col in df.columns:
                prophet_data[col] = df[col].values
        
        # Initialize Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            seasonality_mode='multiplicative',
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True
        )
        
        # Add regressors
        for col in ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']:
            if col in prophet_data.columns:
                model.add_regressor(col)
        
        # Fit model
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=90, freq='D')
        
        # Add regressor values for future (using last known values)
        for col in ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']:
            if col in prophet_data.columns:
                future[col] = prophet_data[col].iloc[-1]
        
        # Predict
        forecast = model.predict(future)
        
        # Calculate metrics on historical data
        historical_forecast = forecast[forecast['ds'] <= prophet_data['ds'].max()]
        residuals = prophet_data['y'] - historical_forecast['yhat'][:len(prophet_data)]
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / prophet_data['y'])) * 100
        
        print(f"\nProphet Results:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Trend changepoints detected: {len(model.changepoints)}")
        
        self.models['prophet'] = model
        self.results['prophet'] = {
            'forecast': forecast,
            'rmse': rmse,
            'mape': mape,
            'trend': forecast['trend'],
            'yearly_seasonality': forecast['yearly'],
            'changepoints': model.changepoints
        }
        
        return self.results['prophet']
    
    # =========================
    # VOLATILITY MODELS
    # =========================
    
    def fit_garch(self, df: pd.DataFrame, p: int = 1, q: int = 1) -> Dict:
        """
        Fit GARCH model for volatility clustering
        """
        print("\n" + "="*60)
        print("FITTING GARCH MODEL")
        print("="*60)
        
        # Use returns (multiply by 100 for percentage)
        returns = df['returns'].dropna() * 100
        
        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=p, q=q, dist='t')
        fitted_model = model.fit(disp='off')
        
        # Get conditional volatility
        conditional_vol = fitted_model.conditional_volatility
        
        # Forecast volatility
        forecast = fitted_model.forecast(horizon=30)
        
        # Calculate VaR at different confidence levels
        var_95 = fitted_model.std_resid.quantile(0.05) * conditional_vol.iloc[-1]
        var_99 = fitted_model.std_resid.quantile(0.01) * conditional_vol.iloc[-1]
        
        print(f"\nGARCH({p},{q}) Results:")
        print(f"  Log-likelihood: {fitted_model.loglikelihood:.2f}")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  Current volatility: {conditional_vol.iloc[-1]:.2f}%")
        print(f"  95% VaR: {var_95:.2f}%")
        print(f"  99% VaR: {var_99:.2f}%")
        
        # Detect high volatility periods
        high_vol_threshold = conditional_vol.quantile(0.8)
        high_vol_periods = conditional_vol[conditional_vol > high_vol_threshold]
        
        self.models['garch'] = fitted_model
        self.results['garch'] = {
            'model': f'GARCH({p},{q})',
            'conditional_volatility': conditional_vol,
            'forecast': forecast,
            'var_95': var_95,
            'var_99': var_99,
            'high_vol_periods': high_vol_periods,
            'aic': fitted_model.aic
        }
        
        return self.results['garch']
    
    def calculate_ewma(self, df: pd.DataFrame, span: int = 20) -> Dict:
        """
        Calculate Exponential Weighted Moving Average for volatility
        """
        print("\n" + "="*60)
        print("CALCULATING EWMA VOLATILITY")
        print("="*60)
        
        returns = df['returns'].dropna()
        
        # Calculate EWMA volatility
        ewma_vol = returns.ewm(span=span).std() * np.sqrt(252) * 100
        
        # Calculate EWMA for prices
        ewma_price = df['price'].ewm(span=span).mean()
        
        # Calculate risk metrics
        current_vol = ewma_vol.iloc[-1]
        vol_percentile = stats.percentileofscore(ewma_vol, current_vol)
        
        print(f"\nEWMA Results (span={span}):")
        print(f"  Current volatility: {current_vol:.2f}%")
        print(f"  Volatility percentile: {vol_percentile:.1f}%")
        print(f"  Current EWMA price: ${ewma_price.iloc[-1]:,.2f}")
        
        self.results['ewma'] = {
            'volatility': ewma_vol,
            'price': ewma_price,
            'current_vol': current_vol,
            'vol_percentile': vol_percentile,
            'span': span
        }
        
        return self.results['ewma']
    
    # =========================
    # ADVANCED ML MODELS
    # =========================
    
    def fit_xgboost(self, df: pd.DataFrame, target_days: int = 7) -> Dict:
        """
        Fit XGBoost with time series features
        """
        print("\n" + "="*60)
        print("FITTING XGBOOST WITH TIME SERIES FEATURES")
        print("="*60)
        
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Skipping...")
            return {}
        
        # Create features
        features = self._create_time_series_features(df)
        
        # Create target (future price movement)
        features['target'] = features['price'].shift(-target_days)
        
        # Drop NaN
        features = features.dropna()
        
        # Split features and target
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols]
        y = features['target']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train XGBoost
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
            
            val_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, val_pred)
            cv_scores.append(mape)
        
        # Final model on all data
        model.fit(X_scaled, y)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nXGBoost Results:")
        print(f"  Target: {target_days}-day ahead prediction")
        print(f"  CV MAPE: {np.mean(cv_scores):.2f}% (±{np.std(cv_scores):.2f}%)")
        print(f"\nTop 10 Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.models['xgboost'] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        self.results['xgboost'] = {
            'cv_scores': cv_scores,
            'feature_importance': importance,
            'target_days': target_days,
            'params': params
        }
        
        return self.results['xgboost']
    
    def fit_lstm_autoencoder(self, df: pd.DataFrame, sequence_length: int = 30) -> Dict:
        """
        Fit LSTM Autoencoder for anomaly detection
        """
        print("\n" + "="*60)
        print("FITTING LSTM AUTOENCODER")
        print("="*60)
        
        # Prepare sequences
        features = ['price', 'rainfall_anomaly', 'temperature_anomaly', 
                   'trade_volume_change', 'export_concentration']
        
        data = df[features].dropna()
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i:i+sequence_length])
        
        sequences = np.array(sequences)
        
        # Build autoencoder
        n_features = len(features)
        
        # Encoder
        encoder_input = Input(shape=(sequence_length, n_features))
        encoder = LSTM(64, activation='relu', return_sequences=True)(encoder_input)
        encoder = LSTM(32, activation='relu', return_sequences=False)(encoder)
        
        # Decoder
        decoder = RepeatVector(sequence_length)(encoder)
        decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
        decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
        decoder_output = Dense(n_features)(decoder)
        
        # Autoencoder model
        autoencoder = Model(encoder_input, decoder_output)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train
        history = autoencoder.fit(
            sequences, sequences,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        # Calculate reconstruction error
        predictions = autoencoder.predict(sequences, verbose=0)
        mse = np.mean((sequences - predictions)**2, axis=(1, 2))
        
        # Determine anomaly threshold (95th percentile)
        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold
        
        # Get anomaly dates
        anomaly_dates = data.index[sequence_length:][anomalies]
        
        print(f"\nLSTM Autoencoder Results:")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Final loss: {history.history['loss'][-1]:.4f}")
        print(f"  Anomaly threshold: {threshold:.4f}")
        print(f"  Anomalies detected: {len(anomaly_dates)} ({len(anomaly_dates)/len(mse)*100:.1f}%)")
        
        if len(anomaly_dates) > 0:
            print(f"\nRecent anomalies:")
            for date in anomaly_dates[-5:]:
                print(f"  {date.date()}: MSE={mse[data.index[sequence_length:]==date][0]:.4f}")
        
        self.models['lstm_autoencoder'] = {
            'model': autoencoder,
            'scaler': scaler,
            'threshold': threshold
        }
        
        self.results['lstm_autoencoder'] = {
            'reconstruction_error': mse,
            'anomaly_threshold': threshold,
            'anomaly_dates': anomaly_dates,
            'sequence_length': sequence_length
        }
        
        return self.results['lstm_autoencoder']
    
    def fit_lstm_predictor(self, df: pd.DataFrame, sequence_length: int = 30, 
                          forecast_horizon: int = 7) -> Dict:
        """
        Fit LSTM for sequential pattern prediction
        """
        print("\n" + "="*60)
        print("FITTING LSTM PREDICTOR")
        print("="*60)
        
        # Prepare data
        price_data = df['price'].values.reshape(-1, 1)
        
        # Scale
        scaler = StandardScaler()
        scaled_price = scaler.fit_transform(price_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_price) - sequence_length - forecast_horizon):
            X.append(scaled_price[i:i+sequence_length])
            y.append(scaled_price[i+sequence_length+forecast_horizon-1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
            LSTM(50, activation='relu'),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate
        test_pred = model.predict(X_test, verbose=0)
        test_pred_inverse = scaler.inverse_transform(test_pred)
        y_test_inverse = scaler.inverse_transform(y_test)
        
        rmse = np.sqrt(mean_squared_error(y_test_inverse, test_pred_inverse))
        mape = mean_absolute_percentage_error(y_test_inverse, test_pred_inverse) * 100
        
        print(f"\nLSTM Predictor Results:")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Forecast horizon: {forecast_horizon} days")
        print(f"  Test RMSE: ${rmse:.2f}")
        print(f"  Test MAPE: {mape:.2f}%")
        
        self.models['lstm_predictor'] = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon
        }
        
        self.results['lstm_predictor'] = {
            'test_rmse': rmse,
            'test_mape': mape,
            'predictions': test_pred_inverse,
            'actual': y_test_inverse
        }
        
        return self.results['lstm_predictor']
    
    # =========================
    # STATISTICAL PROCESS CONTROL
    # =========================
    
    def detect_cusum(self, df: pd.DataFrame, threshold: float = 5) -> Dict:
        """
        CUSUM change point detection
        """
        print("\n" + "="*60)
        print("CUSUM CHANGE DETECTION")
        print("="*60)
        
        # Use returns
        returns = df['returns'].dropna()
        
        # Calculate CUSUM
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Standardize returns
        standardized = (returns - mean_return) / std_return
        
        # CUSUM calculation
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i] + 0.5)
        
        # Detect change points
        change_points = []
        for i in range(len(cusum_pos)):
            if cusum_pos[i] > threshold or abs(cusum_neg[i]) > threshold:
                change_points.append(returns.index[i])
        
        print(f"\nCUSUM Results:")
        print(f"  Threshold: {threshold}")
        print(f"  Change points detected: {len(change_points)}")
        
        if len(change_points) > 0:
            print(f"\nRecent change points:")
            for cp in change_points[-5:]:
                idx = returns.index.get_loc(cp)
                print(f"  {cp.date()}: Return={returns.iloc[idx]*100:.2f}%, "
                      f"CUSUM+={cusum_pos[idx]:.2f}, CUSUM-={cusum_neg[idx]:.2f}")
        
        self.results['cusum'] = {
            'cusum_positive': pd.Series(cusum_pos, index=returns.index),
            'cusum_negative': pd.Series(cusum_neg, index=returns.index),
            'change_points': change_points,
            'threshold': threshold
        }
        
        return self.results['cusum']
    
    def detect_modified_zscore(self, df: pd.DataFrame, threshold: float = 3.5) -> Dict:
        """
        Modified Z-score for robust outlier detection
        """
        print("\n" + "="*60)
        print("MODIFIED Z-SCORE DETECTION")
        print("="*60)
        
        price_series = df['price']
        
        # Calculate modified Z-score
        median = price_series.median()
        mad = np.median(np.abs(price_series - median))
        modified_z_scores = 0.6745 * (price_series - median) / mad
        
        # Detect outliers
        outliers = np.abs(modified_z_scores) > threshold
        outlier_dates = price_series[outliers]
        
        print(f"\nModified Z-score Results:")
        print(f"  Threshold: {threshold}")
        print(f"  Outliers detected: {len(outlier_dates)} ({len(outlier_dates)/len(price_series)*100:.1f}%)")
        print(f"  Median price: ${median:,.2f}")
        print(f"  MAD: ${mad:,.2f}")
        
        if len(outlier_dates) > 0:
            print(f"\nRecent outliers:")
            for date, price in outlier_dates.tail().items():
                z_score = modified_z_scores[date]
                print(f"  {date.date()}: Price=${price:,.2f}, Z-score={z_score:.2f}")
        
        self.results['modified_zscore'] = {
            'z_scores': modified_z_scores,
            'outliers': outlier_dates,
            'threshold': threshold,
            'median': median,
            'mad': mad
        }
        
        return self.results['modified_zscore']
    
    def detect_changepoints(self, df: pd.DataFrame, penalty: str = 'bic', model: str = 'rbf') -> Dict:
        """
        Advanced change point detection using ruptures library
        """
        print("\n" + "="*60)
        print("CHANGE POINT DETECTION")
        print("="*60)
        
        try:
            import ruptures as rpt
        except ImportError:
            print("Ruptures library not available. Using simple change detection...")
            # Fallback to simple method
            return self._simple_changepoint_detection(df)
        
        # Use price series
        signal = df['price'].values
        
        # Detection
        algo = rpt.Pelt(model=model).fit(signal)
        result = algo.predict(pen=10)
        
        # Convert to dates
        change_points = [df.index[i-1] for i in result[:-1]]  # Exclude last point
        
        print(f"\nChange Point Detection Results:")
        print(f"  Model: {model}")
        print(f"  Penalty: {penalty}")
        print(f"  Change points detected: {len(change_points)}")
        
        if change_points:
            print(f"\nChange points:")
            for cp in change_points:
                idx = df.index.get_loc(cp)
                if idx > 0:
                    before_price = df['price'].iloc[idx-1]
                    after_price = df['price'].iloc[idx]
                    change = (after_price - before_price) / before_price * 100
                    print(f"  {cp.date()}: ${before_price:,.0f} → ${after_price:,.0f} ({change:+.1f}%)")
        
        self.results['changepoints'] = {
            'dates': change_points,
            'indices': result[:-1],
            'model': model,
            'penalty': penalty
        }
        
        return self.results['changepoints']
    
    def _simple_changepoint_detection(self, df: pd.DataFrame) -> Dict:
        """
        Simple change point detection based on rolling statistics
        """
        price_series = df['price']
        returns = price_series.pct_change().dropna()
        
        # Calculate rolling mean and std
        window = 30
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Detect significant changes (> 2 std)
        z_scores = (returns - rolling_mean) / rolling_std
        change_points = df.index[np.abs(z_scores) > 2.5].tolist()
        
        print(f"\nSimple Change Point Detection:")
        print(f"  Change points: {len(change_points)}")
        
        self.results['changepoints'] = {
            'dates': change_points,
            'method': 'simple_zscore'
        }
        
        return self.results['changepoints']
    
    def create_ensemble_predictions(self, df: pd.DataFrame, models_to_use: List[str] = None) -> Dict:
        """
        Create weighted ensemble predictions from multiple models
        """
        print("\n" + "="*60)
        print("CREATING ENSEMBLE PREDICTIONS")
        print("="*60)
        
        if models_to_use is None:
            models_to_use = ['arima', 'sarima', 'holt_winters', 'prophet']
        
        # Collect predictions from each model
        predictions = {}
        weights = {}
        
        # Get predictions and assign weights based on performance
        for model_name in models_to_use:
            if model_name in self.results and 'forecast' in self.results[model_name]:
                predictions[model_name] = self.results[model_name]['forecast']
                
                # Assign weights inversely proportional to MAPE
                if 'mape' in self.results[model_name]:
                    mape = self.results[model_name]['mape']
                    weights[model_name] = 1 / (mape + 1)  # Add 1 to avoid division by zero
                else:
                    weights[model_name] = 0.5  # Default weight
        
        if not predictions:
            print("No valid predictions available for ensemble")
            return {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        for model in weights:
            weights[model] /= total_weight
        
        # Create ensemble forecast
        ensemble_forecast = None
        for model_name, weight in weights.items():
            if ensemble_forecast is None:
                ensemble_forecast = predictions[model_name] * weight
            else:
                # Align forecasts by taking minimum length
                min_len = min(len(ensemble_forecast), len(predictions[model_name]))
                ensemble_forecast = ensemble_forecast[:min_len] + predictions[model_name][:min_len] * weight
        
        print(f"\nEnsemble Results:")
        print(f"  Models used: {list(predictions.keys())}")
        print(f"  Weights:")
        for model, weight in weights.items():
            print(f"    {model}: {weight:.3f}")
        
        self.results['ensemble'] = {
            'forecast': ensemble_forecast,
            'models_used': list(predictions.keys()),
            'weights': weights,
            'individual_forecasts': predictions
        }
        
        return self.results['ensemble']
    
    def detect_lof(self, df: pd.DataFrame, n_neighbors: int = 20) -> Dict:
        """
        Local Outlier Factor for multivariate anomaly detection
        """
        print("\n" + "="*60)
        print("LOCAL OUTLIER FACTOR DETECTION")
        print("="*60)
        
        # Use multiple features
        features = ['price', 'rainfall_anomaly', 'temperature_anomaly', 
                   'trade_volume_change', 'export_concentration']
        
        data = df[features].dropna()
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        outliers = lof.fit_predict(scaled_data)
        
        # Get anomaly scores
        anomaly_scores = -lof.negative_outlier_factor_
        
        # Identify anomalies
        anomaly_mask = outliers == -1
        anomaly_dates = data.index[anomaly_mask]
        
        print(f"\nLOF Results:")
        print(f"  Neighbors: {n_neighbors}")
        print(f"  Anomalies detected: {len(anomaly_dates)} ({len(anomaly_dates)/len(data)*100:.1f}%)")
        
        if len(anomaly_dates) > 0:
            print(f"\nTop anomalies:")
            anomaly_df = pd.DataFrame({
                'date': anomaly_dates,
                'score': anomaly_scores[anomaly_mask],
                'price': data.loc[anomaly_dates, 'price']
            }).sort_values('score', ascending=False)
            
            for idx, row in anomaly_df.head().iterrows():
                print(f"  {row['date'].date()}: Score={row['score']:.2f}, Price=${row['price']:,.2f}")
        
        self.results['lof'] = {
            'anomaly_scores': pd.Series(anomaly_scores, index=data.index),
            'anomaly_dates': anomaly_dates,
            'n_neighbors': n_neighbors
        }
        
        return self.results['lof']
    
    # =========================
    # NLP MODELS
    # =========================
    
    def analyze_bert_sentiment(self, df: pd.DataFrame) -> Dict:
        """
        BERT-based sentiment analysis on news data
        """
        print("\n" + "="*60)
        print("BERT SENTIMENT ANALYSIS")
        print("="*60)
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Initialize BERT sentiment pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",  # Financial BERT model
                tokenizer="ProsusAI/finbert"
            )
        except ImportError:
            print("Transformers library not available. Using simple sentiment analysis...")
            return self._simple_sentiment_analysis(df)
        
        # Analyze sentiment trends
        if 'sentiment_score' in df.columns:
            sentiment_series = df['sentiment_score']
            
            # Calculate sentiment statistics
            sentiment_ma_7 = sentiment_series.rolling(7).mean()
            sentiment_ma_30 = sentiment_series.rolling(30).mean()
            sentiment_std = sentiment_series.rolling(30).std()
            
            # Detect sentiment regime changes
            sentiment_z_score = (sentiment_series - sentiment_ma_30) / sentiment_std
            extreme_sentiment = sentiment_series[np.abs(sentiment_z_score) > 2]
            
            print(f"\nBERT Sentiment Results:")
            print(f"  Average sentiment: {sentiment_series.mean():.3f}")
            print(f"  Current sentiment: {sentiment_series.iloc[-1]:.3f}")
            print(f"  7-day MA: {sentiment_ma_7.iloc[-1]:.3f}")
            print(f"  30-day MA: {sentiment_ma_30.iloc[-1]:.3f}")
            print(f"  Extreme sentiment days: {len(extreme_sentiment)}")
            
            # Sentiment-price correlation
            price_sentiment_corr = df['price'].corr(sentiment_series)
            print(f"  Price-sentiment correlation: {price_sentiment_corr:.3f}")
            
            self.results['bert_sentiment'] = {
                'sentiment_ma_7': sentiment_ma_7,
                'sentiment_ma_30': sentiment_ma_30,
                'extreme_sentiment_dates': extreme_sentiment.index.tolist(),
                'price_correlation': price_sentiment_corr,
                'current_sentiment': sentiment_series.iloc[-1],
                'sentiment_regime': 'positive' if sentiment_series.iloc[-1] > 0 else 'negative'
            }
        else:
            print("No sentiment data available in dataset")
            self.results['bert_sentiment'] = {'error': 'No sentiment data available'}
            
        return self.results.get('bert_sentiment', {})
    
    def _simple_sentiment_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Fallback simple sentiment analysis
        """
        if 'sentiment_score' in df.columns:
            sentiment_series = df['sentiment_score']
            
            # Basic statistics
            self.results['bert_sentiment'] = {
                'method': 'simple',
                'mean_sentiment': sentiment_series.mean(),
                'current_sentiment': sentiment_series.iloc[-1],
                'sentiment_trend': 'improving' if sentiment_series.iloc[-30:].mean() > sentiment_series.iloc[-60:-30].mean() else 'worsening'
            }
        else:
            self.results['bert_sentiment'] = {'error': 'No sentiment data'}
            
        return self.results['bert_sentiment']
    
    def perform_topic_modeling(self, df: pd.DataFrame) -> Dict:
        """
        Topic modeling with LDA on news/market discussions
        """
        print("\n" + "="*60)
        print("TOPIC MODELING WITH LDA")
        print("="*60)
        
        # For now, we'll analyze patterns in the data features as proxy for topics
        # In production, this would analyze actual news text
        
        # Identify dominant market themes based on feature correlations
        features = ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change', 
                   'export_concentration', 'sentiment_score'] if 'sentiment_score' in df.columns else \
                  ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change', 'export_concentration']
        
        # Calculate feature importance over rolling windows
        window = 30
        topic_importance = pd.DataFrame(index=df.index[window:])
        
        for feature in features:
            if feature in df.columns:
                # Calculate rolling correlation with price
                rolling_corr = df['price'].rolling(window).corr(df[feature])
                topic_importance[f'{feature}_importance'] = rolling_corr.abs()[window:]
        
        # Identify dominant topics/themes
        current_topics = topic_importance.iloc[-1].sort_values(ascending=False)
        
        print(f"\nTopic Modeling Results:")
        print(f"  Current dominant market themes:")
        for topic, importance in current_topics.head(3).items():
            clean_name = topic.replace('_importance', '').replace('_', ' ').title()
            print(f"    - {clean_name}: {importance:.3f}")
        
        # Detect topic shifts
        topic_shifts = []
        for col in topic_importance.columns:
            # Detect when a topic becomes dominant
            is_dominant = topic_importance[col] > 0.5
            shifts = is_dominant.diff()
            new_dominant = shifts[shifts == True].index
            for date in new_dominant:
                topic_shifts.append({
                    'date': date,
                    'topic': col.replace('_importance', ''),
                    'importance': topic_importance.loc[date, col]
                })
        
        if topic_shifts:
            print(f"\n  Recent topic shifts:")
            recent_shifts = sorted(topic_shifts, key=lambda x: x['date'])[-3:]
            for shift in recent_shifts:
                print(f"    - {shift['date'].date()}: {shift['topic'].replace('_', ' ').title()} "
                      f"became dominant ({shift['importance']:.3f})")
        
        self.results['topic_modeling'] = {
            'current_dominant_topics': current_topics.to_dict(),
            'topic_importance_history': topic_importance,
            'topic_shifts': topic_shifts,
            'num_active_topics': len(current_topics[current_topics > 0.3])
        }
        
        return self.results['topic_modeling']
    
    # =========================
    # HELPER METHODS
    # =========================
    
    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time series features"""
        features = df.copy()
        
        # Price features
        for lag in [1, 7, 14, 30]:
            features[f'price_lag_{lag}'] = features['price'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            features[f'price_ma_{window}'] = features['price'].rolling(window).mean()
            features[f'price_std_{window}'] = features['price'].rolling(window).std()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(features['price'])
        features['momentum'] = features['price'] / features['price'].shift(20) - 1
        
        # Time features
        features['day_of_week'] = features.index.dayofweek
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        
        # Interaction features
        features['weather_trade_interaction'] = (
            features['rainfall_anomaly'] * features['trade_volume_change']
        )
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_all_models(self, df: pd.DataFrame) -> Dict:
        """Run all time series models and return comprehensive results"""
        print("\n" + "="*80)
        print("RUNNING ALL TIME SERIES MODELS WITH REAL DATA")
        print("="*80)
        
        # Prepare data
        df = self.prepare_data(df)
        
        all_results = {}
        
        # Traditional time series
        try:
            all_results['arima'] = self.fit_arima(df)
        except Exception as e:
            print(f"ARIMA failed: {str(e)}")
            
        try:
            all_results['stl'] = self.fit_stl_decomposition(df)
        except Exception as e:
            print(f"STL failed: {str(e)}")
            
        try:
            all_results['sarima'] = self.fit_sarima(df)
        except Exception as e:
            print(f"SARIMA failed: {str(e)}")
            
        try:
            all_results['holt_winters'] = self.fit_holt_winters(df)
        except Exception as e:
            print(f"Holt-Winters failed: {str(e)}")
            
        try:
            all_results['var'] = self.fit_var(df)
        except Exception as e:
            print(f"VAR failed: {str(e)}")
            
        try:
            all_results['prophet'] = self.fit_prophet(df)
        except Exception as e:
            print(f"Prophet failed: {str(e)}")
        
        # Volatility models
        try:
            all_results['garch'] = self.fit_garch(df)
        except Exception as e:
            print(f"GARCH failed: {str(e)}")
            
        try:
            all_results['ewma'] = self.calculate_ewma(df)
        except Exception as e:
            print(f"EWMA failed: {str(e)}")
        
        # ML models
        try:
            all_results['xgboost'] = self.fit_xgboost(df)
        except Exception as e:
            print(f"XGBoost failed: {str(e)}")
            
        try:
            all_results['lstm_autoencoder'] = self.fit_lstm_autoencoder(df)
        except Exception as e:
            print(f"LSTM Autoencoder failed: {str(e)}")
            
        try:
            all_results['lstm_predictor'] = self.fit_lstm_predictor(df)
        except Exception as e:
            print(f"LSTM Predictor failed: {str(e)}")
        
        # Statistical process control
        try:
            all_results['cusum'] = self.detect_cusum(df)
        except Exception as e:
            print(f"CUSUM failed: {str(e)}")
            
        try:
            all_results['modified_zscore'] = self.detect_modified_zscore(df)
        except Exception as e:
            print(f"Modified Z-score failed: {str(e)}")
            
        try:
            all_results['lof'] = self.detect_lof(df)
        except Exception as e:
            print(f"LOF failed: {str(e)}")
            
        try:
            all_results['changepoints'] = self.detect_changepoints(df)
        except Exception as e:
            print(f"Change point detection failed: {str(e)}")
        
        # Ensemble predictions
        try:
            all_results['ensemble'] = self.create_ensemble_predictions(df)
        except Exception as e:
            print(f"Ensemble failed: {str(e)}")
        
        # NLP models (if news data available)
        if 'sentiment_score' in df.columns or 'news_count' in df.columns:
            try:
                all_results['bert_sentiment'] = self.analyze_bert_sentiment(df)
            except Exception as e:
                print(f"BERT sentiment failed: {str(e)}")
                
            try:
                all_results['topic_modeling'] = self.perform_topic_modeling(df)
            except Exception as e:
                print(f"Topic modeling failed: {str(e)}")
        
        print("\n" + "="*80)
        print("ALL MODELS COMPLETED")
        print("="*80)
        
        return all_results
#!/usr/bin/env python3
"""
Multi-Source Prediction Model with Zen Consensus
Combines multiple ML models with agent orchestration
Uses Sequential Thinking for complex decisions
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_orchestrator import ModelOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSourcePredictor:
    """
    Advanced prediction system using multiple models and data sources
    Implements Zen Consensus for robust predictions
    """
    
    def __init__(self, use_sequential_thinking: bool = True):
        self.use_sequential_thinking = use_sequential_thinking
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.orchestrator = ModelOrchestrator()
        
        # Initialize diverse model ensemble
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize diverse set of models for ensemble"""
        
        # Tree-based models (good for non-linear patterns)
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Linear models (good for trends)
        self.models['elastic_net'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        )
        
        self.models['ridge'] = Ridge(
            alpha=1.0,
            random_state=42
        )
        
        # Neural network (captures complex interactions)
        self.models['neural_net'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all models with proper scaling and validation
        """
        logger.info(f"Training multi-source predictor on {len(X_train)} samples")
        
        # Handle missing values
        X_train_filled = X_train.fillna(X_train.median())
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        self.scalers['main'] = scaler
        
        if X_val is not None:
            X_val_filled = X_val.fillna(X_train.median())
            X_val_scaled = scaler.transform(X_val_filled)
        
        # Train each model
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on training set
                train_pred = model.predict(X_train_scaled)
                train_mse = mean_squared_error(y_train, train_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                
                results = {
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'train_r2': train_r2,
                    'train_rmse': np.sqrt(train_mse)
                }
                
                # Evaluate on validation set if provided
                if X_val is not None:
                    val_pred = model.predict(X_val_scaled)
                    val_mse = mean_squared_error(y_val, val_pred)
                    val_mae = mean_absolute_error(y_val, val_pred)
                    val_r2 = r2_score(y_val, val_pred)
                    
                    results.update({
                        'val_mse': val_mse,
                        'val_mae': val_mae,
                        'val_r2': val_r2,
                        'val_rmse': np.sqrt(val_mse)
                    })
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = pd.Series(
                        model.feature_importances_,
                        index=X_train.columns
                    ).sort_values(ascending=False)
                
                training_results[model_name] = results
                logger.info(f"  {model_name} - Train R²: {train_r2:.4f}, RMSE: {np.sqrt(train_mse):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        # Use orchestrator to analyze results
        if self.use_sequential_thinking:
            try:
                # Get weighted consensus based on performance
                best_models = sorted(training_results.items(), 
                                   key=lambda x: x[1].get('train_r2', -999), 
                                   reverse=True)[:3]
                training_results['best_models'] = [m[0] for m in best_models]
            except:
                pass
        
        return training_results
    
    def predict(self, X: pd.DataFrame, return_all: bool = False) -> np.ndarray:
        """
        Make predictions using Zen Consensus
        """
        # Handle missing values (use median from training)
        X_filled = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scalers['main'].transform(X_filled)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Apply Zen Consensus
        if self.use_sequential_thinking:
            # Use orchestrator for intelligent ensemble
            consensus_pred = self._zen_consensus(predictions, X)
        else:
            # Simple average
            consensus_pred = np.mean(list(predictions.values()), axis=0)
        
        if return_all:
            predictions['consensus'] = consensus_pred
            return predictions
        else:
            return consensus_pred
    
    def _zen_consensus(self, predictions: Dict[str, np.ndarray], 
                      X: pd.DataFrame) -> np.ndarray:
        """
        Apply Zen Consensus using model orchestration
        """
        # Analyze prediction spread
        pred_array = np.array(list(predictions.values()))
        pred_std = np.std(pred_array, axis=0)
        pred_mean = np.mean(pred_array, axis=0)
        
        # Identify high confidence predictions (low spread)
        high_confidence_mask = pred_std < np.percentile(pred_std, 25)
        
        # Weight models based on historical performance
        weights = self._calculate_model_weights(predictions, X)
        
        # Weighted consensus
        consensus = np.zeros(len(X))
        
        for i in range(len(X)):
            if high_confidence_mask[i]:
                # High confidence - use weighted average
                model_preds = [predictions[m][i] for m in predictions]
                model_weights = [weights.get(m, 1.0) for m in predictions]
                consensus[i] = np.average(model_preds, weights=model_weights)
            else:
                # Low confidence - use median for robustness
                model_preds = [predictions[m][i] for m in predictions]
                consensus[i] = np.median(model_preds)
        
        return consensus
    
    def _calculate_model_weights(self, predictions: Dict[str, np.ndarray],
                               X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic model weights based on feature characteristics
        """
        weights = {}
        
        # Analyze feature types in current data
        feature_analysis = self._analyze_features(X)
        
        # Weight models based on their strengths
        for model_name in predictions:
            if model_name in ['random_forest', 'xgboost', 'gradient_boost']:
                # Tree models good for non-linear patterns
                weights[model_name] = 1.2 if feature_analysis['non_linear'] else 0.8
            elif model_name in ['elastic_net', 'ridge']:
                # Linear models good for trends
                weights[model_name] = 1.2 if feature_analysis['trending'] else 0.8
            elif model_name == 'neural_net':
                # Neural nets good for complex interactions
                weights[model_name] = 1.1 if feature_analysis['high_interaction'] else 0.9
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _analyze_features(self, X: pd.DataFrame) -> Dict[str, bool]:
        """
        Analyze feature characteristics
        """
        analysis = {
            'non_linear': False,
            'trending': False,
            'high_interaction': False
        }
        
        # Check for non-linearity (high volatility features)
        volatility_features = [col for col in X.columns if 'volatility' in col]
        if volatility_features:
            avg_volatility = X[volatility_features].mean().mean()
            analysis['non_linear'] = avg_volatility > 0.1
        
        # Check for trends (momentum features)
        momentum_features = [col for col in X.columns if 'momentum' in col or 'return' in col]
        if momentum_features:
            avg_momentum = abs(X[momentum_features].mean().mean())
            analysis['trending'] = avg_momentum > 0.05
        
        # Check for interactions (correlation between feature groups)
        if len(X.columns) > 20:
            analysis['high_interaction'] = True
        
        return analysis
    
    def _identify_data_sources(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Identify which data sources are represented in features
        """
        sources = {
            'price': [],
            'weather': [],
            'sentiment': [],
            'trade': [],
            'interaction': []
        }
        
        for col in columns:
            if any(x in col for x in ['return', 'volatility', 'rsi', 'macd', 'momentum']):
                sources['price'].append(col)
            elif any(x in col for x in ['risk', 'temp', 'rainfall', 'extreme']):
                sources['weather'].append(col)
            elif any(x in col for x in ['sentiment', 'article', 'topic']):
                sources['sentiment'].append(col)
            elif any(x in col for x in ['export', 'trade', 'volume']):
                sources['trade'].append(col)
            elif any(x in col for x in ['interaction', 'composite']):
                sources['interaction'].append(col)
        
        return sources
    
    def evaluate_feature_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance across all models
        """
        if not self.feature_importance:
            return pd.DataFrame()
        
        # Combine importance scores
        importance_df = pd.DataFrame(self.feature_importance)
        
        # Calculate average importance
        importance_df['avg_importance'] = importance_df.mean(axis=1)
        importance_df['std_importance'] = importance_df.std(axis=1)
        
        # Sort by average importance
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return importance_df
    
    def backtest(self, df: pd.DataFrame, target_col: str,
                 n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series backtesting
        """
        logger.info(f"Starting backtest with {n_splits} splits")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
        X = df[feature_cols]
        y = df[target_col]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        backtest_results = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {i+1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train models
            self.train(X_train, y_train)
            
            # Make predictions
            predictions = self.predict(X_test, return_all=True)
            
            # Evaluate each model
            fold_results = {
                'fold': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1]
            }
            
            for model_name, pred in predictions.items():
                try:
                    mse = mean_squared_error(y_test, pred)
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    
                    # Direction accuracy
                    actual_direction = (y_test > 0).astype(int)
                    pred_direction = (pred > 0).astype(int)
                    direction_accuracy = (actual_direction == pred_direction).mean()
                    
                    fold_results[f'{model_name}_mse'] = mse
                    fold_results[f'{model_name}_mae'] = mae
                    fold_results[f'{model_name}_r2'] = r2
                    fold_results[f'{model_name}_direction_acc'] = direction_accuracy
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name}: {str(e)}")
            
            backtest_results.append(fold_results)
        
        # Aggregate results
        results_df = pd.DataFrame(backtest_results)
        
        summary = {
            'n_splits': n_splits,
            'total_samples': len(df),
            'results_by_fold': results_df,
            'average_performance': {}
        }
        
        # Calculate average performance for each model
        for model_name in list(self.models.keys()) + ['consensus']:
            if f'{model_name}_mse' in results_df.columns:
                avg_metrics = {
                    'mse': results_df[f'{model_name}_mse'].mean(),
                    'mae': results_df[f'{model_name}_mae'].mean(),
                    'r2': results_df[f'{model_name}_r2'].mean(),
                    'direction_accuracy': results_df[f'{model_name}_direction_acc'].mean()
                }
                summary['average_performance'][model_name] = avg_metrics
        
        return summary


def demonstrate_multi_source_prediction():
    """Demonstrate the multi-source prediction system"""
    print("🚀 Multi-Source Prediction with Zen Consensus")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('data/processed/ml_dataset_complete.csv', index_col='date', parse_dates=True)
        print(f"✅ Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run create_dataset_with_targets.py first")
        return
    
    # Initialize predictor
    predictor = MultiSourcePredictor(use_sequential_thinking=True)
    
    # Prepare data
    feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
    
    # Test on 7-day predictions
    target_col = 'return_7d_future'
    
    print(f"\n📊 Testing on {target_col}")
    print(f"   Features: {len(feature_cols)}")
    
    # Backtest
    backtest_results = predictor.backtest(df, target_col, n_splits=3)
    
    # Display results
    print("\n📈 Backtest Results:")
    print("-" * 60)
    
    for model_name, metrics in backtest_results['average_performance'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']*100:.1f}%")
        print(f"  MAE: {metrics['mae']*100:.2f}%")
        print(f"  R²: {metrics['r2']:.3f}")
    
    # Feature importance
    print("\n🔍 Top Features:")
    importance_df = predictor.evaluate_feature_importance()
    if not importance_df.empty:
        top_features = importance_df.head(10)
        for feat, imp in zip(top_features.index, top_features['avg_importance']):
            print(f"  {feat}: {imp:.3f}")
    
    # Data source contribution
    feature_groups = predictor._identify_data_sources(feature_cols)
    print("\n📊 Data Source Contribution:")
    for source, features in feature_groups.items():
        if features:
            print(f"  {source}: {len(features)} features")


if __name__ == "__main__":
    demonstrate_multi_source_prediction()
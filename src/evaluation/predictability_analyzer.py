#!/usr/bin/env python3
"""
Comprehensive Model Predictability Analysis
Evaluates whether the models have true predictive power or are just fitting noise
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.multi_source_predictor import MultiSourcePredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictabilityAnalyzer:
    """
    Comprehensive analysis of model predictability
    Tests for true predictive power vs random chance
    """
    
    def __init__(self):
        self.results = {}
        self.visualizations = []
        
    def analyze_predictability(self, df: pd.DataFrame, 
                             target_cols: List[str],
                             n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive predictability analysis
        """
        logger.info("Starting comprehensive predictability analysis")
        
        # 1. Statistical tests for predictability
        statistical_tests = self._run_statistical_tests(df, target_cols)
        
        # 2. Benchmark comparisons
        benchmark_results = self._compare_to_benchmarks(df, target_cols)
        
        # 3. Feature importance stability
        feature_stability = self._analyze_feature_stability(df, target_cols)
        
        # 4. Temporal consistency
        temporal_analysis = self._analyze_temporal_consistency(df, target_cols)
        
        # 5. Random permutation tests
        permutation_tests = self._run_permutation_tests(df, target_cols, n_simulations)
        
        # 6. Economic significance
        economic_analysis = self._analyze_economic_significance(df, target_cols)
        
        # 7. Prediction confidence analysis
        confidence_analysis = self._analyze_prediction_confidence(df, target_cols)
        
        # Compile results
        self.results = {
            'statistical_tests': statistical_tests,
            'benchmark_comparison': benchmark_results,
            'feature_stability': feature_stability,
            'temporal_consistency': temporal_analysis,
            'permutation_tests': permutation_tests,
            'economic_significance': economic_analysis,
            'confidence_analysis': confidence_analysis,
            'overall_verdict': self._generate_verdict()
        }
        
        return self.results
    
    def _run_statistical_tests(self, df: pd.DataFrame, 
                              target_cols: List[str]) -> Dict[str, Any]:
        """
        Run statistical tests for predictability
        """
        tests = {}
        
        for target in target_cols:
            logger.info(f"Running statistical tests for {target}")
            
            # 1. Autocorrelation test
            autocorr = self._test_autocorrelation(df[target])
            
            # 2. Random walk test
            random_walk = self._test_random_walk(df[target])
            
            # 3. Mean reversion test
            mean_reversion = self._test_mean_reversion(df[target])
            
            # 4. Variance ratio test
            variance_ratio = self._variance_ratio_test(df[target])
            
            tests[target] = {
                'autocorrelation': autocorr,
                'random_walk': random_walk,
                'mean_reversion': mean_reversion,
                'variance_ratio': variance_ratio,
                'is_predictable': self._assess_predictability(
                    autocorr, random_walk, mean_reversion, variance_ratio
                )
            }
        
        return tests
    
    def _test_autocorrelation(self, series: pd.Series) -> Dict[str, Any]:
        """Test for autocorrelation in returns"""
        # Calculate autocorrelations
        acf_values = []
        for lag in range(1, min(11, len(series) // 3)):
            if len(series) > lag:
                acf = series.autocorr(lag=lag)
                acf_values.append((lag, acf))
        
        # Ljung-Box test
        if len(series) > 10:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(series.dropna(), lags=min(10, len(series)//4), return_df=True)
            
            return {
                'autocorrelations': acf_values,
                'ljung_box_stat': float(result['lb_stat'].iloc[-1]),
                'ljung_box_pvalue': float(result['lb_pvalue'].iloc[-1]),
                'has_autocorrelation': float(result['lb_pvalue'].iloc[-1]) < 0.05
            }
        else:
            return {
                'autocorrelations': acf_values,
                'ljung_box_stat': None,
                'ljung_box_pvalue': None,
                'has_autocorrelation': False
            }
    
    def _test_random_walk(self, series: pd.Series) -> Dict[str, Any]:
        """Test if series follows random walk"""
        if len(series) < 10:
            return {'is_random_walk': None, 'adf_stat': None, 'adf_pvalue': None}
            
        from statsmodels.tsa.stattools import adfuller
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'is_random_walk': adf_result[1] > 0.05,  # Fail to reject null = random walk
            'adf_stat': adf_result[0],
            'adf_pvalue': adf_result[1]
        }
    
    def _test_mean_reversion(self, series: pd.Series) -> Dict[str, Any]:
        """Test for mean reversion"""
        if len(series) < 20:
            return {'has_mean_reversion': None, 'half_life': None}
            
        # Calculate half-life of mean reversion
        lagged = series.shift(1)
        delta = series - lagged
        
        # Remove NaN values
        mask = ~(delta.isna() | lagged.isna())
        delta_clean = delta[mask]
        lagged_clean = lagged[mask]
        
        if len(delta_clean) < 10:
            return {'has_mean_reversion': None, 'half_life': None}
        
        # OLS regression
        beta = np.cov(delta_clean, lagged_clean)[0, 1] / np.var(lagged_clean)
        
        # Half-life calculation
        half_life = -np.log(2) / beta if beta < 0 else None
        
        return {
            'has_mean_reversion': beta < 0,
            'half_life': half_life,
            'mean_reversion_speed': abs(beta) if beta < 0 else 0
        }
    
    def _variance_ratio_test(self, series: pd.Series, periods: List[int] = [2, 4, 8]) -> Dict[str, Any]:
        """Variance ratio test for random walk"""
        results = {}
        
        for period in periods:
            if len(series) < period * 4:
                continue
                
            # Calculate variance ratio
            ret_1 = series.pct_change().dropna()
            ret_k = series.pct_change(period).dropna()
            
            if len(ret_1) > 0 and len(ret_k) > 0:
                var_1 = ret_1.var()
                var_k = ret_k.var() / period
                
                vr = var_k / var_1 if var_1 > 0 else 1.0
                
                results[f'period_{period}'] = {
                    'variance_ratio': vr,
                    'rejects_random_walk': abs(vr - 1) > 0.2
                }
        
        return results
    
    def _assess_predictability(self, autocorr: Dict, random_walk: Dict,
                             mean_reversion: Dict, variance_ratio: Dict) -> Dict[str, Any]:
        """Assess overall predictability based on tests"""
        
        evidence_for_predictability = 0
        evidence_against = 0
        
        # Check autocorrelation
        if autocorr.get('has_autocorrelation'):
            evidence_for_predictability += 1
        else:
            evidence_against += 1
        
        # Check random walk
        if random_walk.get('is_random_walk') is False:
            evidence_for_predictability += 1
        elif random_walk.get('is_random_walk') is True:
            evidence_against += 1
        
        # Check mean reversion
        if mean_reversion.get('has_mean_reversion'):
            evidence_for_predictability += 1
        
        # Check variance ratios
        for period_data in variance_ratio.values():
            if isinstance(period_data, dict) and period_data.get('rejects_random_walk'):
                evidence_for_predictability += 0.5
        
        total_evidence = evidence_for_predictability + evidence_against
        predictability_score = evidence_for_predictability / total_evidence if total_evidence > 0 else 0.5
        
        return {
            'predictability_score': predictability_score,
            'is_predictable': predictability_score > 0.6,
            'confidence': 'high' if predictability_score > 0.8 or predictability_score < 0.2 else 'medium'
        }
    
    def _compare_to_benchmarks(self, df: pd.DataFrame, 
                              target_cols: List[str]) -> Dict[str, Any]:
        """Compare model performance to simple benchmarks"""
        logger.info("Comparing to benchmark models")
        
        # Prepare data
        feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
        
        benchmarks = {}
        
        for target in target_cols:
            logger.info(f"Testing benchmarks for {target}")
            
            # Create benchmarks
            bench_results = {}
            
            # 1. Random predictions (baseline)
            bench_results['random'] = self._test_random_predictor(df, target)
            
            # 2. Always predict zero (no change)
            bench_results['zero'] = self._test_zero_predictor(df, target)
            
            # 3. Simple momentum (previous return)
            bench_results['momentum'] = self._test_momentum_predictor(df, target)
            
            # 4. Mean prediction
            bench_results['mean'] = self._test_mean_predictor(df, target)
            
            # 5. Our ML model
            bench_results['ml_model'] = self._test_ml_model(df[feature_cols], df[target])
            
            # Calculate outperformance
            ml_score = bench_results['ml_model']['direction_accuracy']
            best_benchmark = max(
                bench_results['random']['direction_accuracy'],
                bench_results['zero']['direction_accuracy'],
                bench_results['momentum']['direction_accuracy'],
                bench_results['mean']['direction_accuracy']
            )
            
            benchmarks[target] = {
                'results': bench_results,
                'ml_outperformance': ml_score - best_benchmark,
                'beats_all_benchmarks': ml_score > best_benchmark,
                'best_benchmark': max(bench_results.items(), 
                                    key=lambda x: x[1]['direction_accuracy'])[0]
            }
        
        return benchmarks
    
    def _test_random_predictor(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Test random prediction baseline"""
        np.random.seed(42)
        y_true = df[target].values
        
        # Multiple random trials
        accuracies = []
        for _ in range(100):
            y_pred = np.random.choice([-1, 1], size=len(y_true))
            accuracy = ((y_true > 0) == (y_pred > 0)).mean()
            accuracies.append(accuracy)
        
        return {
            'direction_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
    
    def _test_zero_predictor(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Test always predicting no change"""
        y_true = df[target].values
        y_pred = np.zeros_like(y_true)
        
        return {
            'direction_accuracy': ((y_true > 0) == (y_pred > 0)).mean(),
            'mae': np.abs(y_true - y_pred).mean()
        }
    
    def _test_momentum_predictor(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Test simple momentum predictor"""
        # Use previous period return as prediction
        if 'return_1d' in df.columns:
            y_true = df[target].values[1:]
            y_pred = df['return_1d'].values[:-1]
            
            return {
                'direction_accuracy': ((y_true > 0) == (y_pred > 0)).mean(),
                'correlation': np.corrcoef(y_true, y_pred)[0, 1]
            }
        else:
            return {'direction_accuracy': 0.5, 'correlation': 0.0}
    
    def _test_mean_predictor(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Test predicting historical mean"""
        y_true = df[target].values
        
        # Use expanding window mean
        predictions = []
        for i in range(1, len(y_true)):
            pred = y_true[:i].mean()
            predictions.append(pred)
        
        if predictions:
            y_pred = np.array(predictions)
            y_true_adj = y_true[1:]
            
            return {
                'direction_accuracy': ((y_true_adj > 0) == (y_pred > 0)).mean(),
                'mae': np.abs(y_true_adj - y_pred).mean()
            }
        else:
            return {'direction_accuracy': 0.5, 'mae': 0.0}
    
    def _test_ml_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Test our ML model using time series CV"""
        if len(X) < 10:
            return {'direction_accuracy': 0.5, 'mae': 0.0}
            
        predictor = MultiSourcePredictor(use_sequential_thinking=False)
        
        # Simple train/test split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if len(X_train) < 5 or len(X_test) < 2:
            return {'direction_accuracy': 0.5, 'mae': 0.0}
        
        try:
            # Train
            predictor.train(X_train, y_train)
            
            # Predict
            y_pred = predictor.predict(X_test)
            
            return {
                'direction_accuracy': ((y_test > 0) == (y_pred > 0)).mean(),
                'mae': np.abs(y_test - y_pred).mean(),
                'correlation': np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
            }
        except:
            return {'direction_accuracy': 0.5, 'mae': 0.0}
    
    def _analyze_feature_stability(self, df: pd.DataFrame, 
                                  target_cols: List[str]) -> Dict[str, Any]:
        """Analyze stability of feature importance over time"""
        logger.info("Analyzing feature importance stability")
        
        feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
        
        stability_results = {}
        
        for target in target_cols:
            # Use rolling windows to check feature importance stability
            window_size = min(10, len(df) // 3)
            
            if window_size < 5:
                stability_results[target] = {'stable_features': [], 'stability_score': 0}
                continue
                
            importance_over_time = []
            
            for i in range(window_size, len(df) - 5):
                window_data = df.iloc[i-window_size:i]
                
                try:
                    # Train simple model
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(window_data[feature_cols].fillna(0), window_data[target])
                    
                    # Get importance
                    importance = pd.Series(rf.feature_importances_, index=feature_cols)
                    importance_over_time.append(importance)
                except:
                    continue
            
            if len(importance_over_time) > 2:
                # Calculate stability
                importance_df = pd.DataFrame(importance_over_time)
                stability_scores = importance_df.std() / (importance_df.mean() + 1e-6)
                
                # Find stable features (low coefficient of variation)
                stable_features = stability_scores[stability_scores < 0.5].index.tolist()
                
                stability_results[target] = {
                    'stable_features': stable_features[:10],
                    'stability_score': 1 - stability_scores.mean(),
                    'most_important_stable': importance_df[stable_features].mean().nlargest(5).to_dict()
                }
            else:
                stability_results[target] = {'stable_features': [], 'stability_score': 0}
        
        return stability_results
    
    def _analyze_temporal_consistency(self, df: pd.DataFrame, 
                                    target_cols: List[str]) -> Dict[str, Any]:
        """Check if predictive patterns are consistent over time"""
        logger.info("Analyzing temporal consistency")
        
        consistency_results = {}
        
        for target in target_cols:
            # Divide data into time periods
            n_periods = min(3, len(df) // 6)
            
            if n_periods < 2:
                consistency_results[target] = {'is_consistent': False, 'consistency_score': 0}
                continue
                
            period_size = len(df) // n_periods
            period_stats = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df)
                
                period_data = df[target].iloc[start_idx:end_idx]
                
                period_stats.append({
                    'mean': period_data.mean(),
                    'std': period_data.std(),
                    'skew': period_data.skew(),
                    'positive_ratio': (period_data > 0).mean()
                })
            
            # Check consistency
            stats_df = pd.DataFrame(period_stats)
            
            # Coefficient of variation for each statistic
            cv_scores = stats_df.std() / (stats_df.mean().abs() + 1e-6)
            
            consistency_results[target] = {
                'is_consistent': cv_scores.mean() < 0.5,
                'consistency_score': 1 - cv_scores.mean(),
                'period_statistics': period_stats,
                'most_consistent_metric': cv_scores.idxmin()
            }
        
        return consistency_results
    
    def _run_permutation_tests(self, df: pd.DataFrame, 
                               target_cols: List[str],
                               n_simulations: int) -> Dict[str, Any]:
        """Test if model performance is better than random permutations"""
        logger.info(f"Running permutation tests with {n_simulations} simulations")
        
        feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
        permutation_results = {}
        
        for target in target_cols:
            logger.info(f"Permutation test for {target}")
            
            # Get actual model performance
            actual_performance = self._test_ml_model(df[feature_cols], df[target])
            actual_accuracy = actual_performance['direction_accuracy']
            
            # Run permutations
            permuted_accuracies = []
            
            for i in range(min(n_simulations, 100)):  # Limit for performance
                # Shuffle target
                shuffled_target = df[target].sample(frac=1, random_state=i).reset_index(drop=True)
                
                # Test on shuffled data
                perm_performance = self._test_ml_model(
                    df[feature_cols].reset_index(drop=True), 
                    shuffled_target
                )
                permuted_accuracies.append(perm_performance['direction_accuracy'])
            
            # Calculate p-value
            p_value = (np.array(permuted_accuracies) >= actual_accuracy).mean()
            
            permutation_results[target] = {
                'actual_accuracy': actual_accuracy,
                'mean_permuted_accuracy': np.mean(permuted_accuracies),
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'percentile': (np.array(permuted_accuracies) < actual_accuracy).mean() * 100
            }
        
        return permutation_results
    
    def _analyze_economic_significance(self, df: pd.DataFrame, 
                                     target_cols: List[str]) -> Dict[str, Any]:
        """Analyze if predictions are economically meaningful"""
        logger.info("Analyzing economic significance")
        
        economic_results = {}
        
        for target in target_cols:
            # Calculate potential returns from predictions
            feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
            
            if len(df) < 20:
                economic_results[target] = {
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profitable_trades_pct': 0.5
                }
                continue
            
            # Simple backtest
            predictor = MultiSourcePredictor(use_sequential_thinking=False)
            
            # Use expanding window
            predictions = []
            actuals = []
            
            min_train = 10
            for i in range(min_train, len(df)):
                train_data = df.iloc[:i]
                test_point = df.iloc[i:i+1]
                
                try:
                    predictor.train(train_data[feature_cols], train_data[target])
                    pred = predictor.predict(test_point[feature_cols])[0]
                    
                    predictions.append(pred)
                    actuals.append(test_point[target].values[0])
                except:
                    continue
            
            if len(predictions) > 5:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                # Calculate strategy returns (go long if positive prediction)
                strategy_returns = actuals * np.sign(predictions)
                
                # Key metrics
                sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-6)
                
                # Maximum drawdown
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Win rate
                profitable = (strategy_returns > 0).mean()
                
                economic_results[target] = {
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'profitable_trades_pct': profitable,
                    'annual_return': strategy_returns.mean() * 252,
                    'annual_volatility': strategy_returns.std() * np.sqrt(252),
                    'is_profitable': sharpe > 0.5
                }
            else:
                economic_results[target] = {
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profitable_trades_pct': 0.5
                }
        
        return economic_results
    
    def _analyze_prediction_confidence(self, df: pd.DataFrame, 
                                     target_cols: List[str]) -> Dict[str, Any]:
        """Analyze model confidence and calibration"""
        logger.info("Analyzing prediction confidence")
        
        confidence_results = {}
        feature_cols = [col for col in df.columns if 'future' not in col and col != 'current_price']
        
        for target in target_cols:
            if len(df) < 20:
                confidence_results[target] = {
                    'well_calibrated': False,
                    'confidence_correlation': 0
                }
                continue
            
            predictor = MultiSourcePredictor(use_sequential_thinking=True)
            
            # Get predictions with all models
            split_idx = int(len(df) * 0.7)
            X_train, X_test = df[feature_cols].iloc[:split_idx], df[feature_cols].iloc[split_idx:]
            y_train, y_test = df[target].iloc[:split_idx], df[target].iloc[split_idx:]
            
            if len(X_train) < 5 or len(X_test) < 2:
                confidence_results[target] = {
                    'well_calibrated': False,
                    'confidence_correlation': 0
                }
                continue
            
            try:
                predictor.train(X_train, y_train)
                all_predictions = predictor.predict(X_test, return_all=True)
                
                # Calculate prediction spread as confidence measure
                pred_array = np.array([all_predictions[m] for m in predictor.models.keys()])
                pred_std = np.std(pred_array, axis=0)
                pred_mean = all_predictions['consensus']
                
                # Check calibration: low spread should mean higher accuracy
                errors = np.abs(y_test.values - pred_mean)
                
                # Correlation between confidence (inverse of spread) and accuracy
                confidence = 1 / (pred_std + 1e-6)
                accuracy = 1 / (errors + 1e-6)
                
                correlation = np.corrcoef(confidence, accuracy)[0, 1]
                
                # Calibration buckets
                n_buckets = min(3, len(pred_std) // 3)
                if n_buckets >= 2:
                    # Sort by confidence
                    sorted_idx = np.argsort(pred_std)
                    bucket_size = len(sorted_idx) // n_buckets
                    
                    calibration_data = []
                    for i in range(n_buckets):
                        start = i * bucket_size
                        end = (i + 1) * bucket_size if i < n_buckets - 1 else len(sorted_idx)
                        bucket_idx = sorted_idx[start:end]
                        
                        bucket_errors = errors[bucket_idx].mean()
                        bucket_confidence = pred_std[bucket_idx].mean()
                        
                        calibration_data.append({
                            'confidence_level': f'bucket_{i+1}',
                            'avg_spread': bucket_confidence,
                            'avg_error': bucket_errors
                        })
                    
                    # Well calibrated if high confidence buckets have lower errors
                    is_calibrated = calibration_data[0]['avg_error'] > calibration_data[-1]['avg_error']
                else:
                    calibration_data = []
                    is_calibrated = False
                
                confidence_results[target] = {
                    'well_calibrated': is_calibrated,
                    'confidence_correlation': correlation,
                    'calibration_data': calibration_data,
                    'avg_prediction_spread': pred_std.mean()
                }
            except Exception as e:
                logger.error(f"Error in confidence analysis: {str(e)}")
                confidence_results[target] = {
                    'well_calibrated': False,
                    'confidence_correlation': 0
                }
        
        return confidence_results
    
    def _generate_verdict(self) -> Dict[str, Any]:
        """Generate overall verdict on predictability"""
        
        verdict = {
            'has_predictive_power': False,
            'confidence_level': 'low',
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze all evidence
        evidence_scores = []
        
        # 1. Statistical evidence
        for target, tests in self.results.get('statistical_tests', {}).items():
            if tests.get('is_predictable', {}).get('is_predictable'):
                evidence_scores.append(1)
                verdict['key_findings'].append(f"{target} shows statistical predictability")
            else:
                evidence_scores.append(0)
        
        # 2. Benchmark outperformance
        for target, bench in self.results.get('benchmark_comparison', {}).items():
            if bench.get('beats_all_benchmarks'):
                evidence_scores.append(1)
                outperformance = bench.get('ml_outperformance', 0)
                verdict['key_findings'].append(
                    f"ML model beats benchmarks by {outperformance*100:.1f}% for {target}"
                )
            else:
                evidence_scores.append(0)
        
        # 3. Permutation test significance
        for target, perm in self.results.get('permutation_tests', {}).items():
            if perm.get('is_significant'):
                evidence_scores.append(1)
                verdict['key_findings'].append(
                    f"Performance on {target} is statistically significant (p={perm['p_value']:.3f})"
                )
            else:
                evidence_scores.append(0)
        
        # 4. Economic significance
        for target, econ in self.results.get('economic_significance', {}).items():
            if econ.get('sharpe_ratio', 0) > 0.5:
                evidence_scores.append(1)
                verdict['key_findings'].append(
                    f"Economically significant with Sharpe ratio {econ['sharpe_ratio']:.2f}"
                )
            else:
                evidence_scores.append(0)
        
        # Overall assessment
        if evidence_scores:
            avg_score = np.mean(evidence_scores)
            
            if avg_score > 0.7:
                verdict['has_predictive_power'] = True
                verdict['confidence_level'] = 'high'
                verdict['recommendations'].append("Model shows strong predictive power - proceed with deployment")
            elif avg_score > 0.4:
                verdict['has_predictive_power'] = True
                verdict['confidence_level'] = 'medium'
                verdict['recommendations'].append("Model shows moderate predictive power - continue development")
            else:
                verdict['has_predictive_power'] = False
                verdict['confidence_level'] = 'low'
                verdict['recommendations'].append("Limited predictive power - need more data or features")
        
        # Specific recommendations
        if self.results.get('feature_stability'):
            stable_features = []
            for target, stability in self.results['feature_stability'].items():
                stable_features.extend(stability.get('stable_features', []))
            
            if stable_features:
                verdict['recommendations'].append(
                    f"Focus on stable features: {', '.join(set(stable_features)[:5])}"
                )
        
        # Data recommendations
        if len(evidence_scores) < 10:
            verdict['recommendations'].append("Collect more data for robust evaluation")
        
        return verdict
    
    def generate_report(self, output_file: str = 'predictability_report.html'):
        """Generate comprehensive HTML report"""
        logger.info(f"Generating predictability report: {output_file}")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Predictability Analysis</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    background-color: #1a1a1a;
                    color: #e9ecef;
                }
                h1, h2, h3 { color: #6f42c1; }
                .verdict {
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    background-color: #272b30;
                    border: 2px solid #6f42c1;
                }
                .positive { color: #10b981; }
                .negative { color: #ef4444; }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #52575c;
                    padding: 8px;
                    text-align: left;
                }
                th { background-color: #272b30; }
                .section {
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #272b30;
                    border-radius: 8px;
                }
            </style>
        </head>
        <body>
            <h1>Model Predictability Analysis Report</h1>
        """
        
        # Overall verdict
        verdict = self.results.get('overall_verdict', {})
        verdict_class = 'positive' if verdict.get('has_predictive_power') else 'negative'
        
        html_content += f"""
        <div class="verdict">
            <h2>Overall Verdict</h2>
            <p><strong>Predictive Power:</strong> 
                <span class="{verdict_class}">
                    {'YES' if verdict.get('has_predictive_power') else 'NO'}
                </span>
            </p>
            <p><strong>Confidence Level:</strong> {verdict.get('confidence_level', 'unknown').upper()}</p>
            
            <h3>Key Findings:</h3>
            <ul>
        """
        
        for finding in verdict.get('key_findings', []):
            html_content += f"<li>{finding}</li>"
        
        html_content += """
            </ul>
            
            <h3>Recommendations:</h3>
            <ul>
        """
        
        for rec in verdict.get('recommendations', []):
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
        </div>
        """
        
        # Statistical tests section
        html_content += """
        <div class="section">
            <h2>Statistical Tests</h2>
            <table>
                <tr>
                    <th>Target</th>
                    <th>Autocorrelation</th>
                    <th>Random Walk</th>
                    <th>Mean Reversion</th>
                    <th>Predictable?</th>
                </tr>
        """
        
        for target, tests in self.results.get('statistical_tests', {}).items():
            predictable = tests.get('is_predictable', {}).get('is_predictable', False)
            html_content += f"""
                <tr>
                    <td>{target}</td>
                    <td>{'Yes' if tests.get('autocorrelation', {}).get('has_autocorrelation') else 'No'}</td>
                    <td>{'No' if tests.get('random_walk', {}).get('is_random_walk') else 'Yes'}</td>
                    <td>{'Yes' if tests.get('mean_reversion', {}).get('has_mean_reversion') else 'No'}</td>
                    <td class="{'positive' if predictable else 'negative'}">
                        {'Yes' if predictable else 'No'}
                    </td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
        
        # Benchmark comparison
        html_content += """
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <table>
                <tr>
                    <th>Target</th>
                    <th>ML Model</th>
                    <th>Best Benchmark</th>
                    <th>Outperformance</th>
                    <th>Beats All?</th>
                </tr>
        """
        
        for target, bench in self.results.get('benchmark_comparison', {}).items():
            ml_acc = bench.get('results', {}).get('ml_model', {}).get('direction_accuracy', 0)
            best_bench = bench.get('best_benchmark', 'unknown')
            outperform = bench.get('ml_outperformance', 0)
            beats_all = bench.get('beats_all_benchmarks', False)
            
            html_content += f"""
                <tr>
                    <td>{target}</td>
                    <td>{ml_acc*100:.1f}%</td>
                    <td>{best_bench}</td>
                    <td class="{'positive' if outperform > 0 else 'negative'}">
                        {outperform*100:+.1f}%
                    </td>
                    <td class="{'positive' if beats_all else 'negative'}">
                        {'Yes' if beats_all else 'No'}
                    </td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
        
        # Economic significance
        html_content += """
        <div class="section">
            <h2>Economic Significance</h2>
            <table>
                <tr>
                    <th>Target</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Profitable?</th>
                </tr>
        """
        
        for target, econ in self.results.get('economic_significance', {}).items():
            sharpe = econ.get('sharpe_ratio', 0)
            dd = econ.get('max_drawdown', 0)
            win_rate = econ.get('profitable_trades_pct', 0.5)
            profitable = econ.get('is_profitable', False)
            
            html_content += f"""
                <tr>
                    <td>{target}</td>
                    <td class="{'positive' if sharpe > 0.5 else 'negative'}">
                        {sharpe:.2f}
                    </td>
                    <td>{dd*100:.1f}%</td>
                    <td>{win_rate*100:.1f}%</td>
                    <td class="{'positive' if profitable else 'negative'}">
                        {'Yes' if profitable else 'No'}
                    </td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        
        </body>
        </html>
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_file}")


def run_predictability_analysis():
    """Run comprehensive predictability analysis"""
    print("🔍 Comprehensive Model Predictability Analysis")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('data/processed/ml_dataset_complete.csv', 
                        index_col='date', parse_dates=True)
        print(f"✅ Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run create_dataset_with_targets.py first")
        return
    
    # Initialize analyzer
    analyzer = PredictabilityAnalyzer()
    
    # Define targets to analyze
    target_cols = ['return_1d_future', 'return_7d_future', 'return_30d_future']
    
    # Run analysis
    results = analyzer.analyze_predictability(df, target_cols, n_simulations=100)
    
    # Display results
    print("\n📊 PREDICTABILITY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Overall verdict
    verdict = results['overall_verdict']
    print(f"\n🎯 Overall Verdict:")
    print(f"   Has Predictive Power: {'YES' if verdict['has_predictive_power'] else 'NO'}")
    print(f"   Confidence Level: {verdict['confidence_level'].upper()}")
    
    print("\n📌 Key Findings:")
    for finding in verdict['key_findings']:
        print(f"   • {finding}")
    
    print("\n💡 Recommendations:")
    for rec in verdict['recommendations']:
        print(f"   • {rec}")
    
    # Generate detailed report
    analyzer.generate_report('predictability_analysis.html')
    print("\n📄 Detailed report saved to: predictability_analysis.html")
    
    return results


if __name__ == "__main__":
    results = run_predictability_analysis()
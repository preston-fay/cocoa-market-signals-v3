#!/usr/bin/env python3
"""
Regularized Ensemble Model with Overfitting Prevention
Implements multiple strategies to reduce overfitting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegularizedEnsemble:
    """
    Ensemble model with strong regularization to prevent overfitting
    """
    
    def __init__(self, 
                 n_features_to_select: int = 30,
                 use_feature_selection: bool = True,
                 use_pca: bool = False,
                 n_components: int = 20):
        
        self.n_features_to_select = n_features_to_select
        self.use_feature_selection = use_feature_selection
        self.use_pca = use_pca
        self.n_components = n_components
        
        # Initialize models with stronger regularization
        self.models = {
            'rf_regularized': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,  # Limit depth
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,   # Require more samples in leaves
                max_features='sqrt',  # Use fewer features per split
                random_state=42,
                n_jobs=-1
            ),
            
            'gb_regularized': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,  # Very shallow trees
                learning_rate=0.01,  # Slower learning
                subsample=0.8,  # Use subset of data
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            
            'xgb_regularized': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42
            ),
            
            'ridge_strong': Ridge(
                alpha=10.0,  # Strong regularization
                random_state=42
            ),
            
            'lasso': Lasso(
                alpha=0.01,
                random_state=42,
                max_iter=2000
            ),
            
            'elastic_net_strong': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            )
        }
        
        self.scalers = {}
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        self.model_weights = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select most important features to reduce overfitting
        """
        if not self.use_feature_selection:
            return X
            
        logger.info(f"Selecting top {self.n_features_to_select} features from {X.shape[1]}")
        
        # Method 1: Univariate feature selection
        selector = SelectKBest(f_regression, k=min(self.n_features_to_select, X.shape[1]))
        selector.fit(X.fillna(X.median()), y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Method 2: Tree-based feature importance
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(X.fillna(X.median()), y)
        
        # Get feature importances
        importances = rf_selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Combine both methods
        top_features_rf = X.columns[indices[:self.n_features_to_select]].tolist()
        
        # Union of both methods
        combined_features = list(set(self.selected_features + top_features_rf))[:self.n_features_to_select]
        self.selected_features = combined_features
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return X[self.selected_features]
    
    def apply_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        """
        if not self.use_pca:
            return X
            
        if self.pca is None:
            self.pca = PCA(n_components=min(self.n_components, X.shape[1]), random_state=42)
            X_pca = self.pca.fit_transform(X)
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA: {self.n_components} components explain {explained_var:.2%} variance")
        else:
            X_pca = self.pca.transform(X)
            
        return X_pca
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train ensemble with regularization and cross-validation
        """
        logger.info(f"Training regularized ensemble on {len(X_train)} samples")
        
        # Feature selection
        X_train_selected = self.select_features(X_train, y_train)
        
        # Handle missing values
        X_train_filled = X_train_selected.fillna(X_train_selected.median())
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train_filled)
        self.scalers['main'] = scaler
        
        # Apply PCA if enabled
        X_train_final = self.apply_pca(X_train_scaled)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        cv_scores = {}
        
        # Train each model
        for name, model in self.models.items():
            try:
                # Cross-validation
                scores = cross_val_score(model, X_train_final, y_train, 
                                       cv=tscv, scoring='neg_mean_squared_error')
                cv_score = -scores.mean()
                cv_scores[name] = cv_score
                
                # Train on full data
                model.fit(X_train_final, y_train)
                
                # Training score
                train_pred = model.predict(X_train_final)
                train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
                train_r2 = 1 - (np.sum((y_train - train_pred) ** 2) / 
                               np.sum((y_train - y_train.mean()) ** 2))
                
                results[name] = {
                    'cv_rmse': np.sqrt(cv_score),
                    'train_rmse': train_rmse,
                    'train_r2': train_r2
                }
                
                logger.info(f"{name} - CV RMSE: {np.sqrt(cv_score):.4f}, Train R²: {train_r2:.4f}")
                
            except Exception as e:
                logger.warning(f"Error training {name}: {e}")
                cv_scores[name] = float('inf')
        
        # Calculate model weights based on CV performance
        self._calculate_weights(cv_scores)
        
        return results
    
    def _calculate_weights(self, cv_scores: Dict[str, float]):
        """
        Calculate model weights based on cross-validation performance
        """
        # Convert scores to weights (inverse of error)
        weights = {}
        for name, score in cv_scores.items():
            if score != float('inf'):
                weights[name] = 1.0 / (score + 1e-6)
            else:
                weights[name] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Equal weights if all failed
            self.model_weights = {k: 1.0/len(self.models) for k in self.models.keys()}
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with weighted ensemble
        """
        # Apply same preprocessing
        if self.use_feature_selection and self.selected_features:
            X_test = X_test[self.selected_features]
        
        X_test_filled = X_test.fillna(X_test.median())
        X_test_scaled = self.scalers['main'].transform(X_test_filled)
        X_test_final = self.apply_pca(X_test_scaled)
        
        # Collect predictions
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test_final)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
        
        # Weighted average
        if not predictions:
            return np.zeros(len(X_test))
        
        weighted_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 0)
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance from tree-based models
        """
        if not self.selected_features:
            return {}
        
        importance_dict = {}
        
        # Get importance from tree-based models
        for name in ['rf_regularized', 'gb_regularized', 'xgb_regularized']:
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                importances = self.models[name].feature_importances_
                
                # Map back to original features
                if self.use_pca and self.pca is not None:
                    # For PCA, we can't directly map importance
                    continue
                else:
                    for i, feat in enumerate(self.selected_features[:len(importances)]):
                        if feat not in importance_dict:
                            importance_dict[feat] = 0
                        importance_dict[feat] += importances[i] * self.model_weights.get(name, 0)
        
        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict
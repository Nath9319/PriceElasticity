#!/usr/bin/env python3
"""
Enhanced Price Elasticity Models for Dual Price Prediction
===========================================================

This module implements advanced price elasticity models that predict two key price points:
1. Accurate Price: The price customers are most likely to accept (high probability)
2. Stretch Price: The maximum price for profit optimization (price ceiling)

Key Features:
- Customer-Product level modeling
- Dual price output system
- Advanced price elasticity calculations
- Risk-aware pricing strategies

Author: Enhanced for dual pricing requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import warnings
import joblib
import json
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import config_loader, logger


class DualPriceElasticityModel(BaseEstimator, ABC):
    """
    Abstract base class for dual price elasticity models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config_loader if config is None else config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance_ = {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'DualPriceElasticityModel':
        """Fit the dual price model"""
        pass
    
    @abstractmethod
    def predict_dual_prices(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both accurate and stretch prices
        
        Returns:
            Tuple of (accurate_prices, stretch_prices)
        """
        pass
    
    def predict_win_probability(self, X: pd.DataFrame, price: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict win probability for given features and prices"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use current offered price if not provided
        if price is None:
            price = X.get('Offered_Price', X.get('Net_Price', np.zeros(len(X))))
        
        # Calculate price elasticity features
        X_pred = self._prepare_prediction_features(X, price)
        
        # Use win probability model if available
        if 'win_probability' in self.models:
            return self.models['win_probability'].predict_proba(X_pred)[:, 1]
        else:
            # Fallback elasticity calculation
            return self._calculate_win_probability_elasticity(X_pred, price)
    
    def _prepare_prediction_features(self, X: pd.DataFrame, price: np.ndarray) -> pd.DataFrame:
        """Prepare features for prediction including price elasticity"""
        X_pred = X.copy()
        
        # Add price elasticity features
        if 'List_Price' in X.columns:
            X_pred['price_elasticity'] = (X['List_Price'] - price) / X['List_Price']
            X_pred['discount_depth'] = np.maximum(0, (X['List_Price'] - price) / X['List_Price'])
        
        # Customer value features
        if 'Customer_Segment' in X.columns:
            segment_sensitivity = {
                'Enterprise': 0.3,   # Less price sensitive
                'Strategic': 0.2,    # Least price sensitive  
                'Mid-Market': 0.5,   # Moderate sensitivity
                'SMB': 0.7          # Most price sensitive
            }
            X_pred['segment_sensitivity'] = X['Customer_Segment'].map(segment_sensitivity).fillna(0.5)
        
        # Product category effects
        if 'Product_Category' in X.columns:
            category_elasticity = {
                'Software': -0.8,    # Less elastic (sticky)
                'Services': -1.0,    # Moderate elasticity
                'Hardware': -1.2,    # More elastic (commoditized)
                'Support': -0.6      # Less elastic (necessary)
            }
            X_pred['category_elasticity'] = X['Product_Category'].map(category_elasticity).fillna(-1.0)
        
        # Competition impact
        if 'Competition_Status' in X.columns:
            competition_impact = {
                'None': 0.1,
                'Low': 0.3,
                'Medium': 0.6,
                'High': 0.9
            }
            X_pred['competition_impact'] = X['Competition_Status'].map(competition_impact).fillna(0.5)
        
        return X_pred
    
    def _calculate_win_probability_elasticity(self, X: pd.DataFrame, price: np.ndarray) -> np.ndarray:
        """Calculate win probability using price elasticity principles"""
        base_probability = 0.5  # Baseline 50% win rate
        
        # Price elasticity effect
        if 'price_elasticity' in X.columns:
            elasticity_effect = np.where(
                X['price_elasticity'] > 0,  # Discount given
                X['price_elasticity'] * 0.8,  # Positive effect from discounts
                X['price_elasticity'] * 1.2   # Negative effect from premiums
            )
        else:
            elasticity_effect = 0
            
        # Segment sensitivity
        segment_effect = X.get('segment_sensitivity', 0.5) * X.get('price_elasticity', 0)
        
        # Category elasticity
        category_effect = X.get('category_elasticity', -1.0) * X.get('discount_depth', 0) * 0.3
        
        # Competition effect
        competition_effect = -X.get('competition_impact', 0.5) * 0.2
        
        # Combine effects
        total_effect = elasticity_effect + segment_effect + category_effect + competition_effect
        win_prob = base_probability + total_effect
        
        # Bound between 0.05 and 0.95
        return np.clip(win_prob, 0.05, 0.95)


class EnhancedHierarchicalBayesianModel(DualPriceElasticityModel):
    """
    Enhanced Hierarchical Bayesian Model with dual price outputs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.segment_effects = {}
        self.price_models = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnhancedHierarchicalBayesianModel':
        """Fit the hierarchical Bayesian model with dual price prediction"""
        self.logger.info("Training Enhanced Hierarchical Bayesian Model...")
        
        # Prepare features
        X_processed = self._preprocess_features(X)
        
        # Fit win probability model
        self._fit_win_probability_model(X_processed, y)
        
        # Fit price prediction models
        self._fit_price_models(X_processed, X)
        
        # Calculate segment effects
        self._calculate_segment_effects(X_processed, y)
        
        self.is_fitted = True
        
        # Calculate performance metrics
        y_pred_prob = self.predict_win_probability(X)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        performance = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'precision': precision_score(y, y_pred_binary, zero_division=0),
            'recall': recall_score(y, y_pred_binary, zero_division=0),
            'f1': f1_score(y, y_pred_binary, zero_division=0),
            'auc': roc_auc_score(y, y_pred_prob),
            'log_loss': log_loss(y, y_pred_prob)
        }
        
        self.performance_ = performance
        self.logger.info(f"Enhanced Hierarchical Bayesian model trained. AUC: {performance['auc']:.3f}")
        
        return self
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for modeling"""
        X_proc = X.copy()
        
        # Encode categorical variables
        categorical_cols = X_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_proc[col] = self.encoders[col].fit_transform(X_proc[col].fillna('Unknown'))
            else:
                # Transform using existing encoder
                unknown_mask = ~X_proc[col].isin(self.encoders[col].classes_)
                X_proc[col] = X_proc[col].fillna('Unknown')
                X_proc.loc[unknown_mask, col] = 'Unknown'
                
                # Add Unknown to encoder if not present
                if 'Unknown' not in self.encoders[col].classes_:
                    # Extend the encoder classes
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'Unknown')
                
                X_proc[col] = self.encoders[col].transform(X_proc[col])
        
        # Scale numerical features
        numerical_cols = X_proc.select_dtypes(include=[np.number]).columns
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_proc[numerical_cols] = self.scalers['scaler'].fit_transform(X_proc[numerical_cols].fillna(0))
        else:
            X_proc[numerical_cols] = self.scalers['scaler'].transform(X_proc[numerical_cols].fillna(0))
        
        return X_proc
    
    def _fit_win_probability_model(self, X_processed: pd.DataFrame, y: pd.Series):
        """Fit the win probability prediction model"""
        # Use ensemble of models for robustness
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            try:
                model.fit(X_processed, y)
                y_pred = model.predict_proba(X_processed)[:, 1]
                score = roc_auc_score(y, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                self.logger.warning(f"Failed to train {name}: {e}")
                continue
        
        self.models['win_probability'] = best_model
        
    def _fit_price_models(self, X_processed: pd.DataFrame, X_original: pd.DataFrame):
        """Fit models to predict accurate and stretch prices"""
        # Extract price targets
        if 'Net_Price' in X_original.columns and 'List_Price' in X_original.columns:
            # Target 1: Accurate Price (conservative, high win probability)
            # Use actual winning prices as targets for accurate pricing
            accurate_price_target = X_original['Net_Price'] * 0.95  # Slightly lower for safety
            
            # Target 2: Stretch Price (aggressive, maximum revenue)
            # Use list price as upper bound, but moderate based on elasticity
            stretch_price_target = X_original['List_Price'] * 0.85  # 85% of list price
            
            # Fit accurate price model (conservative)
            self.price_models['accurate'] = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
            self.price_models['accurate'].fit(X_processed, accurate_price_target)
            
            # Fit stretch price model (aggressive)
            self.price_models['stretch'] = GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            )
            self.price_models['stretch'].fit(X_processed, stretch_price_target)
            
    def _calculate_segment_effects(self, X_processed: pd.DataFrame, y: pd.Series):
        """Calculate hierarchical effects by segment"""
        if 'Customer_Segment' in X_processed.columns:
            segment_col = X_processed['Customer_Segment']
            unique_segments = np.unique(segment_col)
            
            for segment in unique_segments:
                segment_mask = segment_col == segment
                if segment_mask.sum() > 0:
                    segment_win_rate = y[segment_mask].mean()
                    overall_win_rate = y.mean()
                    self.segment_effects[f'segment_{segment}'] = segment_win_rate - overall_win_rate
        else:
            self.segment_effects['default'] = 0.0
            
    def predict_dual_prices(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both accurate and stretch prices"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = self._preprocess_features(X)
        
        # Predict accurate prices (conservative)
        if 'accurate' in self.price_models:
            accurate_prices = self.price_models['accurate'].predict(X_processed)
        else:
            # Fallback: Use 85% of list price or net price
            accurate_prices = X.get('Net_Price', X.get('List_Price', 1000)) * 0.85
            
        # Predict stretch prices (aggressive)  
        if 'stretch' in self.price_models:
            stretch_prices = self.price_models['stretch'].predict(X_processed)
        else:
            # Fallback: Use 95% of list price
            stretch_prices = X.get('List_Price', X.get('Net_Price', 1000)) * 0.95
            
        # Apply business rules
        accurate_prices = np.maximum(accurate_prices, 0)  # No negative prices
        stretch_prices = np.maximum(stretch_prices, accurate_prices * 1.05)  # Stretch >= Accurate
        
        # Apply segment-specific adjustments
        if 'Customer_Segment' in X.columns:
            for i, segment in enumerate(X['Customer_Segment']):
                # Segment sensitivity adjustments
                if segment == 'Enterprise':
                    stretch_prices[i] *= 1.1  # Can handle higher prices
                elif segment == 'SMB':
                    accurate_prices[i] *= 0.95  # More price sensitive
                elif segment == 'Strategic':
                    stretch_prices[i] *= 1.15  # Premium customers
                    
        return accurate_prices, stretch_prices


class EnhancedGraphNeuralNetworkModel(DualPriceElasticityModel):
    """
    Enhanced Graph Neural Network Model with dual price outputs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.graph_features = {}
        self.network_effects = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnhancedGraphNeuralNetworkModel':
        """Fit the enhanced GNN model"""
        self.logger.info("Training Enhanced Graph Neural Network Model...")
        
        # Create graph features
        X_graph = self._create_graph_features(X)
        
        # Preprocess features
        X_processed = self._preprocess_features(X_graph)
        
        # Fit models
        self._fit_models(X_processed, y, X)
        
        self.is_fitted = True
        
        # Calculate performance
        y_pred_prob = self.predict_win_probability(X)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        performance = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'auc': roc_auc_score(y, y_pred_prob)
        }
        
        self.performance_ = performance
        self.logger.info(f"Enhanced GNN model trained. AUC: {performance['auc']:.3f}")
        
        return self
        
    def _create_graph_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create graph-based network features"""
        X_graph = X.copy()
        
        # Customer-Product interaction network
        if 'Customer_ID' in X.columns and 'Product_ID' in X.columns:
            # Customer centrality (how connected is this customer)
            customer_counts = X['Customer_ID'].value_counts()
            X_graph['customer_centrality'] = X['Customer_ID'].map(customer_counts).fillna(1)
            
            # Product popularity (how many customers bought this product)
            product_counts = X['Product_ID'].value_counts()
            X_graph['product_popularity'] = X['Product_ID'].map(product_counts).fillna(1)
            
            # Customer-Product affinity (historical interaction strength)
            customer_product_counts = X.groupby(['Customer_ID', 'Product_ID']).size()
            X_graph['cp_affinity'] = X.apply(
                lambda row: customer_product_counts.get((row['Customer_ID'], row['Product_ID']), 0),
                axis=1
            )
            
        # Segment network effects
        if 'Customer_Segment' in X.columns:
            segment_sizes = X['Customer_Segment'].value_counts()
            X_graph['segment_network_size'] = X['Customer_Segment'].map(segment_sizes).fillna(1)
            
        # Category network effects
        if 'Product_Category' in X.columns:
            category_sizes = X['Product_Category'].value_counts()
            X_graph['category_network_size'] = X['Product_Category'].map(category_sizes).fillna(1)
            
        return X_graph
        
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features including graph features"""
        X_proc = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_proc[col] = self.encoders[col].fit_transform(X_proc[col].fillna('Unknown'))
            else:
                # Handle unseen categories
                unknown_mask = ~X_proc[col].isin(self.encoders[col].classes_)
                X_proc[col] = X_proc[col].fillna('Unknown')
                X_proc.loc[unknown_mask, col] = 'Unknown'
                
                if 'Unknown' not in self.encoders[col].classes_:
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'Unknown')
                    
                X_proc[col] = self.encoders[col].transform(X_proc[col])
        
        # Scale numerical features
        numerical_cols = X_proc.select_dtypes(include=[np.number]).columns
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_proc[numerical_cols] = self.scalers['scaler'].fit_transform(X_proc[numerical_cols].fillna(0))
        else:
            X_proc[numerical_cols] = self.scalers['scaler'].transform(X_proc[numerical_cols].fillna(0))
            
        return X_proc
        
    def _fit_models(self, X_processed: pd.DataFrame, y: pd.Series, X_original: pd.DataFrame):
        """Fit win probability and price models"""
        # Win probability model
        self.models['win_probability'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.models['win_probability'].fit(X_processed, y)
        
        # Price models
        if 'Net_Price' in X_original.columns and 'List_Price' in X_original.columns:
            # Accurate price model (conservative)
            accurate_target = X_original['Net_Price'] * 0.92
            self.models['accurate_price'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            self.models['accurate_price'].fit(X_processed, accurate_target)
            
            # Stretch price model (optimistic)
            stretch_target = X_original['List_Price'] * 0.88
            self.models['stretch_price'] = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
            self.models['stretch_price'].fit(X_processed, stretch_target)
            
    def predict_dual_prices(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both accurate and stretch prices"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_graph = self._create_graph_features(X)
        X_processed = self._preprocess_features(X_graph)
        
        # Predict prices
        if 'accurate_price' in self.models and 'stretch_price' in self.models:
            accurate_prices = self.models['accurate_price'].predict(X_processed)
            stretch_prices = self.models['stretch_price'].predict(X_processed)
        else:
            # Fallback
            accurate_prices = X.get('Net_Price', 1000) * 0.9
            stretch_prices = X.get('List_Price', 1200) * 0.9
            
        # Apply network effects
        network_multiplier = 1 + (X_graph.get('customer_centrality', 1) / 100)
        stretch_prices = stretch_prices * network_multiplier
        
        # Ensure business rules
        accurate_prices = np.maximum(accurate_prices, 0)
        stretch_prices = np.maximum(stretch_prices, accurate_prices * 1.05)
        
        return accurate_prices, stretch_prices


class EnhancedEnsembleModel(DualPriceElasticityModel):
    """
    Ensemble model combining Hierarchical Bayesian and Graph Neural Network models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.base_models = {}
        self.ensemble_weights = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnhancedEnsembleModel':
        """Fit the ensemble model"""
        self.logger.info("Training Enhanced Ensemble Model...")
        
        # Initialize base models
        self.base_models['hierarchical_bayesian'] = EnhancedHierarchicalBayesianModel(self.config)
        self.base_models['graph_neural_network'] = EnhancedGraphNeuralNetworkModel(self.config)
        
        # Fit base models
        model_scores = {}
        for name, model in self.base_models.items():
            try:
                model.fit(X, y)
                
                # Get performance score
                y_pred_prob = model.predict_win_probability(X)
                score = roc_auc_score(y, y_pred_prob)
                model_scores[name] = score
                
                self.logger.info(f"{name} trained with AUC: {score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                model_scores[name] = 0.0
                
        # Calculate ensemble weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            for name, score in model_scores.items():
                self.ensemble_weights[name] = score / total_score
        else:
            # Equal weights as fallback
            num_models = len(self.base_models)
            for name in self.base_models.keys():
                self.ensemble_weights[name] = 1.0 / num_models
                
        self.is_fitted = True
        
        # Calculate ensemble performance
        y_pred_prob = self.predict_win_probability(X)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        performance = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'precision': precision_score(y, y_pred_binary, zero_division=0),
            'recall': recall_score(y, y_pred_binary, zero_division=0),
            'f1': f1_score(y, y_pred_binary, zero_division=0),
            'auc': roc_auc_score(y, y_pred_prob)
        }
        
        self.performance_ = performance
        self.logger.info(f"Enhanced Ensemble model trained. AUC: {performance['auc']:.3f}")
        
        return self
        
    def predict_win_probability(self, X: pd.DataFrame, price: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict win probability using ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = []
        weights = []
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict_win_probability(X, price)
                predictions.append(pred)
                weights.append(self.ensemble_weights.get(name, 0))
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from {name}: {e}")
                continue
                
        if not predictions:
            # Fallback to simple elasticity calculation
            return np.full(len(X), 0.5)
            
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
            
        return ensemble_pred
        
    def predict_dual_prices(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both accurate and stretch prices using ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        accurate_predictions = []
        stretch_predictions = []
        weights = []
        
        for name, model in self.base_models.items():
            try:
                acc_pred, str_pred = model.predict_dual_prices(X)
                accurate_predictions.append(acc_pred)
                stretch_predictions.append(str_pred)
                weights.append(self.ensemble_weights.get(name, 0))
            except Exception as e:
                self.logger.warning(f"Failed to get price prediction from {name}: {e}")
                continue
                
        if not accurate_predictions:
            # Fallback pricing
            fallback_accurate = X.get('Net_Price', 1000) * 0.9
            fallback_stretch = X.get('List_Price', 1200) * 0.9
            return fallback_accurate, fallback_stretch
            
        # Weighted ensemble
        accurate_predictions = np.array(accurate_predictions)
        stretch_predictions = np.array(stretch_predictions)
        weights = np.array(weights)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
            ensemble_accurate = np.average(accurate_predictions, axis=0, weights=weights)
            ensemble_stretch = np.average(stretch_predictions, axis=0, weights=weights)
        else:
            ensemble_accurate = np.mean(accurate_predictions, axis=0)
            ensemble_stretch = np.mean(stretch_predictions, axis=0)
            
        return ensemble_accurate, ensemble_stretch


class PriceOptimizationEngine:
    """
    Price optimization engine for finding optimal accurate and stretch prices
    """
    
    def __init__(self, model: DualPriceElasticityModel):
        self.model = model
        self.logger = logger
        
    def optimize_prices_for_customer_product(self, 
                                           customer_data: pd.Series,
                                           target_win_probability: Dict[str, float] = None) -> Dict[str, float]:
        """
        Optimize prices for a specific customer-product combination
        
        Args:
            customer_data: Single row of customer-product data
            target_win_probability: Dict with 'accurate' and 'stretch' target probabilities
            
        Returns:
            Dict with optimized accurate and stretch prices
        """
        if target_win_probability is None:
            target_win_probability = {'accurate': 0.75, 'stretch': 0.25}
            
        # Convert series to dataframe
        customer_df = pd.DataFrame([customer_data])
        
        # Get initial price predictions
        accurate_price, stretch_price = self.model.predict_dual_prices(customer_df)
        
        # Optimize accurate price for high win probability
        def objective_accurate(price):
            win_prob = self.model.predict_win_probability(customer_df, np.array([price[0]]))
            return (win_prob[0] - target_win_probability['accurate']) ** 2
            
        # Optimize stretch price for maximum revenue while maintaining minimum win probability
        def objective_stretch(price):
            win_prob = self.model.predict_win_probability(customer_df, np.array([price[0]]))
            if win_prob[0] < target_win_probability['stretch']:
                return 1e6  # Penalty for too low win probability
            return -price[0]  # Negative because we want to maximize price
            
        # Price bounds
        list_price = customer_data.get('List_Price', 1000)
        min_price = list_price * 0.5  # 50% minimum discount
        max_price = list_price * 1.1   # 10% premium maximum
        
        # Optimize accurate price
        try:
            result_accurate = minimize(
                objective_accurate,
                x0=[accurate_price[0]],
                bounds=[(min_price, max_price)],
                method='L-BFGS-B'
            )
            optimized_accurate = result_accurate.x[0]
        except:
            optimized_accurate = accurate_price[0]
            
        # Optimize stretch price
        try:
            result_stretch = minimize(
                objective_stretch,
                x0=[stretch_price[0]],
                bounds=[(optimized_accurate, max_price)],
                method='L-BFGS-B'
            )
            optimized_stretch = result_stretch.x[0]
        except:
            optimized_stretch = stretch_price[0]
            
        # Final win probabilities
        final_accurate_prob = self.model.predict_win_probability(customer_df, np.array([optimized_accurate]))[0]
        final_stretch_prob = self.model.predict_win_probability(customer_df, np.array([optimized_stretch]))[0]
        
        return {
            'accurate_price': optimized_accurate,
            'stretch_price': optimized_stretch,
            'accurate_win_probability': final_accurate_prob,
            'stretch_win_probability': final_stretch_prob,
            'expected_revenue_accurate': optimized_accurate * final_accurate_prob,
            'expected_revenue_stretch': optimized_stretch * final_stretch_prob
        }

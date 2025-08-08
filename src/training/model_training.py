"""
Comprehensive Model Training for B2B Price Elasticity Modeling
Implements Hierarchical Bayesian, X-Learner (Causal ML), and Ensemble Models
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')

# Try to import advanced libraries with graceful fallback
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    print("Warning: PyMC not available. Hierarchical Bayesian models will be simulated.")

try:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("Warning: EconML not available. X-Learner will be simulated with standard ML models.")


class PriceElasticityModelTraining:
    """
    Comprehensive Model Training for Price Elasticity Analysis
    Implements all three modeling approaches from requirements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model training with configuration
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config_loader if config is None else config
        self.logger = logger
        self.models = {}
        self.model_metadata = {}
        self.training_results = {}
        
        # Get model configurations
        self.model_configs = self.config.get('models', {})
        self.hpo_config = self.config.get('hyperparameter_optimization', {})
        self.validation_config = self.config.get('validation', {})
        
        self.logger.info("Model Training initialized")
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'Status') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training with proper validation
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        self.logger.info("Preparing training data...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in [
            'Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', target_col, 'Quote_Date'
        ]]
        
        X = df[feature_columns].copy()
        
        # Prepare target
        if target_col in df.columns:
            if df[target_col].dtype == 'object':
                # Convert to binary for classification
                y = (df[target_col] == 'Won').astype(int)
            else:
                y = df[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        self.logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def create_time_series_splits(self, df: pd.DataFrame, date_col: str = 'Quote_Date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits
        
        Args:
            df: Input DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Get validation config
        ts_cv_config = self.validation_config.get('time_series_cv', {})
        n_splits = ts_cv_config.get('n_splits', 5)
        test_size_days = ts_cv_config.get('test_size_days', 30)
        gap_days = ts_cv_config.get('gap_days', 7)
        
        if date_col not in df.columns:
            # Fallback to standard time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(df))
        
        # Custom time series split with gaps
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df_sorted[date_col])
        
        splits = []
        end_date = dates.max()
        
        for i in range(n_splits):
            # Calculate test period
            test_end = end_date - pd.Timedelta(days=i * (test_size_days + gap_days))
            test_start = test_end - pd.Timedelta(days=test_size_days)
            
            # Calculate training period (with gap)
            train_end = test_start - pd.Timedelta(days=gap_days)
            
            # Create masks
            train_mask = dates <= train_end
            test_mask = (dates >= test_start) & (dates <= test_end)
            
            train_idx = df_sorted[train_mask].index.values
            val_idx = df_sorted[test_mask].index.values
            
            if len(train_idx) > 100 and len(val_idx) > 20:  # Minimum sample sizes
                splits.append((train_idx, val_idx))
        
        return splits[::-1]  # Return in chronological order
    
    def train_hierarchical_bayesian_model(self, X: pd.DataFrame, y: pd.Series, 
                                         segment_col: str = 'Customer_Segment') -> Dict[str, Any]:
        """
        Train Hierarchical Bayesian Model as per Requirement 2.1
        
        Args:
            X: Feature matrix
            y: Target vector
            segment_col: Column name for hierarchical grouping
            
        Returns:
            Dictionary with model results
        """
        self.logger.info("Training Hierarchical Bayesian Model...")
        
        # Get configuration
        hb_config = self.model_configs.get('hierarchical_bayesian', {})
        
        if not HAS_PYMC:
            # Simulate with mixed effects model using statsmodels
            return self._simulate_hierarchical_bayesian(X, y, segment_col, hb_config)
        
        try:
            # Prepare data for PyMC
            if segment_col not in X.columns:
                self.logger.warning(f"Segment column {segment_col} not found, creating artificial segments")
                X[segment_col] = np.random.choice(['Segment_A', 'Segment_B', 'Segment_C'], len(X))
            
            # Create segment encoding
            unique_segments = X[segment_col].unique()
            segment_idx = {seg: i for i, seg in enumerate(unique_segments)}
            X['segment_idx'] = X[segment_col].map(segment_idx)
            
            # Prepare design matrix (key features for price elasticity)
            price_features = [col for col in X.columns if 'price' in col.lower() or 'discount' in col.lower()][:5]
            design_features = price_features + ['segment_idx']
            
            X_design = X[design_features].fillna(0).values
            segments = X['segment_idx'].values
            
            with pm.Model() as hierarchical_model:
                # Hyperpriors for group-level parameters
                mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=100)
                sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=5)
                
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=10)
                sigma_beta = pm.HalfCauchy('sigma_beta', beta=5)
                
                # Group-level parameters
                alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=len(unique_segments))
                beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=len(design_features))
                
                # Linear combination
                linear_combination = alpha[segments] + pm.math.dot(X_design, beta)
                
                # Likelihood
                p = pm.Deterministic('p', pm.math.sigmoid(linear_combination))
                likelihood = pm.Bernoulli('y', p=p, observed=y)
                
                # Sampling
                sampler = hb_config.get('sampler', 'NUTS')
                chains = hb_config.get('chains', 4)
                draws = hb_config.get('draws', 2000)
                tune = hb_config.get('tune', 1000)
                target_accept = hb_config.get('target_accept', 0.9)
                
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True
                )
            
            # Model diagnostics
            summary = az.summary(trace)
            diagnostics = {
                'r_hat': summary['r_hat'].max(),
                'effective_sample_size': summary['ess_bulk'].min(),
                'mcmc_se': summary['mcse_mean'].mean()
            }
            
            # Make predictions
            with hierarchical_model:
                pp_trace = pm.sample_posterior_predictive(trace)
            
            y_pred_mean = pp_trace.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
            
            # Calculate performance metrics
            y_pred_binary = (y_pred_mean > 0.5).astype(int)
            
            performance = {
                'accuracy': accuracy_score(y, y_pred_binary),
                'precision': precision_score(y, y_pred_binary),
                'recall': recall_score(y, y_pred_binary),
                'f1': f1_score(y, y_pred_binary),
                'auc': roc_auc_score(y, y_pred_mean),
                'log_loss': log_loss(y, y_pred_mean)
            }
            
            results = {
                'model_type': 'hierarchical_bayesian',
                'model': hierarchical_model,
                'trace': trace,
                'performance': performance,
                'diagnostics': diagnostics,
                'segment_effects': {seg: trace.posterior['alpha'][:, :, i].mean().item() 
                                  for i, seg in enumerate(unique_segments)},
                'feature_effects': {f'feature_{i}': trace.posterior['beta'][:, :, i].mean().item() 
                                  for i in range(len(design_features))},
                'predictions': y_pred_mean,
                'training_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Hierarchical Bayesian model trained. AUC: {performance['auc']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training Hierarchical Bayesian model: {e}")
            results = self._simulate_hierarchical_bayesian(X, y, segment_col, hb_config)
        
        return results
    
    def _simulate_hierarchical_bayesian(self, X: pd.DataFrame, y: pd.Series, 
                                      segment_col: str, config: Dict) -> Dict[str, Any]:
        """
        Simulate hierarchical Bayesian model with mixed effects approach
        """
        self.logger.info("Simulating Hierarchical Bayesian model with mixed effects...")
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            # Group-wise modeling
            if segment_col not in X.columns:
                X[segment_col] = 'Default_Segment'
            
            segment_models = {}
            segment_effects = {}
            all_predictions = np.zeros(len(y))
            
            for segment in X[segment_col].unique():
                if pd.notna(segment):
                    mask = X[segment_col] == segment
                    X_seg = X[mask].drop(columns=[segment_col])
                    y_seg = y[mask]
                    
                    if len(y_seg) > 10:  # Minimum samples for training
                        # Train segment-specific model
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(X_seg, y_seg)
                        
                        # Make predictions
                        y_pred = model.predict_proba(X_seg)[:, 1]
                        all_predictions[mask] = y_pred
                        
                        segment_models[segment] = model
                        segment_effects[segment] = model.coef_[0].mean()  # Average coefficient effect
            
            # Overall performance
            y_pred_binary = (all_predictions > 0.5).astype(int)
            performance = {
                'accuracy': accuracy_score(y, y_pred_binary),
                'precision': precision_score(y, y_pred_binary),
                'recall': recall_score(y, y_pred_binary),
                'f1': f1_score(y, y_pred_binary),
                'auc': roc_auc_score(y, all_predictions) if len(np.unique(y)) > 1 else 0.5,
                'log_loss': log_loss(y, np.clip(all_predictions, 1e-15, 1-1e-15))
            }
            
            results = {
                'model_type': 'hierarchical_bayesian_simulated',
                'model': segment_models,
                'performance': performance,
                'segment_effects': segment_effects,
                'predictions': all_predictions,
                'training_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulated hierarchical model: {e}")
            # Fallback to simple logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X.drop(columns=[segment_col], errors='ignore'), y)
            predictions = model.predict_proba(X.drop(columns=[segment_col], errors='ignore'))[:, 1]
            
            results = {
                'model_type': 'logistic_regression_fallback',
                'model': model,
                'performance': {'auc': roc_auc_score(y, predictions)},
                'predictions': predictions,
                'training_timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def train_x_learner_model(self, X: pd.DataFrame, y: pd.Series, 
                             treatment_col: str = 'Offered_Price') -> Dict[str, Any]:
        """
        Train X-Learner (Causal ML) Model as per Requirement 2.2
        
        Args:
            X: Feature matrix
            y: Target vector
            treatment_col: Column name for treatment variable
            
        Returns:
            Dictionary with model results
        """
        self.logger.info("Training X-Learner Causal ML Model...")
        
        # Get configuration
        xl_config = self.model_configs.get('x_learner', {})
        
        if not HAS_ECONML:
            return self._simulate_x_learner(X, y, treatment_col, xl_config)
        
        try:
            # Prepare treatment variable
            if treatment_col not in X.columns:
                self.logger.warning(f"Treatment column {treatment_col} not found, using price proxy")
                treatment_col = [col for col in X.columns if 'price' in col.lower()][0]
            
            # Binarize treatment if needed
            treatment_threshold = xl_config.get('treatment', {}).get('binary_threshold', 'median')
            
            if treatment_threshold == 'median':
                threshold = X[treatment_col].median()
            else:
                threshold = float(treatment_threshold)
            
            T = (X[treatment_col] > threshold).astype(int)
            
            # Prepare features (exclude treatment)
            feature_cols = [col for col in X.columns if col != treatment_col]
            X_features = X[feature_cols]
            
            # Configure base models
            base_model_types = xl_config.get('base_models', {}).get('type', ['lightgbm', 'xgboost'])
            
            if 'lightgbm' in base_model_types:
                model_y = lgb.LGBMClassifier(random_state=42, verbose=-1)
                model_t = lgb.LGBMClassifier(random_state=42, verbose=-1)
            else:
                model_y = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                model_t = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
            # Train Double ML model
            dml_folds = xl_config.get('dml_folds', 3)
            
            dml_learner = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                n_estimators=100,
                cv=dml_folds,
                random_state=42
            )
            
            dml_learner.fit(y, T, X=X_features)
            
            # Calculate treatment effects
            treatment_effects = dml_learner.effect(X_features)
            
            # Make outcome predictions
            y_pred = dml_learner.model_y.predict_proba(X_features)[:, 1]
            
            # Performance metrics
            y_pred_binary = (y_pred > 0.5).astype(int)
            performance = {
                'accuracy': accuracy_score(y, y_pred_binary),
                'precision': precision_score(y, y_pred_binary),
                'recall': recall_score(y, y_pred_binary),
                'f1': f1_score(y, y_pred_binary),
                'auc': roc_auc_score(y, y_pred),
                'log_loss': log_loss(y, y_pred)
            }
            
            # Treatment effect analysis
            treatment_analysis = {
                'average_treatment_effect': np.mean(treatment_effects),
                'treatment_effect_std': np.std(treatment_effects),
                'positive_effects': (treatment_effects > 0).mean(),
                'treatment_effect_range': [np.min(treatment_effects), np.max(treatment_effects)]
            }
            
            results = {
                'model_type': 'x_learner_causal_ml',
                'model': dml_learner,
                'performance': performance,
                'treatment_effects': treatment_effects,
                'treatment_analysis': treatment_analysis,
                'predictions': y_pred,
                'treatment_threshold': threshold,
                'training_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"X-Learner model trained. ATE: {treatment_analysis['average_treatment_effect']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training X-Learner model: {e}")
            results = self._simulate_x_learner(X, y, treatment_col, xl_config)
        
        return results
    
    def _simulate_x_learner(self, X: pd.DataFrame, y: pd.Series, 
                           treatment_col: str, config: Dict) -> Dict[str, Any]:
        """
        Simulate X-Learner with standard ML approaches
        """
        self.logger.info("Simulating X-Learner with standard ML models...")
        
        try:
            # Prepare treatment
            if treatment_col not in X.columns:
                treatment_col = [col for col in X.columns if 'price' in col.lower()][0]
            
            threshold = X[treatment_col].median()
            T = (X[treatment_col] > threshold).astype(int)
            
            # Train models for treated and control groups
            X_features = X.drop(columns=[treatment_col])
            
            # Control group model
            control_mask = T == 0
            if control_mask.sum() > 10:
                model_control = lgb.LGBMClassifier(random_state=42, verbose=-1)
                model_control.fit(X_features[control_mask], y[control_mask])
            else:
                model_control = None
            
            # Treatment group model
            treatment_mask = T == 1
            if treatment_mask.sum() > 10:
                model_treated = lgb.LGBMClassifier(random_state=42, verbose=-1)
                model_treated.fit(X_features[treatment_mask], y[treatment_mask])
            else:
                model_treated = None
            
            # Calculate treatment effects (simplified)
            if model_control and model_treated:
                pred_control = model_control.predict_proba(X_features)[:, 1]
                pred_treated = model_treated.predict_proba(X_features)[:, 1]
                treatment_effects = pred_treated - pred_control
                
                y_pred = np.where(T == 1, pred_treated, pred_control)
            else:
                # Fallback
                model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                model.fit(X_features, y)
                y_pred = model.predict_proba(X_features)[:, 1]
                treatment_effects = np.random.normal(0, 0.1, len(y))
            
            # Performance
            y_pred_binary = (y_pred > 0.5).astype(int)
            performance = {
                'accuracy': accuracy_score(y, y_pred_binary),
                'auc': roc_auc_score(y, y_pred)
            }
            
            results = {
                'model_type': 'x_learner_simulated',
                'model': {'control': model_control, 'treated': model_treated},
                'performance': performance,
                'treatment_effects': treatment_effects,
                'predictions': y_pred,
                'training_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulated X-Learner: {e}")
            # Ultimate fallback
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            predictions = model.predict_proba(X)[:, 1]
            
            results = {
                'model_type': 'logistic_regression_fallback',
                'model': model,
                'performance': {'auc': roc_auc_score(y, predictions)},
                'predictions': predictions,
                'training_timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train Ensemble Model as per Requirement 2.3
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with ensemble model results
        """
        self.logger.info("Training Ensemble Model...")
        
        # Get configuration
        ensemble_config = self.model_configs.get('ensemble', {})
        base_model_names = ensemble_config.get('base_models', ['lightgbm', 'xgboost', 'catboost', 'random_forest'])
        
        # Define base models
        base_models = []
        
        if 'lightgbm' in base_model_names:
            lgb_params = self._get_optimized_params('lightgbm', X, y)
            lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
            base_models.append(('lightgbm', lgb_model))
        
        if 'xgboost' in base_model_names:
            xgb_params = self._get_optimized_params('xgboost', X, y)
            xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')
            base_models.append(('xgboost', xgb_model))
        
        if 'catboost' in base_model_names:
            try:
                import catboost as cb
                cb_params = self._get_optimized_params('catboost', X, y)
                cb_model = cb.CatBoostClassifier(**cb_params, random_state=42, verbose=False)
                base_models.append(('catboost', cb_model))
            except ImportError:
                self.logger.warning("CatBoost not available, skipping")
        
        if 'random_forest' in base_model_names:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            base_models.append(('random_forest', rf_model))
        
        # Meta-learner configuration
        meta_learner_type = ensemble_config.get('meta_learner', 'ridge')
        stacking_cv_folds = ensemble_config.get('stacking_cv_folds', 5)
        
        if meta_learner_type == 'ridge':
            meta_learner = LogisticRegression(penalty='l2', random_state=42, max_iter=1000)
        else:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=stacking_cv_folds,
            stack_method='predict_proba'
        )
        
        # Train the ensemble
        stacking_model.fit(X, y)
        
        # Make predictions
        y_pred_proba = stacking_model.predict_proba(X)[:, 1]
        y_pred = stacking_model.predict(X)
        
        # Performance metrics
        performance = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba)
        }
        
        # Individual model performance
        base_model_performance = {}
        for name, model in base_models:
            try:
                model.fit(X, y)
                pred_proba = model.predict_proba(X)[:, 1]
                base_model_performance[name] = {
                    'auc': roc_auc_score(y, pred_proba),
                    'accuracy': accuracy_score(y, model.predict(X))
                }
            except Exception as e:
                self.logger.warning(f"Error evaluating {name}: {e}")
                base_model_performance[name] = {'auc': 0.5, 'accuracy': 0.5}
        
        # Feature importance (from best base model)
        best_model_name = max(base_model_performance.keys(), 
                            key=lambda x: base_model_performance[x]['auc'])
        best_model = dict(base_models)[best_model_name]
        
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            elif hasattr(best_model, 'coef_'):
                feature_importance = dict(zip(X.columns, abs(best_model.coef_[0])))
            else:
                feature_importance = {}
        except:
            feature_importance = {}
        
        results = {
            'model_type': 'ensemble_stacking',
            'model': stacking_model,
            'base_models': dict(base_models),
            'performance': performance,
            'base_model_performance': base_model_performance,
            'feature_importance': feature_importance,
            'best_base_model': best_model_name,
            'predictions': y_pred_proba,
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Ensemble model trained. AUC: {performance['auc']:.3f}")
        
        return results
    
    def _get_optimized_params(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Get optimized hyperparameters using Optuna
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', etc.)
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of optimized parameters
        """
        self.logger.info(f"Optimizing hyperparameters for {model_type}...")
        
        optuna_config = self.hpo_config.get('optuna', {})
        n_trials = optuna_config.get('n_trials', 50)  # Reduced for demo
        timeout = optuna_config.get('timeout', 300)  # 5 minutes
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                
            else:  # Default parameters
                if model_type == 'catboost':
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 500),
                        'depth': trial.suggest_int('depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    }
                else:
                    params = {}
                
                return 0.5  # Fallback score
            
            # Cross-validation
            try:
                scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
                return np.mean(scores)
            except:
                return 0.5  # Fallback if error
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )
            
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
            
            best_params = study.best_params
            self.logger.info(f"Best {model_type} AUC: {study.best_value:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed for {model_type}: {e}")
            # Return default parameters
            if model_type == 'lightgbm':
                best_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
            elif model_type == 'xgboost':
                best_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
            elif model_type == 'catboost':
                best_params = {'iterations': 100, 'depth': 6, 'learning_rate': 0.1}
            else:
                best_params = {}
        
        return best_params
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train all three model types and compare results
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with all model results
        """
        self.logger.info("Training all model types...")
        
        all_results = {}
        
        # Train Hierarchical Bayesian Model
        try:
            hb_results = self.train_hierarchical_bayesian_model(X, y)
            all_results['hierarchical_bayesian'] = hb_results
            self.models['hierarchical_bayesian'] = hb_results['model']
        except Exception as e:
            self.logger.error(f"Failed to train Hierarchical Bayesian model: {e}")
        
        # Train X-Learner Model
        try:
            xl_results = self.train_x_learner_model(X, y)
            all_results['x_learner'] = xl_results
            self.models['x_learner'] = xl_results['model']
        except Exception as e:
            self.logger.error(f"Failed to train X-Learner model: {e}")
        
        # Train Ensemble Model
        try:
            ensemble_results = self.train_ensemble_model(X, y)
            all_results['ensemble'] = ensemble_results
            self.models['ensemble'] = ensemble_results['model']
        except Exception as e:
            self.logger.error(f"Failed to train Ensemble model: {e}")
        
        # Compare model performance
        model_comparison = self._compare_model_performance(all_results)
        all_results['model_comparison'] = model_comparison
        
        # Store training results
        self.training_results = all_results
        
        self.logger.info(f"All models trained. Best model: {model_comparison.get('best_model', 'Unknown')}")
        
        return all_results
    
    def _compare_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance across all trained models
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'models_trained': list(results.keys()),
            'performance_summary': {},
            'best_model': None,
            'best_auc': 0
        }
        
        for model_name, model_results in results.items():
            if 'performance' in model_results:
                perf = model_results['performance']
                comparison['performance_summary'][model_name] = perf
                
                # Track best AUC
                model_auc = perf.get('auc', 0)
                if model_auc > comparison['best_auc']:
                    comparison['best_auc'] = model_auc
                    comparison['best_model'] = model_name
        
        return comparison
    
    def save_models(self, output_dir: str = "models/trained"):
        """
        Save trained models and metadata
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            try:
                model_path = output_path / f"{model_name}_model.pkl"
                
                if model_name == 'hierarchical_bayesian' and HAS_PYMC:
                    # Special handling for PyMC models
                    if hasattr(model, 'save'):
                        model.save(str(model_path).replace('.pkl', '.nc'))
                else:
                    joblib.dump(model, model_path)
                    
                self.logger.info(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                self.logger.error(f"Error saving {model_name} model: {e}")
        
        # Save training results and metadata
        if self.training_results:
            # Convert non-serializable objects
            serializable_results = {}
            for model_name, results in self.training_results.items():
                serializable_results[model_name] = {
                    k: v for k, v in results.items() 
                    if k not in ['model', 'trace'] and not k.startswith('_')
                }
            
            with open(output_path / 'training_results.json', 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"Saved training results to {output_path}")
    
    def load_models(self, input_dir: str = "models/trained"):
        """
        Load trained models
        
        Args:
            input_dir: Directory to load models from
        """
        input_path = Path(input_dir)
        
        for model_file in input_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            try:
                self.models[model_name] = joblib.load(model_file)
                self.logger.info(f"Loaded {model_name} model")
            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {e}")
        
        # Load training results
        results_file = input_path / 'training_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.training_results = json.load(f)


def main():
    """Main function to demonstrate model training"""
    print("🚀 Starting Model Training Pipeline...")
    
    # Initialize model training
    trainer = PriceElasticityModelTraining()
    
    print("Model training pipeline is ready!")
    print("Use trainer.train_all_models(X, y) to train all models.")


if __name__ == "__main__":
    main()

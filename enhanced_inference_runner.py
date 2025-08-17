#!/usr/bin/env python3
"""
Enhanced Inference Runner with Dual Price Predictions
=====================================================

This script runs inference with enhanced models that provide:
1. Accurate Price: The price customers are most likely to accept (high probability)
2. Stretch Price: The maximum price for profit optimization (price ceiling)
3. Customer-Product level analysis

Usage:
    python enhanced_inference_runner.py
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union

# Add src to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
from utils.config_loader import config_loader, logger

# Import enhanced models
sys.path.append(str(current_dir / "src" / "models"))
try:
    from enhanced_price_elasticity_models import (
        EnhancedHierarchicalBayesianModel,
        EnhancedGraphNeuralNetworkModel, 
        EnhancedEnsembleModel,
        PriceOptimizationEngine
    )
    HAS_ENHANCED_MODELS = True
except ImportError as e:
    logger.warning(f"Enhanced models not available: {e}")
    HAS_ENHANCED_MODELS = False

warnings.filterwarnings('ignore')


class EnhancedInferenceRunner:
    """Enhanced inference runner with dual price prediction capabilities"""
    
    def __init__(self):
        """Initialize the enhanced inference runner"""
        self.config = config_loader
        self.logger = logger
        self.feature_engineer = PriceElasticityFeatureEngineering()
        self.models = {}
        self.enhanced_models = {}
        self.training_results = {}
        self.data = {}
        
        self.logger.info("Enhanced Inference runner initialized")
    
    def load_data(self):
        """Load the datasets used for training"""
        self.logger.info("Loading datasets...")
        
        data_config = self.config.get_data_config()
        datasets_path = Path(data_config['datasets_path'])
        files = data_config['files']
        
        for name, filename in files.items():
            file_path = datasets_path / filename
            if file_path.exists():
                self.data[name] = pd.read_csv(file_path)
                self.logger.info(f"Loaded {name}: {self.data[name].shape}")
            else:
                self.logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        return self.data
    
    def create_unified_dataset(self):
        """Create unified analytical dataset"""
        self.logger.info("Creating unified dataset...")
        
        # Start with quote_history as central table
        unified = self.data['quote_history'].copy()
        
        # Join with sales_history for historical context
        if 'sales_history' in self.data:
            sales_agg = self.data['sales_history'].groupby(['Customer_ID', 'Product_ID']).agg({
                'Quantity': ['sum', 'mean'],
                'Unit_Price': ['mean', 'std'],
                'Total_Revenue': 'sum',
                'Sale_Date': 'max'
            }).round(2)
            
            sales_agg.columns = [f'hist_{col[0]}_{col[1]}' for col in sales_agg.columns]
            sales_agg = sales_agg.reset_index()
            unified = unified.merge(sales_agg, on=['Customer_ID', 'Product_ID'], how='left')
        
        # Join with customer tables
        for table_name in ['customer_master', 'customer_segmentation']:
            if table_name in self.data:
                unified = unified.merge(self.data[table_name], on='Customer_ID', how='left')
        
        # Join with product_master
        if 'product_master' in self.data:
            unified = unified.merge(self.data['product_master'], on='Product_ID', how='left')
        
        self.logger.info(f"Unified dataset created: {unified.shape}")
        return unified
    
    def load_artifacts(self):
        """Load trained models and feature engineering artifacts"""
        self.logger.info("Loading trained models and artifacts...")
        
        # Load feature engineering artifacts
        fe_path = Path("models/feature_engineering")
        if fe_path.exists():
            self.feature_engineer.load_feature_engineering_artifacts()
            self.logger.info("Feature engineering artifacts loaded")
        else:
            raise FileNotFoundError("Feature engineering artifacts not found")
        
        # Load trained models (legacy)
        models_path = Path("models/trained")
        if models_path.exists():
            for model_file in models_path.glob("*_model.pkl"):
                model_name = model_file.stem.replace('_model', '')
                try:
                    self.models[model_name] = joblib.load(model_file)
                    self.logger.info(f"Loaded legacy model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Could not load {model_name}: {e}")
            
            # Load training results
            results_file = models_path / 'training_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.training_results = json.load(f)
        
        # Try to train enhanced models if available
        if HAS_ENHANCED_MODELS:
            self.logger.info("Enhanced models available - will train on-the-fly for demonstration")
        
        if not self.models and not HAS_ENHANCED_MODELS:
            raise ValueError("No trained models found and enhanced models not available")
        
        self.logger.info(f"Loaded {len(self.models)} legacy models")
    
    def run_enhanced_inference(self, unified_data):
        """Run inference with enhanced dual pricing models"""
        self.logger.info("Running enhanced inference with dual pricing...")
        
        # Apply feature engineering
        self.logger.info("Applying feature engineering...")
        featured_data = self.feature_engineer.create_comprehensive_features(unified_data, fit=False)
        
        # Store original data columns we want to keep
        available_cols = unified_data.columns.tolist()
        keep_cols_mapping = {
            'Quote_ID': 'Quote_ID',
            'Customer_ID': 'Customer_ID', 
            'Product_ID': 'Product_ID',
            'Quote_Date': 'Quote_Date',
            'List_Price': 'List_Price_x' if 'List_Price_x' in available_cols else 'List_Price',
            'Net_Price': 'Net_Price',
            'Offered_Price': 'Offered_Price', 
            'Status': 'Status',
            'Customer_Segment': 'Customer_Segment',
            'Product_Category': 'Product_Category_x' if 'Product_Category_x' in available_cols else 'Product_Category'
        }
        
        # Filter to only available columns
        available_keep_cols = [col for col in keep_cols_mapping.values() if col in available_cols]
        
        # Create results dataframe
        results_df = unified_data[available_keep_cols].copy()
        
        # Rename columns to remove suffixes for cleaner output
        rename_mapping = {v: k for k, v in keep_cols_mapping.items() if v in available_keep_cols}
        results_df = results_df.rename(columns=rename_mapping)
        
        # Train enhanced models on-the-fly for demonstration
        if HAS_ENHANCED_MODELS:
            self._train_enhanced_models_on_demand(results_df)
        
        # Initialize dual price results
        accurate_prices_ensemble = []
        stretch_prices_ensemble = []
        ensemble_weights = []
        
        # Run enhanced model predictions
        if self.enhanced_models:
            for model_name, model in self.enhanced_models.items():
                self.logger.info(f"Running enhanced predictions with {model_name}")
                
                try:
                    # Get dual price predictions
                    win_probabilities = model.predict_win_probability(results_df)
                    accurate_prices, stretch_prices = model.predict_dual_prices(results_df)
                    
                    # Store results
                    results_df[f'{model_name}_win_probability'] = win_probabilities
                    results_df[f'{model_name}_prediction'] = (win_probabilities > 0.5).map({True: 'Won', False: 'Lost'})
                    results_df[f'{model_name}_accurate_price'] = accurate_prices
                    results_df[f'{model_name}_stretch_price'] = stretch_prices
                    
                    # Calculate price optimization insights
                    price_insights = self._calculate_price_insights(
                        results_df, accurate_prices, stretch_prices, win_probabilities
                    )
                    results_df[f'{model_name}_price_recommendation'] = price_insights['recommendation']
                    results_df[f'{model_name}_expected_revenue_accurate'] = price_insights['expected_revenue_accurate']
                    results_df[f'{model_name}_expected_revenue_stretch'] = price_insights['expected_revenue_stretch']
                    
                    # Store for ensemble
                    accurate_prices_ensemble.append(accurate_prices)
                    stretch_prices_ensemble.append(stretch_prices)
                    
                    # Weight based on model performance
                    performance = getattr(model, 'performance_', {'auc': 0.5})
                    ensemble_weights.append(performance.get('auc', 0.5))
                    
                except Exception as e:
                    self.logger.error(f"Error with enhanced model {model_name}: {e}")
                    
        # Run legacy model predictions with fallback pricing
        legacy_accurate_prices = []
        legacy_stretch_prices = []
        
        for model_name, model in self.models.items():
            self.logger.info(f"Running legacy predictions with {model_name}")
            
            try:
                # Prepare features for legacy models
                exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
                feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
                X = featured_data[feature_cols]
                
                # Handle feature name mismatches
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X)
                        if probabilities.shape[1] == 2:
                            win_probs = probabilities[:, 1]
                        else:
                            win_probs = np.max(probabilities, axis=1)
                        
                        predictions = model.predict(X)
                        
                        # Generate fallback pricing
                        fallback_accurate, fallback_stretch = self._generate_fallback_dual_prices(
                            results_df, win_probs
                        )
                        
                        results_df[f'{model_name}_win_probability'] = win_probs
                        results_df[f'{model_name}_prediction'] = predictions
                        results_df[f'{model_name}_accurate_price'] = fallback_accurate
                        results_df[f'{model_name}_stretch_price'] = fallback_stretch
                        
                        legacy_accurate_prices.append(fallback_accurate)
                        legacy_stretch_prices.append(fallback_stretch)
                        
                    except Exception as feature_error:
                        self.logger.error(f"Feature alignment failed for {model_name}: {feature_error}")
                        # Fill with errors
                        results_df[f'{model_name}_win_probability'] = np.nan
                        results_df[f'{model_name}_prediction'] = 'Error'
                        results_df[f'{model_name}_accurate_price'] = np.nan
                        results_df[f'{model_name}_stretch_price'] = np.nan
                        
                elif isinstance(model, dict):
                    # Handle dict-type models (like hierarchical Bayesian results)
                    results_df[f'{model_name}_win_probability'] = np.nan
                    results_df[f'{model_name}_prediction'] = 'Error'
                    results_df[f'{model_name}_accurate_price'] = np.nan
                    results_df[f'{model_name}_stretch_price'] = np.nan
                    
            except Exception as e:
                self.logger.error(f"Error running predictions with {model_name}: {e}")
                results_df[f'{model_name}_win_probability'] = np.nan
                results_df[f'{model_name}_prediction'] = 'Error'
                results_df[f'{model_name}_accurate_price'] = np.nan
                results_df[f'{model_name}_stretch_price'] = np.nan
        
        # Create ensemble predictions with dual pricing
        if accurate_prices_ensemble or legacy_accurate_prices:
            all_accurate = accurate_prices_ensemble + legacy_accurate_prices
            all_stretch = stretch_prices_ensemble + legacy_stretch_prices
            all_weights = ensemble_weights + [0.3] * len(legacy_accurate_prices)  # Lower weight for legacy
            
            if all_accurate and all_stretch:
                weights_array = np.array(all_weights)
                weights_normalized = weights_array / weights_array.sum() if weights_array.sum() > 0 else np.ones_like(weights_array) / len(weights_array)
                
                accurate_array = np.array(all_accurate)
                stretch_array = np.array(all_stretch)
                
                ensemble_accurate_prices = np.average(accurate_array, axis=0, weights=weights_normalized)
                ensemble_stretch_prices = np.average(stretch_array, axis=0, weights=weights_normalized)
            else:
                # Final fallback
                ensemble_accurate_prices, ensemble_stretch_prices = self._generate_fallback_dual_prices(
                    results_df, np.full(len(results_df), 0.5)
                )
        else:
            # Final fallback
            ensemble_accurate_prices, ensemble_stretch_prices = self._generate_fallback_dual_prices(
                results_df, np.full(len(results_df), 0.5)
            )
        
        # Calculate ensemble win probabilities
        ensemble_win_prob = []
        ensemble_prediction = []
        
        for idx in range(len(results_df)):
            probs = []
            for col in results_df.columns:
                if col.endswith('_win_probability'):
                    prob = results_df.iloc[idx][col]
                    if not pd.isna(prob) and prob != 'Error':
                        try:
                            probs.append(float(prob))
                        except (ValueError, TypeError):
                            continue
            
            if probs:
                avg_prob = np.mean(probs)
                ensemble_win_prob.append(avg_prob)
                ensemble_prediction.append('Won' if avg_prob > 0.5 else 'Lost')
            else:
                ensemble_win_prob.append(0.5)
                ensemble_prediction.append('Lost')
        
        # Add ensemble results
        results_df['ensemble_win_probability'] = ensemble_win_prob
        results_df['ensemble_prediction'] = ensemble_prediction
        results_df['ensemble_accurate_price'] = ensemble_accurate_prices
        results_df['ensemble_stretch_price'] = ensemble_stretch_prices
        
        # Add pricing insights
        ensemble_price_insights = self._calculate_price_insights(
            results_df, ensemble_accurate_prices, ensemble_stretch_prices, ensemble_win_prob
        )
        results_df['ensemble_price_recommendation'] = ensemble_price_insights['recommendation']
        results_df['ensemble_expected_revenue_accurate'] = ensemble_price_insights['expected_revenue_accurate']
        results_df['ensemble_expected_revenue_stretch'] = ensemble_price_insights['expected_revenue_stretch']
        results_df['ensemble_profit_uplift_potential'] = (
            ensemble_price_insights['expected_revenue_stretch'] - 
            ensemble_price_insights['expected_revenue_accurate']
        )
        
        # Add analysis columns
        results_df['actual_status'] = results_df.get('Status', 'Unknown')
        if 'List_Price' in results_df.columns and 'Net_Price' in results_df.columns:
            results_df['current_discount_percent'] = (
                (results_df['List_Price'] - results_df['Net_Price']) / results_df['List_Price']
            )
            results_df['accurate_price_discount'] = (
                (results_df['List_Price'] - results_df['ensemble_accurate_price']) / results_df['List_Price']
            )
            results_df['stretch_price_discount'] = (
                (results_df['List_Price'] - results_df['ensemble_stretch_price']) / results_df['List_Price']
            )
        
        # Customer-Product level insights
        results_df = self._add_customer_product_insights(results_df)
        
        # Calculate model accuracies
        self._calculate_model_accuracies(results_df)
        
        self.logger.info(f"Enhanced inference completed. Results shape: {results_df.shape}")
        self.logger.info(f"Average accurate price: ${results_df['ensemble_accurate_price'].mean():.2f}")
        self.logger.info(f"Average stretch price: ${results_df['ensemble_stretch_price'].mean():.2f}")
        self.logger.info(f"Average profit uplift potential: ${results_df['ensemble_profit_uplift_potential'].mean():.2f}")
        
        return results_df
    
    def _train_enhanced_models_on_demand(self, data_df):
        """Train enhanced models on-the-fly for demonstration"""
        self.logger.info("Training enhanced models on-the-fly...")
        
        try:
            # Prepare target variable
            if 'Status' in data_df.columns:
                y = (data_df['Status'] == 'Won').astype(int)
            else:
                # Generate synthetic target for demonstration
                y = np.random.choice([0, 1], size=len(data_df), p=[0.6, 0.4])
            
            # Sample data for faster training (demonstration purposes)
            sample_size = min(1000, len(data_df))
            sample_indices = np.random.choice(len(data_df), size=sample_size, replace=False)
            X_sample = data_df.iloc[sample_indices]
            y_sample = y[sample_indices] if isinstance(y, np.ndarray) else y.iloc[sample_indices]
            
            # Train Ensemble Model (combines other models internally)
            if 'ensemble' not in self.enhanced_models:
                ensemble_model = EnhancedEnsembleModel(self.config)
                ensemble_model.fit(X_sample, y_sample)
                self.enhanced_models['ensemble'] = ensemble_model
                self.logger.info("Enhanced Ensemble model trained")
                
        except Exception as e:
            self.logger.error(f"Error training enhanced models: {e}")
            # Continue without enhanced models
    
    def _generate_fallback_dual_prices(self, data_df, win_probabilities):
        """Generate fallback dual pricing using price elasticity principles"""
        try:
            # Use current prices as base
            if 'Net_Price' in data_df.columns:
                base_prices = data_df['Net_Price'].values
            elif 'List_Price' in data_df.columns:
                base_prices = data_df['List_Price'].values * 0.85  # 15% discount
            else:
                base_prices = np.full(len(data_df), 1000)  # Default price
            
            # Accurate price: Conservative (high win probability)
            # Lower price for higher certainty
            price_adjustment_accurate = (0.7 - win_probabilities) * 0.2  # Max 20% adjustment
            accurate_prices = base_prices * (1 + price_adjustment_accurate)
            
            # Stretch price: Aggressive (optimize for revenue)
            # Higher price when win probability allows
            price_adjustment_stretch = (win_probabilities - 0.3) * 0.15  # Max 15% increase
            stretch_prices = base_prices * (1 + price_adjustment_stretch)
            
            # Apply customer segment adjustments
            if 'Customer_Segment' in data_df.columns:
                for i, segment in enumerate(data_df['Customer_Segment']):
                    if segment == 'Enterprise':
                        stretch_prices[i] *= 1.05  # Premium pricing
                    elif segment == 'Strategic':
                        stretch_prices[i] *= 1.08  # Highest premium
                    elif segment == 'SMB':
                        accurate_prices[i] *= 0.95  # More conservative
            
            # Ensure stretch >= accurate
            stretch_prices = np.maximum(stretch_prices, accurate_prices * 1.02)
            
            # Ensure positive prices
            accurate_prices = np.maximum(accurate_prices, 1)
            stretch_prices = np.maximum(stretch_prices, 1)
            
            return accurate_prices, stretch_prices
            
        except Exception as e:
            self.logger.error(f"Error generating fallback prices: {e}")
            # Return default values
            default_accurate = np.full(len(data_df), 1000)
            default_stretch = np.full(len(data_df), 1100)
            return default_accurate, default_stretch
    
    def _calculate_price_insights(self, data_df, accurate_prices, stretch_prices, win_probabilities):
        """Calculate pricing insights and recommendations"""
        try:
            expected_revenue_accurate = accurate_prices * win_probabilities
            
            # For stretch prices, assume slightly lower win probability
            stretch_win_prob = np.array(win_probabilities) * 0.85  # 15% lower win rate for higher prices
            expected_revenue_stretch = stretch_prices * stretch_win_prob
            
            # Determine recommendation
            recommendations = []
            for i in range(len(data_df)):
                if expected_revenue_stretch[i] > expected_revenue_accurate[i] * 1.1:
                    recommendations.append('Stretch')
                elif win_probabilities[i] > 0.7:
                    recommendations.append('Stretch')
                elif win_probabilities[i] < 0.4:
                    recommendations.append('Accurate')
                else:
                    recommendations.append('Balanced')
            
            return {
                'recommendation': recommendations,
                'expected_revenue_accurate': expected_revenue_accurate,
                'expected_revenue_stretch': expected_revenue_stretch
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price insights: {e}")
            return {
                'recommendation': ['Balanced'] * len(data_df),
                'expected_revenue_accurate': np.zeros(len(data_df)),
                'expected_revenue_stretch': np.zeros(len(data_df))
            }
    
    def _add_customer_product_insights(self, results_df):
        """Add customer-product level insights"""
        try:
            # Customer price sensitivity analysis
            if 'Customer_ID' in results_df.columns:
                customer_stats = results_df.groupby('Customer_ID').agg({
                    'ensemble_win_probability': 'mean',
                    'ensemble_accurate_price': 'mean',
                    'ensemble_stretch_price': 'mean'
                }).round(2)
                
                customer_stats.columns = ['customer_avg_win_prob', 'customer_avg_accurate_price', 'customer_avg_stretch_price']
                results_df = results_df.merge(customer_stats, left_on='Customer_ID', right_index=True, how='left')
            
            # Product pricing patterns
            if 'Product_ID' in results_df.columns:
                product_stats = results_df.groupby('Product_ID').agg({
                    'ensemble_win_probability': 'mean',
                    'ensemble_accurate_price': 'mean',
                    'ensemble_stretch_price': 'mean'
                }).round(2)
                
                product_stats.columns = ['product_avg_win_prob', 'product_avg_accurate_price', 'product_avg_stretch_price']
                results_df = results_df.merge(product_stats, left_on='Product_ID', right_index=True, how='left')
            
        except Exception as e:
            self.logger.error(f"Error adding customer-product insights: {e}")
        
        return results_df
    
    def _calculate_model_accuracies(self, results_df):
        """Calculate and log model accuracies"""
        if 'actual_status' in results_df.columns:
            for col in results_df.columns:
                if col.endswith('_prediction'):
                    model_name = col.replace('_prediction', '')
                    try:
                        accuracy = (results_df[col] == results_df['actual_status']).mean()
                        self.logger.info(f"{model_name} accuracy: {accuracy:.3f}")
                    except:
                        self.logger.info(f"{model_name} accuracy: Error")
    
    def save_enhanced_results(self, results_df):
        """Save enhanced inference results with dual pricing"""
        self.logger.info("Saving enhanced results to CSV files...")
        
        # Create output directory
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_results_path = output_dir / f"enhanced_inference_results_{timestamp}.csv"
        results_df.to_csv(full_results_path, index=False)
        self.logger.info(f"Full enhanced results saved to: {full_results_path}")
        
        # Save pricing summary
        pricing_summary = self._create_pricing_summary(results_df)
        pricing_path = output_dir / f"pricing_summary_{timestamp}.csv"
        pricing_summary.to_csv(pricing_path, index=False)
        self.logger.info(f"Pricing summary saved to: {pricing_path}")
        
        # Save customer-product analysis
        if 'Customer_ID' in results_df.columns and 'Product_ID' in results_df.columns:
            cp_analysis = self._create_customer_product_analysis(results_df)
            cp_path = output_dir / f"customer_product_analysis_{timestamp}.csv"
            cp_analysis.to_csv(cp_path, index=False)
            self.logger.info(f"Customer-Product analysis saved to: {cp_path}")
        
        # Save model comparison
        model_comparison = self._create_model_comparison(results_df)
        model_path = output_dir / f"model_comparison_{timestamp}.csv"
        model_comparison.to_csv(model_path, index=False)
        self.logger.info(f"Model comparison saved to: {model_path}")
        
        return {
            'full_results': full_results_path,
            'pricing_summary': pricing_path,
            'customer_product_analysis': cp_path if 'Customer_ID' in results_df.columns else None,
            'model_comparison': model_path
        }
    
    def _create_pricing_summary(self, results_df):
        """Create pricing strategy summary"""
        summary_data = []
        
        # Overall summary
        summary_data.append({
            'Category': 'Overall',
            'Segment': 'All',
            'Count': len(results_df),
            'Avg_Accurate_Price': results_df['ensemble_accurate_price'].mean(),
            'Avg_Stretch_Price': results_df['ensemble_stretch_price'].mean(),
            'Avg_Win_Probability': results_df['ensemble_win_probability'].mean(),
            'Avg_Expected_Revenue_Accurate': results_df['ensemble_expected_revenue_accurate'].mean(),
            'Avg_Expected_Revenue_Stretch': results_df['ensemble_expected_revenue_stretch'].mean(),
            'Avg_Profit_Uplift_Potential': results_df['ensemble_profit_uplift_potential'].mean(),
            'Stretch_Recommended_Pct': (results_df['ensemble_price_recommendation'] == 'Stretch').mean() * 100
        })
        
        # By customer segment
        if 'Customer_Segment' in results_df.columns:
            for segment in results_df['Customer_Segment'].unique():
                if pd.notna(segment):
                    segment_data = results_df[results_df['Customer_Segment'] == segment]
                    summary_data.append({
                        'Category': 'Customer_Segment',
                        'Segment': segment,
                        'Count': len(segment_data),
                        'Avg_Accurate_Price': segment_data['ensemble_accurate_price'].mean(),
                        'Avg_Stretch_Price': segment_data['ensemble_stretch_price'].mean(),
                        'Avg_Win_Probability': segment_data['ensemble_win_probability'].mean(),
                        'Avg_Expected_Revenue_Accurate': segment_data['ensemble_expected_revenue_accurate'].mean(),
                        'Avg_Expected_Revenue_Stretch': segment_data['ensemble_expected_revenue_stretch'].mean(),
                        'Avg_Profit_Uplift_Potential': segment_data['ensemble_profit_uplift_potential'].mean(),
                        'Stretch_Recommended_Pct': (segment_data['ensemble_price_recommendation'] == 'Stretch').mean() * 100
                    })
        
        # By product category
        if 'Product_Category' in results_df.columns:
            for category in results_df['Product_Category'].unique():
                if pd.notna(category):
                    category_data = results_df[results_df['Product_Category'] == category]
                    summary_data.append({
                        'Category': 'Product_Category',
                        'Segment': category,
                        'Count': len(category_data),
                        'Avg_Accurate_Price': category_data['ensemble_accurate_price'].mean(),
                        'Avg_Stretch_Price': category_data['ensemble_stretch_price'].mean(),
                        'Avg_Win_Probability': category_data['ensemble_win_probability'].mean(),
                        'Avg_Expected_Revenue_Accurate': category_data['ensemble_expected_revenue_accurate'].mean(),
                        'Avg_Expected_Revenue_Stretch': category_data['ensemble_expected_revenue_stretch'].mean(),
                        'Avg_Profit_Uplift_Potential': category_data['ensemble_profit_uplift_potential'].mean(),
                        'Stretch_Recommended_Pct': (category_data['ensemble_price_recommendation'] == 'Stretch').mean() * 100
                    })
        
        return pd.DataFrame(summary_data).round(2)
    
    def _create_customer_product_analysis(self, results_df):
        """Create customer-product level analysis"""
        cp_analysis = results_df.groupby(['Customer_ID', 'Product_ID']).agg({
            'ensemble_win_probability': 'mean',
            'ensemble_accurate_price': 'mean',
            'ensemble_stretch_price': 'mean',
            'ensemble_profit_uplift_potential': 'mean',
            'Quote_ID': 'count',
            'Customer_Segment': 'first',
            'Product_Category': 'first'
        }).round(2)
        
        cp_analysis.columns = [
            'avg_win_probability', 'avg_accurate_price', 'avg_stretch_price', 
            'avg_profit_uplift', 'quote_count', 'customer_segment', 'product_category'
        ]
        
        return cp_analysis.reset_index()
    
    def _create_model_comparison(self, results_df):
        """Create model performance comparison"""
        model_data = []
        
        for col in results_df.columns:
            if col.endswith('_win_probability'):
                model_name = col.replace('_win_probability', '')
                pred_col = f'{model_name}_prediction'
                
                if pred_col in results_df.columns and 'actual_status' in results_df.columns:
                    try:
                        accuracy = (results_df[pred_col] == results_df['actual_status']).mean()
                        avg_win_prob = results_df[col].mean() if not results_df[col].isna().all() else np.nan
                        
                        model_data.append({
                            'model_name': model_name,
                            'accuracy': accuracy,
                            'avg_win_probability': avg_win_prob,
                            'has_dual_pricing': f'{model_name}_accurate_price' in results_df.columns,
                            'prediction_count': (~results_df[pred_col].isin(['Error', np.nan])).sum()
                        })
                    except:
                        model_data.append({
                            'model_name': model_name,
                            'accuracy': np.nan,
                            'avg_win_probability': np.nan,
                            'has_dual_pricing': f'{model_name}_accurate_price' in results_df.columns,
                            'prediction_count': 0
                        })
        
        return pd.DataFrame(model_data).round(4)
    
    def run_complete_inference(self):
        """Run complete enhanced inference pipeline"""
        self.logger.info("🚀 Starting Enhanced Inference Pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Create unified dataset
            unified_data = self.create_unified_dataset()
            
            # Load artifacts
            self.load_artifacts()
            
            # Run enhanced inference
            results_df = self.run_enhanced_inference(unified_data)
            
            # Save results
            saved_files = self.save_enhanced_results(results_df)
            
            # Print summary
            print("✅ Enhanced Inference completed successfully!")
            print(f"📁 Results saved to: results/inference/")
            print(f"📊 Total predictions: {len(results_df):,}")
            print(f"🎯 Models used: {', '.join(self.models.keys()) + ', ' + ', '.join(self.enhanced_models.keys()) if self.enhanced_models else ', '.join(self.models.keys())}")
            
            # Print pricing insights
            if 'ensemble_accurate_price' in results_df.columns:
                avg_accurate = results_df['ensemble_accurate_price'].mean()
                avg_stretch = results_df['ensemble_stretch_price'].mean()
                avg_uplift = results_df['ensemble_profit_uplift_potential'].mean()
                
                print(f"💰 Average Accurate Price: ${avg_accurate:,.2f}")
                print(f"🎯 Average Stretch Price: ${avg_stretch:,.2f}")
                print(f"📈 Average Profit Uplift Potential: ${avg_uplift:,.2f}")
                
                stretch_recommendations = (results_df['ensemble_price_recommendation'] == 'Stretch').mean() * 100
                print(f"🚀 Stretch Pricing Recommended: {stretch_recommendations:.1f}% of quotes")
            
            print("\n🎉 Enhanced Inference pipeline completed successfully!")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Enhanced inference pipeline failed: {e}")
            raise


def main():
    """Main function"""
    runner = EnhancedInferenceRunner()
    return runner.run_complete_inference()


if __name__ == "__main__":
    sys.exit(main())

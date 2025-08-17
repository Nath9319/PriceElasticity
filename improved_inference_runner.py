#!/usr/bin/env python3
"""
Improved Inference Runner with Better Dual Price Logic
=====================================================

This script improves the existing inference pipeline by:
1. Fixing feature compatibility issues
2. Implementing better dual pricing logic
3. Adding proper stretch price recommendations
4. Improving model performance metrics

Usage:
    python improved_inference_runner.py
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

from utils.config_loader import config_loader, logger
warnings.filterwarnings('ignore')

class ImprovedInferenceRunner:
    """Improved inference runner with better dual pricing logic"""
    
    def __init__(self):
        """Initialize the improved inference runner"""
        self.config = config_loader
        self.logger = logger
        self.models = {}
        self.training_results = {}
        
        self.logger.info("Improved Inference runner initialized")
    
    def load_datasets(self):
        """Load all required datasets"""
        self.logger.info("Loading datasets...")
        
        datasets_path = Path("datasets")
        files = {
            'quote_history': 'quote_history.csv',
            'customer_master': 'customer_master.csv',
            'customer_segmentation': 'customer_segmentation.csv',
            'product_master': 'product_master.csv',
            'sales_history': 'sales_history.csv'
        }
        
        data = {}
        for name, filename in files.items():
            file_path = datasets_path / filename
            if file_path.exists():
                data[name] = pd.read_csv(file_path)
                self.logger.info(f"Loaded {name}: {data[name].shape}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        return data
    
    def create_unified_dataset(self, data):
        """Create unified analytical dataset"""
        self.logger.info("Creating unified dataset...")
        
        # Start with quote_history as central table
        unified = data['quote_history'].copy()
        
        # Join with customer tables
        if 'customer_master' in data:
            unified = unified.merge(data['customer_master'], on='Customer_ID', how='left')
        if 'customer_segmentation' in data:
            unified = unified.merge(data['customer_segmentation'], on='Customer_ID', how='left')
        
        # Join with product_master
        if 'product_master' in data:
            unified = unified.merge(data['product_master'], on='Product_ID', how='left')
        
        self.logger.info(f"Unified dataset created: {unified.shape}")
        return unified
    
    def load_trained_models(self):
        """Load trained models"""
        self.logger.info("Loading trained models...")
        
        models_path = Path("models/trained")
        if not models_path.exists():
            self.logger.warning("No trained models directory found")
            return
        
        # Load models
        for model_file in models_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            try:
                self.models[model_name] = joblib.load(model_file)
                self.logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Could not load {model_name}: {e}")
        
        # Load training results
        results_file = models_path / 'training_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.training_results = json.load(f)
    
    def create_robust_features(self, df):
        """Create robust features that are compatible with trained models"""
        self.logger.info("Creating robust features...")
        
        featured_df = df.copy()
        
        # 1. Basic price features
        # Handle column suffixes from merges
        list_price_col = 'List_Price_x' if 'List_Price_x' in featured_df.columns else 'List_Price'
        if list_price_col in featured_df.columns and 'Net_Price' in featured_df.columns:
            featured_df['discount_amount'] = featured_df[list_price_col] - featured_df['Net_Price']
            featured_df['discount_percentage'] = (
                featured_df['discount_amount'] / featured_df[list_price_col]
            ).fillna(0)
        
        # 2. Customer features
        if 'Customer_Segment' in featured_df.columns:
            for segment in ['Strategic', 'Enterprise', 'SMB', 'Mid-Market']:
                featured_df[f'segment_{segment.lower()}'] = (
                    featured_df['Customer_Segment'] == segment
                ).astype(int)
        
        # 3. Product features
        # Handle column suffixes from merges
        product_category_col = 'Product_Category_x' if 'Product_Category_x' in featured_df.columns else 'Product_Category'
        if product_category_col in featured_df.columns:
            for category in ['Software', 'Hardware', 'Services', 'Support']:
                featured_df[f'category_{category.lower()}'] = (
                    featured_df[product_category_col] == category
                ).astype(int)
        
        # 4. Temporal features
        if 'Quote_Date' in featured_df.columns:
            featured_df['Quote_Date'] = pd.to_datetime(featured_df['Quote_Date'])
            featured_df['quote_year'] = featured_df['Quote_Date'].dt.year
            featured_df['quote_month'] = featured_df['Quote_Date'].dt.month
            featured_df['quote_quarter'] = featured_df['Quote_Date'].dt.quarter
            featured_df['quote_day_of_week'] = featured_df['Quote_Date'].dt.dayofweek
        
        # 5. Aggregated features
        if all(col in featured_df.columns for col in ['Customer_ID', 'Net_Price']):
            customer_stats = featured_df.groupby('Customer_ID')['Net_Price'].agg([
                'mean', 'std', 'count'
            ]).add_prefix('customer_')
            featured_df = featured_df.merge(customer_stats, on='Customer_ID', how='left')
        
        if all(col in featured_df.columns for col in ['Product_ID', 'Net_Price']):
            product_stats = featured_df.groupby('Product_ID')['Net_Price'].agg([
                'mean', 'std', 'count'
            ]).add_prefix('product_')
            featured_df = featured_df.merge(product_stats, on='Product_ID', how='left')
        
        # Fill missing values
        numeric_columns = featured_df.select_dtypes(include=[np.number]).columns
        featured_df[numeric_columns] = featured_df[numeric_columns].fillna(0)
        
        self.logger.info(f"Created {featured_df.shape[1]} features")
        return featured_df
    
    def robust_predict(self, model, X, model_name):
        """Make robust predictions handling feature mismatches"""
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                win_probs = probabilities[:, 1] if probabilities.shape[1] == 2 else np.max(probabilities, axis=1)
                predictions = model.predict(X)
                return win_probs, predictions, "Success"
            else:
                predictions = model.predict(X)
                # Estimate probabilities from predictions
                if hasattr(predictions, '__len__'):
                    win_probs = np.where(predictions == 1, 0.7, 0.3)
                else:
                    win_probs = np.full(len(X), 0.5)
                return win_probs, predictions, "Success"
        except Exception as e:
            self.logger.warning(f"Prediction failed for {model_name}: {e}")
            n_samples = len(X)
            win_probs = np.full(n_samples, 0.5)
            predictions = np.array(['Unknown'] * n_samples)
            return win_probs, predictions, f"Error: {str(e)}"
    
    def calculate_dual_prices(self, df, win_probabilities):
        """Calculate accurate and stretch prices based on win probabilities and market data"""
        self.logger.info("Calculating dual prices...")
        
        accurate_prices = []
        stretch_prices = []
        
        for i, row in df.iterrows():
            win_prob = win_probabilities[i] if i < len(win_probabilities) else 0.5
            
            # Base price from current data
            base_price = row.get('Net_Price', row.get('List_Price', 1000))
            if pd.isna(base_price) or base_price <= 0:
                base_price = 1000  # Default fallback
            
            # Accurate price: price with highest acceptance probability
            # Higher win probability suggests we can charge closer to list price
            if 'List_Price' in df.columns and not pd.isna(row.get('List_Price')):
                list_price = row['List_Price']
                # Accurate price balances between current net price and list price
                # based on win probability
                accurate_price = base_price + (list_price - base_price) * win_prob * 0.3
            else:
                # Conservative adjustment if no list price
                accurate_price = base_price * (0.95 + 0.10 * win_prob)
            
            # Stretch price: aggressive ceiling for maximum profit
            # This should be higher than accurate price but with lower acceptance probability
            stretch_multiplier = 1.1 + (0.25 * win_prob)  # 10-35% premium based on win probability
            stretch_price = accurate_price * stretch_multiplier
            
            # Apply customer segment adjustments
            segment = row.get('Customer_Segment', 'SMB')
            if segment == 'Strategic':
                # Strategic customers may pay premium
                stretch_price *= 1.05
            elif segment == 'SMB':
                # SMB customers are more price sensitive
                stretch_price *= 0.95
                accurate_price *= 0.98
            
            # Apply product category adjustments
            category = row.get('Product_Category', 'Software')
            if category == 'Software':
                # Software typically has higher margins
                stretch_price *= 1.03
            elif category == 'Hardware':
                # Hardware has lower margins
                stretch_price *= 0.98
                accurate_price *= 0.99
            
            accurate_prices.append(max(accurate_price, base_price * 0.8))  # Floor at 80% of base
            stretch_prices.append(max(stretch_price, accurate_price * 1.05))  # Floor at 5% above accurate
        
        return np.array(accurate_prices), np.array(stretch_prices)
    
    def calculate_pricing_insights(self, df, accurate_prices, stretch_prices, win_probs):
        """Calculate pricing insights and recommendations"""
        
        insights = []
        for i, row in df.iterrows():
            accurate_price = accurate_prices[i]
            stretch_price = stretch_prices[i]
            win_prob = win_probs[i]
            
            # Expected revenue calculations
            expected_revenue_accurate = accurate_price * win_prob
            expected_revenue_stretch = stretch_price * (win_prob * 0.7)  # Reduced probability for stretch
            
            # Profit uplift potential
            profit_uplift = expected_revenue_stretch - expected_revenue_accurate
            
            # Price recommendation logic
            if win_prob > 0.7 and profit_uplift > 50:  # High confidence and good uplift
                recommendation = 'Stretch'
            elif win_prob > 0.6:
                recommendation = 'Accurate'
            else:
                recommendation = 'Conservative'
            
            insights.append({
                'expected_revenue_accurate': expected_revenue_accurate,
                'expected_revenue_stretch': expected_revenue_stretch,
                'profit_uplift_potential': profit_uplift,
                'price_recommendation': recommendation
            })
        
        return insights
    
    def run_inference(self):
        """Run the complete inference pipeline"""
        self.logger.info("Starting improved inference pipeline...")
        
        # Load data
        data = self.load_datasets()
        if not data:
            raise ValueError("No data loaded")
        
        # Create unified dataset
        unified_df = self.create_unified_dataset(data)
        
        # Load models
        self.load_trained_models()
        
        # Create robust features
        featured_df = self.create_robust_features(unified_df)
        
        # Prepare feature columns for models
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Quote_Date', 'Status', 'Sale_ID']
        feature_columns = [col for col in featured_df.columns if col not in exclude_cols]
        X = featured_df[feature_columns]
        
        # Store base results
        base_columns = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Quote_Date', 'Net_Price', 'Offered_Price', 'Status']
        
        # Handle column name suffixes from merges
        list_price_col = 'List_Price_x' if 'List_Price_x' in unified_df.columns else 'List_Price'
        product_category_col = 'Product_Category_x' if 'Product_Category_x' in unified_df.columns else 'Product_Category'
        
        # Add columns that exist
        available_cols = base_columns.copy()
        if list_price_col in unified_df.columns:
            available_cols.append(list_price_col)
        if 'Customer_Segment' in unified_df.columns:
            available_cols.append('Customer_Segment')
        if product_category_col in unified_df.columns:
            available_cols.append(product_category_col)
            
        results_df = unified_df[available_cols].copy()
        
        # Rename columns for cleaner naming
        rename_map = {
            list_price_col: 'List_Price',
            product_category_col: 'Product_Category'
        }
        results_df = results_df.rename(columns=rename_map)
        
        # Run model predictions
        ensemble_win_probs = []
        model_count = 0
        
        for model_name, model in self.models.items():
            self.logger.info(f"Running predictions with {model_name}")
            
            win_probs, predictions, status = self.robust_predict(model, X, model_name)
            
            # Store individual model results
            results_df[f'{model_name}_win_probability'] = win_probs
            results_df[f'{model_name}_prediction'] = predictions
            
            if status == "Success":
                ensemble_win_probs.append(win_probs)
                model_count += 1
        
        # Calculate ensemble win probabilities
        if ensemble_win_probs:
            ensemble_win_prob = np.mean(ensemble_win_probs, axis=0)
        else:
            # Fallback: use improved heuristic based on historical data
            ensemble_win_prob = self.calculate_heuristic_win_probability(results_df)
        
        results_df['ensemble_win_probability'] = ensemble_win_prob
        results_df['ensemble_prediction'] = np.where(ensemble_win_prob > 0.5, 'Won', 'Lost')
        
        # Calculate dual prices
        accurate_prices, stretch_prices = self.calculate_dual_prices(results_df, ensemble_win_prob)
        results_df['ensemble_accurate_price'] = accurate_prices
        results_df['ensemble_stretch_price'] = stretch_prices
        
        # Calculate pricing insights
        pricing_insights = self.calculate_pricing_insights(results_df, accurate_prices, stretch_prices, ensemble_win_prob)
        
        for i, insights in enumerate(pricing_insights):
            for key, value in insights.items():
                results_df.loc[i, f'ensemble_{key}'] = value
        
        # Add analysis columns
        results_df['actual_status'] = results_df['Status']
        
        if 'List_Price' in results_df.columns and 'Net_Price' in results_df.columns:
            results_df['current_discount_percent'] = (
                (results_df['List_Price'] - results_df['Net_Price']) / results_df['List_Price']
            ).fillna(0)
            results_df['accurate_price_discount'] = (
                (results_df['List_Price'] - results_df['ensemble_accurate_price']) / results_df['List_Price']
            ).fillna(0)
            results_df['stretch_price_discount'] = (
                (results_df['List_Price'] - results_df['ensemble_stretch_price']) / results_df['List_Price']
            ).fillna(0)
        
        # Add customer-product level insights
        results_df = self.add_customer_product_insights(results_df)
        
        self.logger.info(f"Inference completed. Results shape: {results_df.shape}")
        return results_df
    
    def calculate_heuristic_win_probability(self, df):
        """Calculate win probability using heuristic approach when models fail"""
        self.logger.info("Using heuristic win probability calculation...")
        
        win_probs = []
        
        for _, row in df.iterrows():
            prob = 0.5  # Base probability
            
            # Adjust based on customer segment
            segment = row.get('Customer_Segment', 'SMB')
            if segment == 'Strategic':
                prob += 0.1
            elif segment == 'Enterprise':
                prob += 0.05
            elif segment == 'SMB':
                prob -= 0.1
            
            # Adjust based on discount level
            if 'List_Price' in df.columns and 'Net_Price' in df.columns:
                discount = (row.get('List_Price', 0) - row.get('Net_Price', 0)) / max(row.get('List_Price', 1), 1)
                if discount > 0.3:  # High discount
                    prob += 0.15
                elif discount > 0.2:  # Medium discount
                    prob += 0.1
                elif discount < 0.1:  # Low discount
                    prob -= 0.1
            
            # Adjust based on product category
            category = row.get('Product_Category', 'Software')
            if category == 'Software':
                prob += 0.05  # Software typically has better conversion
            
            win_probs.append(np.clip(prob, 0.1, 0.9))  # Keep within reasonable bounds
        
        return np.array(win_probs)
    
    def add_customer_product_insights(self, df):
        """Add customer-product level insights"""
        
        # Customer level insights
        customer_insights = df.groupby('Customer_ID').agg({
            'ensemble_win_probability': 'mean',
            'ensemble_accurate_price': 'mean',
            'ensemble_stretch_price': 'mean'
        }).add_prefix('customer_avg_')
        df = df.merge(customer_insights, on='Customer_ID', how='left')
        
        # Product level insights
        product_insights = df.groupby('Product_ID').agg({
            'ensemble_win_probability': 'mean',
            'ensemble_accurate_price': 'mean',
            'ensemble_stretch_price': 'mean'
        }).add_prefix('product_avg_')
        df = df.merge(product_insights, on='Product_ID', how='left')
        
        return df
    
    def save_results(self, results_df):
        """Save inference results and summaries"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path("results/inference")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / f"improved_inference_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Detailed results saved to {results_file}")
        
        # Create summary
        summary = self.create_summary(results_df)
        summary_file = results_dir / f"improved_pricing_summary_{timestamp}.csv"
        summary.to_csv(summary_file, index=False)
        self.logger.info(f"Summary saved to {summary_file}")
        
        # Print key metrics
        self.print_key_metrics(results_df)
        
        return results_file, summary_file
    
    def create_summary(self, df):
        """Create pricing summary by segment and category"""
        
        def calculate_metrics(group):
            return pd.Series({
                'Count': len(group),
                'Avg_Accurate_Price': group['ensemble_accurate_price'].mean(),
                'Avg_Stretch_Price': group['ensemble_stretch_price'].mean(),
                'Avg_Win_Probability': group['ensemble_win_probability'].mean(),
                'Avg_Expected_Revenue_Accurate': group['ensemble_expected_revenue_accurate'].mean(),
                'Avg_Expected_Revenue_Stretch': group['ensemble_expected_revenue_stretch'].mean(),
                'Avg_Profit_Uplift_Potential': group['ensemble_profit_uplift_potential'].mean(),
                'Stretch_Recommended_Pct': (group['ensemble_price_recommendation'] == 'Stretch').mean() * 100
            })
        
        # Overall summary
        overall = calculate_metrics(df)
        overall_df = pd.DataFrame([overall])
        overall_df.insert(0, 'Category', 'Overall')
        overall_df.insert(1, 'Segment', 'All')
        
        # By customer segment
        segment_summary = df.groupby('Customer_Segment').apply(calculate_metrics).reset_index()
        segment_summary.insert(0, 'Category', 'Customer_Segment')
        segment_summary.rename(columns={'Customer_Segment': 'Segment'}, inplace=True)
        
        # By product category
        category_summary = df.groupby('Product_Category').apply(calculate_metrics).reset_index()
        category_summary.insert(0, 'Category', 'Product_Category')
        category_summary.rename(columns={'Product_Category': 'Segment'}, inplace=True)
        
        # Combine all summaries
        summary = pd.concat([overall_df, segment_summary, category_summary], ignore_index=True)
        return summary
    
    def print_key_metrics(self, df):
        """Print key performance metrics"""
        self.logger.info("\n" + "="*60)
        self.logger.info("INFERENCE RESULTS SUMMARY")
        self.logger.info("="*60)
        
        total_quotes = len(df)
        avg_accurate_price = df['ensemble_accurate_price'].mean()
        avg_stretch_price = df['ensemble_stretch_price'].mean()
        avg_win_prob = df['ensemble_win_probability'].mean()
        avg_uplift = df['ensemble_profit_uplift_potential'].mean()
        stretch_recommended_pct = (df['ensemble_price_recommendation'] == 'Stretch').mean() * 100
        
        self.logger.info(f"Total quotes processed: {total_quotes:,}")
        self.logger.info(f"Average accurate price: ${avg_accurate_price:.2f}")
        self.logger.info(f"Average stretch price: ${avg_stretch_price:.2f}")
        self.logger.info(f"Average win probability: {avg_win_prob:.3f}")
        self.logger.info(f"Average profit uplift potential: ${avg_uplift:.2f}")
        self.logger.info(f"Stretch pricing recommended: {stretch_recommended_pct:.1f}%")
        
        # Model performance
        successful_models = sum(1 for col in df.columns if col.endswith('_win_probability') and 
                              not col.startswith('ensemble_') and df[col].notna().any())
        self.logger.info(f"Models with successful predictions: {successful_models}")

def main():
    """Run the improved inference pipeline"""
    runner = ImprovedInferenceRunner()
    
    try:
        # Run inference
        results = runner.run_inference()
        
        # Save results
        results_file, summary_file = runner.save_results(results)
        
        print(f"\nInference completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        logger.error(f"Inference failed: {e}")

if __name__ == "__main__":
    main()

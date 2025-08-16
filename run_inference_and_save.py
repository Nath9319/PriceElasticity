#!/usr/bin/env python3
"""
Run Inference on Training Data and Save Results to CSV
=====================================================

This script loads the trained models and runs inference on the same data used for training,
then saves the results to CSV files for further analysis.
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

# Add src to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')


class InferenceRunner:
    """Run inference on training data and save results"""
    
    def __init__(self):
        """Initialize the inference runner"""
        self.config = config_loader
        self.logger = logger
        self.feature_engineer = PriceElasticityFeatureEngineering()
        self.models = {}
        self.training_results = {}
        self.data = {}
        
        self.logger.info("Inference runner initialized")
    
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
        
        # Load trained models
        models_path = Path("models/trained")
        if models_path.exists():
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
        
        if not self.models:
            raise ValueError("No trained models found")
        
        self.logger.info(f"Loaded {len(self.models)} models")
    
    def run_inference(self, unified_data):
        """Run inference on the unified dataset"""
        self.logger.info("Running inference on all data...")
        
        # Apply feature engineering
        self.logger.info("Applying feature engineering...")
        featured_data = self.feature_engineer.create_comprehensive_features(unified_data, fit=False)
        
        # Prepare features for prediction
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
        feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
        X = featured_data[feature_cols]
        
        # Store original data columns we want to keep (handle _x, _y suffixes)
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
        
        # Run predictions with each model
        for model_name, model in self.models.items():
            self.logger.info(f"Running predictions with {model_name}")
            
            try:
                if hasattr(model, 'predict_proba'):
                    # Get probabilities for binary classification
                    probabilities = model.predict_proba(X)
                    if probabilities.shape[1] == 2:
                        # Binary classification - get probability of positive class (Won)
                        results_df[f'{model_name}_win_probability'] = probabilities[:, 1]
                    else:
                        # Multi-class - get max probability
                        results_df[f'{model_name}_win_probability'] = np.max(probabilities, axis=1)
                    
                    # Get class predictions
                    predictions = model.predict(X)
                    results_df[f'{model_name}_prediction'] = predictions
                    
                else:
                    # For models without predict_proba, just get predictions
                    predictions = model.predict(X)
                    results_df[f'{model_name}_prediction'] = predictions
                    results_df[f'{model_name}_win_probability'] = predictions  # Assume these are probabilities
                
                self.logger.info(f"Completed predictions with {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error running predictions with {model_name}: {e}")
                # Add null columns so the dataframe structure is consistent
                results_df[f'{model_name}_win_probability'] = np.nan
                results_df[f'{model_name}_prediction'] = 'Error'
        
        # Create ensemble predictions (average of all model probabilities)
        prob_columns = [col for col in results_df.columns if col.endswith('_win_probability')]
        if prob_columns:
            # Calculate ensemble probability as mean of all model probabilities
            results_df['ensemble_win_probability'] = results_df[prob_columns].mean(axis=1)
            
            # Create ensemble prediction based on ensemble probability
            results_df['ensemble_prediction'] = np.where(
                results_df['ensemble_win_probability'] > 0.5, 'Won', 'Lost'
            )
        
        # Add some analysis columns
        results_df['actual_status'] = results_df['Status']
        results_df['discount_percent'] = (results_df['List_Price'] - results_df['Net_Price']) / results_df['List_Price']
        results_df['discount_depth_category'] = pd.cut(
            results_df['discount_percent'], 
            bins=[0, 0.1, 0.2, 0.3, 1.0], 
            labels=['Low (0-10%)', 'Medium (10-20%)', 'High (20-30%)', 'Very High (30%+)']
        )
        
        # Calculate accuracy for each model if we have predictions and actual status
        if 'ensemble_prediction' in results_df.columns:
            for model_name in self.models.keys():
                pred_col = f'{model_name}_prediction'
                if pred_col in results_df.columns:
                    accuracy = (results_df[pred_col] == results_df['actual_status']).mean()
                    self.logger.info(f"{model_name} accuracy: {accuracy:.3f}")
            
            # Ensemble accuracy
            ensemble_accuracy = (results_df['ensemble_prediction'] == results_df['actual_status']).mean()
            self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
        
        self.logger.info(f"Inference completed. Results shape: {results_df.shape}")
        return results_df
    
    def save_results(self, results_df):
        """Save inference results to CSV files"""
        self.logger.info("Saving results to CSV files...")
        
        # Create output directory
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        full_results_path = output_dir / f"full_inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(full_results_path, index=False)
        self.logger.info(f"Full results saved to: {full_results_path}")
        
        # Save summary by model
        model_names = [col.replace('_win_probability', '') for col in results_df.columns if col.endswith('_win_probability')]
        
        summary_data = []
        for model_name in model_names:
            prob_col = f'{model_name}_win_probability'
            pred_col = f'{model_name}_prediction'
            
            if prob_col in results_df.columns:
                model_summary = {
                    'model_name': model_name,
                    'avg_win_probability': results_df[prob_col].mean(),
                    'median_win_probability': results_df[prob_col].median(),
                    'std_win_probability': results_df[prob_col].std(),
                    'min_win_probability': results_df[prob_col].min(),
                    'max_win_probability': results_df[prob_col].max()
                }
                
                if pred_col in results_df.columns and 'actual_status' in results_df.columns:
                    model_summary['accuracy'] = (results_df[pred_col] == results_df['actual_status']).mean()
                    model_summary['predicted_wins'] = (results_df[pred_col] == 'Won').sum()
                    model_summary['predicted_losses'] = (results_df[pred_col] == 'Lost').sum()
                
                summary_data.append(model_summary)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Model summary saved to: {summary_path}")
        
        # Save segment analysis
        if 'Customer_Segment' in results_df.columns and 'ensemble_win_probability' in results_df.columns:
            segment_analysis = results_df.groupby('Customer_Segment').agg({
                'ensemble_win_probability': ['mean', 'median', 'std', 'count'],
                'discount_percent': ['mean', 'median'],
                'Net_Price': ['mean', 'median'],
                'actual_status': lambda x: (x == 'Won').mean()
            }).round(4)
            
            segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
            segment_analysis = segment_analysis.reset_index()
            
            segment_path = output_dir / f"segment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            segment_analysis.to_csv(segment_path, index=False)
            self.logger.info(f"Segment analysis saved to: {segment_path}")
        
        # Save product category analysis
        if 'Product_Category' in results_df.columns and 'ensemble_win_probability' in results_df.columns:
            category_analysis = results_df.groupby('Product_Category').agg({
                'ensemble_win_probability': ['mean', 'median', 'std', 'count'],
                'discount_percent': ['mean', 'median'],
                'Net_Price': ['mean', 'median'],
                'actual_status': lambda x: (x == 'Won').mean()
            }).round(4)
            
            category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns]
            category_analysis = category_analysis.reset_index()
            
            category_path = output_dir / f"category_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            category_analysis.to_csv(category_path, index=False)
            self.logger.info(f"Category analysis saved to: {category_path}")
        
        return full_results_path
    
    def run_complete_inference(self):
        """Run the complete inference pipeline"""
        try:
            print("🚀 Starting Inference Pipeline...")
            
            # Load data
            self.load_data()
            
            # Create unified dataset  
            unified_data = self.create_unified_dataset()
            
            # Load artifacts
            self.load_artifacts()
            
            # Run inference
            results_df = self.run_inference(unified_data)
            
            # Save results
            output_path = self.save_results(results_df)
            
            print("✅ Inference completed successfully!")
            print(f"📁 Results saved to: results/inference/")
            print(f"📊 Total predictions: {len(results_df):,}")
            print(f"🎯 Models used: {', '.join(self.models.keys())}")
            
            if 'ensemble_win_probability' in results_df.columns:
                avg_win_prob = results_df['ensemble_win_probability'].mean()
                print(f"📈 Average win probability: {avg_win_prob:.3f}")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Inference pipeline failed: {e}", exc_info=True)
            print(f"❌ Inference failed: {e}")
            return None


def main():
    """Main function"""
    runner = InferenceRunner()
    results = runner.run_complete_inference()
    
    if results is not None:
        print("\n🎉 Inference pipeline completed successfully!")
        return 0
    else:
        print("\n💥 Inference pipeline failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

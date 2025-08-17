"""
Enhanced Fixed Inference Runner - Addresses Model Prediction Errors
Fixes specific issues: data types (ensemble), feature alignment (graph_neural_network), invalid models filtering
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.append('.')

try:
    from utils.config_loader import ConfigLoader
    from utils.feature_engineering import FeatureEngineering
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    print("Running in standalone mode without utils")

class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

class EnhancedFixedInference:
    """Enhanced inference with targeted fixes for model errors"""
    
    def __init__(self):
        if HAS_UTILS:
            try:
                self.config = ConfigLoader()
                self.feature_engineer = FeatureEngineering(self.config)
                self.logger = self.config.logger
            except:
                self.logger = MockLogger()
                self.feature_engineer = None
        else:
            self.logger = MockLogger()
            self.feature_engineer = None
        
        self.models = {}
        self.model_features = {}
        self.data = {}
        
        self.logger.info("Enhanced Fixed Inference initialized")
    
    def load_data(self):
        """Load datasets"""
        self.logger.info("Loading datasets...")
        
        datasets = {
            'quote_history': 'datasets/quote_history.csv',
            'sales_history': 'datasets/sales_history.csv', 
            'customer_master': 'datasets/customer_master.csv',
            'customer_segmentation': 'datasets/customer_segmentation.csv',
            'product_master': 'datasets/product_master.csv'
        }
        
        for name, path in datasets.items():
            if Path(path).exists():
                self.data[name] = pd.read_csv(path)
                self.logger.info(f"Loaded {name}: {self.data[name].shape}")
            else:
                self.logger.warning(f"File not found: {path}")
        
        return len(self.data) > 0
    
    def create_unified_dataset(self):
        """Create unified dataset"""
        if 'quote_history' not in self.data:
            raise ValueError("quote_history data not found")
        
        unified = self.data['quote_history'].copy()
        
        # Add sales history aggregations
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
        
        # Join customer data
        for table_name in ['customer_master', 'customer_segmentation']:
            if table_name in self.data:
                unified = unified.merge(self.data[table_name], on='Customer_ID', how='left')
        
        # Join product data
        if 'product_master' in self.data:
            unified = unified.merge(self.data['product_master'], on='Product_ID', how='left')
        
        self.logger.info(f"Unified dataset created: {unified.shape}")
        return unified
    
    def load_models_with_validation(self):
        """Load models and validate they have predict methods"""
        self.logger.info("Loading models with validation...")
        
        models_path = Path("models/trained")
        if not models_path.exists():
            self.logger.error("Models directory not found")
            return False
        
        valid_count = 0
        invalid_count = 0
        
        for model_file in models_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            
            try:
                model = joblib.load(model_file)
                
                # Check if model has predict method
                if hasattr(model, 'predict'):
                    self.models[model_name] = model
                    
                    # Store expected features if available
                    if hasattr(model, 'feature_names_in_'):
                        self.model_features[model_name] = list(model.feature_names_in_)
                        self.logger.info(f"✓ {model_name}: Valid model with {len(self.model_features[model_name])} features")
                    else:
                        self.logger.info(f"✓ {model_name}: Valid model (no feature info)")
                    
                    valid_count += 1
                else:
                    self.logger.warning(f"✗ {model_name}: Invalid - no predict method (type: {type(model).__name__})")
                    invalid_count += 1
                    
            except Exception as e:
                self.logger.error(f"✗ {model_name}: Failed to load - {e}")
                invalid_count += 1
        
        self.logger.info(f"Models loaded: {valid_count} valid, {invalid_count} invalid")
        return len(self.models) > 0
    
    def fix_data_types_for_ensemble(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fix data type issues that cause ensemble model to fail"""
        self.logger.info("Fixing data types for ensemble model...")
        X_fixed = X.copy()
        
        # Convert datetime columns that are still objects
        datetime_issues = ['hist_Sale_Date_max']
        for col in datetime_issues:
            if col in X_fixed.columns and X_fixed[col].dtype == 'object':
                self.logger.info(f"Converting datetime column: {col}")
                try:
                    X_fixed[col] = pd.to_datetime(X_fixed[col], errors='coerce')
                    X_fixed[col] = (X_fixed[col] - pd.Timestamp('1970-01-01')).dt.days.astype(float)
                    X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
                except:
                    X_fixed = X_fixed.drop(columns=[col])
        
        # Drop problematic text columns
        text_columns = ['Customer_Name', 'Product_Name']
        for col in text_columns:
            if col in X_fixed.columns:
                X_fixed = X_fixed.drop(columns=[col])
                self.logger.info(f"Dropped text column: {col}")
        
        # Convert any remaining object columns
        for col in X_fixed.columns:
            if X_fixed[col].dtype == 'object':
                self.logger.info(f"Converting object column: {col}")
                X_fixed[col] = pd.to_numeric(X_fixed[col], errors='coerce')
                if X_fixed[col].isnull().all():
                    X_fixed = X_fixed.drop(columns=[col])
                else:
                    X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
        
        # Final NaN check and fill
        nan_cols = X_fixed.columns[X_fixed.isnull().any()].tolist()
        if nan_cols:
            self.logger.info(f"Filling NaN values in columns: {nan_cols}")
            for col in nan_cols:
                if X_fixed[col].dtype in ['float64', 'int64']:
                    X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
                else:
                    X_fixed[col] = X_fixed[col].fillna(0)
        
        # Ensure no NaN values remain
        X_fixed = X_fixed.fillna(0)
        
        return X_fixed
    
    def align_features_for_model(self, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Align features with model requirements"""
        if model_name not in self.model_features:
            return X
        
        required_features = self.model_features[model_name]
        self.logger.info(f"Aligning {len(required_features)} features for {model_name}")
        
        X_aligned = pd.DataFrame(index=X.index)
        missing_count = 0
        
        for feature in required_features:
            if feature in X.columns:
                X_aligned[feature] = X[feature]
            else:
                X_aligned[feature] = 0.0  # Fill missing features with 0
                missing_count += 1
        
        if missing_count > 0:
            self.logger.warning(f"{model_name}: {missing_count} features missing, filled with 0")
        
        return X_aligned
    
    def create_features_with_fallback(self, unified_data: pd.DataFrame) -> pd.DataFrame:
        """Create features using available methods"""
        self.logger.info("Creating features...")
        
        # Try to use full feature engineering if available
        if self.feature_engineer and HAS_UTILS:
            try:
                fe_path = Path("models/feature_engineering")
                if fe_path.exists():
                    self.feature_engineer.load_feature_engineering_artifacts()
                    featured_data = self.feature_engineer.create_comprehensive_features(unified_data, fit=False)
                    self.logger.info(f"Full feature engineering applied: {featured_data.shape}")
                    return featured_data
            except Exception as e:
                self.logger.warning(f"Full feature engineering failed: {e}, using fallback")
        
        # Fallback feature creation
        featured_data = unified_data.copy()
        
        # Basic price features
        if 'List_Price' in featured_data.columns and 'Net_Price' in featured_data.columns:
            featured_data['Discount_Percent'] = (
                (featured_data['List_Price'] - featured_data['Net_Price']) / featured_data['List_Price']
            ).fillna(0)
        
        if 'Net_Price' in featured_data.columns and 'Offered_Price' in featured_data.columns:
            featured_data['Offered_Discount'] = (
                (featured_data['Net_Price'] - featured_data['Offered_Price']) / featured_data['Net_Price']
            ).fillna(0)
        
        # Simple categorical encoding
        for col in featured_data.columns:
            if featured_data[col].dtype == 'object' and col not in ['Quote_ID', 'Customer_ID', 'Product_ID', 'Status']:
                featured_data[col] = featured_data[col].astype('category').cat.codes
        
        self.logger.info(f"Fallback features created: {featured_data.shape}")
        return featured_data
    
    def predict_with_fixed_model(self, model, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Make predictions with model-specific fixes"""
        try:
            self.logger.info(f"Making predictions with {model_name}...")
            
            # Apply fixes based on known issues
            X_processed = X.copy()
            
            # Fix 1: Data type issues (for ensemble model)
            X_processed = self.fix_data_types_for_ensemble(X_processed)
            
            # Fix 2: Feature alignment (for graph neural network model)  
            X_processed = self.align_features_for_model(X_processed, model_name)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)
                win_prob = probabilities[:, 1] if probabilities.shape[1] == 2 else np.max(probabilities, axis=1)
                predictions = model.predict(X_processed)
                
                # Ensure predictions are in consistent format
                predictions = ['Won' if p == 1 or p == 'Won' else 'Lost' for p in predictions]
            else:
                predictions = model.predict(X_processed)
                win_prob = np.where(np.array(predictions) == 'Won', 0.7, 0.3)
            
            self.logger.info(f"✅ {model_name}: Successful prediction")
            return {
                'win_probability': win_prob,
                'prediction': predictions,
                'status': 'Success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ {model_name}: Prediction failed - {e}")
            return {
                'win_probability': np.full(len(X), np.nan),
                'prediction': ['Error'] * len(X),
                'status': 'Error'
            }
    
    def calculate_dual_prices(self, base_data: pd.DataFrame, win_probs: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """Calculate accurate and elastic prices"""
        accurate_prices = {}
        elastic_prices = {}
        
        list_price = base_data.get('List_Price', base_data.get('List_Price_x', pd.Series([1000] * len(base_data))))
        net_price = base_data.get('Net_Price', pd.Series([800] * len(base_data)))
        
        for model_name, probs in win_probs.items():
            if np.isnan(probs).all():
                # Fallback pricing
                accurate_prices[model_name] = net_price * 0.95
                elastic_prices[model_name] = list_price * 0.85
            else:
                # Elasticity-based pricing
                accurate_mult = 1 - (0.70 - probs) * 0.3  # Target 70% win rate
                elastic_mult = 1 - (0.50 - probs) * 0.4   # Target 50% win rate
                
                accurate_prices[model_name] = net_price * np.clip(accurate_mult, 0.7, 0.95)
                elastic_prices[model_name] = list_price * np.clip(elastic_mult, 0.8, 1.1)
        
        return accurate_prices, elastic_prices
    
    def run_enhanced_inference(self) -> pd.DataFrame:
        """Run enhanced inference with fixes"""
        self.logger.info("Starting enhanced inference with model fixes...")
        
        # Load data and models
        if not self.load_data():
            raise ValueError("Failed to load data")
        
        unified_data = self.create_unified_dataset()
        
        if not self.load_models_with_validation():
            raise ValueError("No valid models found")
        
        # Create features
        featured_data = self.create_features_with_fallback(unified_data)
        
        # Prepare feature matrix
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
        feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
        X = featured_data[feature_cols]
        
        self.logger.info(f"Feature matrix: {X.shape}")
        
        # Setup results dataframe
        keep_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Quote_Date', 'List_Price', 'Net_Price', 
                    'Offered_Price', 'Status', 'Customer_Segment', 'Product_Category']
        
        available_keep_cols = []
        for col in keep_cols:
            if col in unified_data.columns:
                available_keep_cols.append(col)
            elif f"{col}_x" in unified_data.columns:
                available_keep_cols.append(f"{col}_x")
        
        results_df = unified_data[available_keep_cols].copy()
        
        # Clean column names
        results_df.columns = [col.replace('_x', '') for col in results_df.columns]
        
        # Run predictions for each model
        successful_models = []
        win_probabilities = {}
        
        for model_name, model in self.models.items():
            pred_result = self.predict_with_fixed_model(model, X, model_name)
            
            # Store individual model results
            results_df[f'{model_name}_win_probability'] = pred_result['win_probability']
            results_df[f'{model_name}_prediction'] = pred_result['prediction']
            
            if pred_result['status'] == 'Success':
                successful_models.append(model_name)
                win_probabilities[model_name] = pred_result['win_probability']
        
        # Calculate dual prices
        accurate_prices, elastic_prices = self.calculate_dual_prices(results_df, win_probabilities)
        
        # Add dual price columns for each model
        for model_name in self.models.keys():
            if model_name in accurate_prices:
                results_df[f'{model_name}_accurate_price'] = accurate_prices[model_name]
                results_df[f'{model_name}_elastic_price'] = elastic_prices[model_name]
            else:
                results_df[f'{model_name}_accurate_price'] = np.nan
                results_df[f'{model_name}_elastic_price'] = np.nan
        
        # Create ensemble predictions from successful models
        if successful_models:
            valid_probs = [win_probabilities[name] for name in successful_models]
            valid_accurate = [accurate_prices[name] for name in successful_models]
            valid_elastic = [elastic_prices[name] for name in successful_models]
            
            results_df['ensemble_win_probability'] = np.mean(valid_probs, axis=0)
            results_df['ensemble_prediction'] = np.where(results_df['ensemble_win_probability'] > 0.5, 'Won', 'Lost')
            results_df['ensemble_accurate_price'] = np.mean(valid_accurate, axis=0)
            results_df['ensemble_elastic_price'] = np.mean(valid_elastic, axis=0)
        else:
            # Fallback ensemble
            results_df['ensemble_win_probability'] = 0.35
            results_df['ensemble_prediction'] = 'Lost'
            results_df['ensemble_accurate_price'] = results_df.get('Net_Price', 800) * 0.9
            results_df['ensemble_elastic_price'] = results_df.get('List_Price', 1000) * 0.85
        
        # Add metadata columns
        results_df['actual_status'] = results_df['Status']
        results_df['current_discount_percent'] = (
            (results_df.get('List_Price', 1000) - results_df.get('Net_Price', 800)) / 
            results_df.get('List_Price', 1000)
        ).fillna(0)
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        results_df['models_used'] = ', '.join(self.models.keys())
        results_df['successful_models'] = ', '.join(successful_models)
        
        self.logger.info(f"Enhanced inference complete: {results_df.shape}")
        self.logger.info(f"Successful models: {successful_models}")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame) -> str:
        """Save results to CSV"""
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"enhanced_fixed_inference_{timestamp}.csv"
        
        results_df.to_csv(output_file, index=False)
        return str(output_file)

def main():
    """Main function to run enhanced inference"""
    try:
        print("🚀 Starting Enhanced Fixed Inference...")
        print("="*50)
        
        runner = EnhancedFixedInference()
        results_df = runner.run_enhanced_inference()
        output_file = runner.save_results(results_df)
        
        print("\n🎉 SUCCESS! Enhanced inference completed!")
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Quotes processed: {len(results_df)}")
        print(f"🤖 Models loaded: {list(runner.models.keys())}")
        
        # Show prediction success rates
        print("\n📈 Model Performance:")
        for model_name in runner.models.keys():
            error_count = (results_df[f'{model_name}_prediction'] == 'Error').sum()
            success_rate = ((len(results_df) - error_count) / len(results_df)) * 100
            print(f"   • {model_name}: {success_rate:.1f}% success")
        
        # Show sample of results
        print(f"\n📋 Sample Results:")
        sample_cols = ['Quote_ID', 'ensemble_win_probability', 'ensemble_prediction', 
                      'ensemble_accurate_price', 'ensemble_elastic_price']
        available_cols = [col for col in sample_cols if col in results_df.columns]
        print(results_df[available_cols].head(3).to_string(index=False))
        
        return results_df
        
    except Exception as e:
        print(f"❌ Enhanced inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

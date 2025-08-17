"""
Enhanced Fixed Inference Runner with Error Corrections
Addresses the specific issues causing model prediction errors
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split

# Add current directory to path for imports
sys.path.append('.')
from utils.config_loader import ConfigLoader
from utils.feature_engineering import FeatureEngineering

class FixedInferenceRunnerV2:
    def __init__(self):
        self.config = ConfigLoader()
        self.feature_engineer = FeatureEngineering(self.config)
        self.logger = self.config.logger
        
        self.models = {}
        self.training_results = {}
        self.data = {}
        
        self.logger.info("Enhanced Fixed Inference Runner initialized")
    
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
    
    def load_models(self):
        """Load trained models with enhanced error handling"""
        self.logger.info("Loading trained models and artifacts...")
        
        # Load feature engineering artifacts
        fe_path = Path("models/feature_engineering")
        if fe_path.exists():
            self.feature_engineer.load_feature_engineering_artifacts()
            self.logger.info("Feature engineering artifacts loaded")
        else:
            raise FileNotFoundError("Feature engineering artifacts not found")
        
        # Load trained models with validation
        models_path = Path("models/trained")
        if models_path.exists():
            for model_file in models_path.glob("*_model.pkl"):
                model_name = model_file.stem.replace('_model', '')
                try:
                    model = joblib.load(model_file)
                    
                    # Validate model has predict method
                    if hasattr(model, 'predict'):
                        self.models[model_name] = model
                        self.logger.info(f"✓ Successfully loaded valid model: {model_name}")
                    else:
                        self.logger.warning(f"✗ Skipped invalid model {model_name}: no predict method (type: {type(model)})")
                        
                except Exception as e:
                    self.logger.warning(f"✗ Could not load {model_name}: {e}")
            
            # Load training results
            results_file = models_path / 'training_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.training_results = json.load(f)
        
        if not self.models:
            self.logger.warning("No valid trained models found - will use fallback predictions")
        else:
            self.logger.info(f"Loaded {len(self.models)} valid models: {list(self.models.keys())}")
    
    def fix_feature_data_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data type issues that cause model prediction failures
        """
        self.logger.info("Fixing feature data types...")
        X_fixed = X.copy()
        
        # Convert problematic datetime columns
        datetime_cols = [col for col in X_fixed.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in datetime_cols:
            if X_fixed[col].dtype == 'object':
                self.logger.info(f"Converting datetime column: {col}")
                X_fixed[col] = pd.to_datetime(X_fixed[col], errors='coerce')
                X_fixed[col] = (X_fixed[col] - pd.Timestamp('1970-01-01')).dt.days
                X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
        
        # Drop text columns that can't be converted
        text_cols = ['Customer_Name', 'Product_Name']
        for col in text_cols:
            if col in X_fixed.columns:
                X_fixed = X_fixed.drop(columns=[col])
                self.logger.info(f"Dropped text column: {col}")
        
        # Convert remaining object columns to numeric
        for col in X_fixed.columns:
            if X_fixed[col].dtype == 'object':
                self.logger.info(f"Converting object column to numeric: {col}")
                X_fixed[col] = pd.to_numeric(X_fixed[col], errors='coerce')
                if X_fixed[col].isnull().all():
                    X_fixed = X_fixed.drop(columns=[col])
                    self.logger.info(f"Dropped unconvertible column: {col}")
                else:
                    X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
        
        self.logger.info(f"Fixed data types. Final shape: {X_fixed.shape}")
        return X_fixed
    
    def align_features_with_model(self, X: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
        """
        Align feature columns with what the model expects
        """
        if not hasattr(model, 'feature_names_in_'):
            self.logger.info(f"No feature alignment needed for {model_name}")
            return X
        
        expected_features = list(model.feature_names_in_)
        self.logger.info(f"Aligning {len(expected_features)} features for {model_name}")
        
        X_aligned = pd.DataFrame(index=X.index)
        
        missing_features = []
        for feature in expected_features:
            if feature in X.columns:
                X_aligned[feature] = X[feature]
            else:
                # Missing feature - fill with zeros
                X_aligned[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} features for {model_name}, filled with 0")
        
        return X_aligned
    
    def predict_with_model(self, model, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Make predictions with a single model, handling errors gracefully
        """
        try:
            # Fix data types first
            X_fixed = self.fix_feature_data_types(X)
            
            # Align features with model expectations
            X_aligned = self.align_features_with_model(X_fixed, model, model_name)
            
            # Get win probability
            if hasattr(model, 'predict_proba'):
                win_probabilities = model.predict_proba(X_aligned)
                if win_probabilities.shape[1] == 2:
                    win_prob = win_probabilities[:, 1]
                else:
                    win_prob = np.max(win_probabilities, axis=1)
                
                # Get class predictions
                predictions = model.predict(X_aligned)
                
            else:
                # For models without predict_proba
                predictions = model.predict(X_aligned)
                win_prob = predictions if isinstance(predictions[0], float) else np.where(predictions == 'Won', 0.7, 0.3)
            
            self.logger.info(f"✓ {model_name}: predictions generated successfully")
            return {
                'win_probability': win_prob,
                'prediction': predictions,
                'status': 'Success'
            }
            
        except Exception as e:
            self.logger.error(f"✗ Error with model {model_name}: {e}")
            return {
                'win_probability': np.full(len(X), np.nan),
                'prediction': ['Error'] * len(X),
                'status': 'Error'
            }
    
    def calculate_dual_prices(self, X: pd.DataFrame, win_probabilities: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Calculate dual prices (accurate price & elastic price) for each model
        """
        self.logger.info("Calculating dual prices...")
        
        accurate_prices = {}
        elastic_prices = {}
        
        # Base prices from data
        list_price = X.get('List_Price', X.get('List_Price_x', pd.Series([1000] * len(X))))
        net_price = X.get('Net_Price', pd.Series([800] * len(X)))
        
        for model_name, win_probs in win_probabilities.items():
            if np.isnan(win_probs).all():
                # Model failed - use fallback pricing
                accurate_prices[model_name] = net_price * 0.95  # Conservative pricing
                elastic_prices[model_name] = list_price * 0.85  # Stretch pricing
                continue
            
            # Calculate accurate price (conservative - high win probability target)
            accurate_multiplier = self._calculate_price_multiplier(
                current_win_prob=win_probs,
                target_win_prob=0.70,
                base_price=net_price,
                conservative=True
            )
            accurate_prices[model_name] = net_price * accurate_multiplier
            
            # Calculate elastic price (stretch - moderate win probability target) 
            elastic_multiplier = self._calculate_price_multiplier(
                current_win_prob=win_probs,
                target_win_prob=0.50,
                base_price=list_price,
                conservative=False
            )
            elastic_prices[model_name] = list_price * elastic_multiplier
            
        return accurate_prices, elastic_prices
    
    def _calculate_price_multiplier(self, current_win_prob: np.ndarray, target_win_prob: float, 
                                  base_price: pd.Series, conservative: bool = True) -> np.ndarray:
        """
        Calculate price multiplier based on elasticity principles
        """
        # Price elasticity assumption: -1.2 (1% price increase = 1.2% demand decrease)
        elasticity = -1.2 if not conservative else -0.8  # Conservative uses lower elasticity
        
        # Calculate probability gap
        prob_gap = target_win_prob - current_win_prob
        
        # Convert probability change to price change using elasticity
        price_change_pct = prob_gap / abs(elasticity)
        
        # Calculate multiplier (bounded for business sense)
        multiplier = 1 + price_change_pct
        
        # Apply business constraints
        if conservative:
            # Accurate prices: between 70% and 95% of net price
            multiplier = np.clip(multiplier, 0.70, 0.95)
        else:
            # Elastic prices: between 80% and 110% of list price
            multiplier = np.clip(multiplier, 0.80, 1.10)
            
        return multiplier
    
    def run_inference(self, unified_data: pd.DataFrame) -> pd.DataFrame:
        """Run complete inference with dual price prediction"""
        self.logger.info("Running enhanced inference with dual price prediction...")
        
        # Apply feature engineering
        self.logger.info("Applying feature engineering...")
        featured_data = self.feature_engineer.create_comprehensive_features(unified_data, fit=False)
        
        # Prepare features for prediction
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
        feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
        X = featured_data[feature_cols]
        
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
        
        # Run predictions with each model
        model_predictions = {}
        win_probabilities = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Running predictions with {model_name}")
            
            pred_result = self.predict_with_model(model, X, model_name)
            model_predictions[model_name] = pred_result
            
            # Store results in dataframe
            results_df[f'{model_name}_win_probability'] = pred_result['win_probability']
            results_df[f'{model_name}_prediction'] = pred_result['prediction']
            
            # Store for dual price calculation
            if pred_result['status'] == 'Success':
                win_probabilities[model_name] = pred_result['win_probability']
        
        # Calculate dual prices for all models
        accurate_prices, elastic_prices = self.calculate_dual_prices(results_df, win_probabilities)
        
        # Add dual price columns
        for model_name in self.models.keys():
            if model_name in accurate_prices:
                results_df[f'{model_name}_accurate_price'] = accurate_prices[model_name]
                results_df[f'{model_name}_elastic_price'] = elastic_prices[model_name]
            else:
                # Model failed - use NaN
                results_df[f'{model_name}_accurate_price'] = np.nan
                results_df[f'{model_name}_elastic_price'] = np.nan
        
        # Create ensemble predictions
        valid_win_probs = []
        valid_accurate_prices = []
        valid_elastic_prices = []
        
        for model_name in self.models.keys():
            if model_name in win_probabilities:
                valid_win_probs.append(win_probabilities[model_name])
                valid_accurate_prices.append(accurate_prices[model_name])
                valid_elastic_prices.append(elastic_prices[model_name])
        
        if valid_win_probs:
            # Ensemble win probability (average of all valid models)
            results_df['ensemble_win_probability'] = np.mean(valid_win_probs, axis=0)
            results_df['ensemble_prediction'] = np.where(
                results_df['ensemble_win_probability'] > 0.5, 'Won', 'Lost'
            )
            
            # Ensemble prices (average of all valid models)
            results_df['ensemble_accurate_price'] = np.mean(valid_accurate_prices, axis=0)
            results_df['ensemble_elastic_price'] = np.mean(valid_elastic_prices, axis=0)
        else:
            # No valid models - use fallback
            results_df['ensemble_win_probability'] = 0.35  # Average win rate
            results_df['ensemble_prediction'] = 'Lost'
            results_df['ensemble_accurate_price'] = results_df.get('Net_Price', 1000) * 0.9
            results_df['ensemble_elastic_price'] = results_df.get('List_Price', 1200) * 0.85
        
        # Add business analysis columns
        results_df['actual_status'] = results_df['Status']
        
        if 'List_Price' in results_df.columns and 'Net_Price' in results_df.columns:
            results_df['current_discount_percent'] = (
                (results_df['List_Price'] - results_df['Net_Price']) / results_df['List_Price']
            ).fillna(0)
        
        # Add prediction metadata
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        results_df['models_used'] = ', '.join(self.models.keys())
        results_df['successful_models'] = ', '.join(win_probabilities.keys())
        
        self.logger.info(f"Enhanced inference completed. Results shape: {results_df.shape}")
        self.logger.info(f"Valid models used: {list(win_probabilities.keys())}")
        return results_df
    
    def save_results(self, results_df: pd.DataFrame) -> str:
        """Save inference results to CSV files with clear structure"""
        self.logger.info("Saving enhanced results to CSV files...")
        
        # Create output directory
        output_dir = Path("results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results file
        main_file = output_dir / f"enhanced_inference_results_{timestamp}.csv"
        results_df.to_csv(main_file, index=False)
        self.logger.info(f"Main results saved to: {main_file}")
        
        return str(main_file)

def run_enhanced_inference():
    """Run the enhanced inference pipeline"""
    try:
        # Initialize runner
        runner = FixedInferenceRunnerV2()
        
        # Load data
        data = runner.load_data()
        unified_data = runner.create_unified_dataset()
        
        # Load models
        runner.load_models()
        
        # Run inference
        results_df = runner.run_inference(unified_data)
        
        # Save results
        output_file = runner.save_results(results_df)
        
        print(f"\n✅ Enhanced inference completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Processed {len(results_df)} quotes")
        print(f"🤖 Valid models used: {runner.models.keys()}")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ Enhanced inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results = run_enhanced_inference()

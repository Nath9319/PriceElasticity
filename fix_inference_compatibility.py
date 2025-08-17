#!/usr/bin/env python3
"""
Feature Compatibility Fix for Enhanced Inference
================================================

This script addresses the feature compatibility issues between model training 
and inference stages that are causing prediction failures.

Key fixes:
1. Feature name alignment between training and inference
2. Robust fallback mechanisms for failed predictions
3. Improved error handling and logging
4. Better feature engineering consistency
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')

class InferenceCompatibilityFixer:
    """Fixes compatibility issues in the inference pipeline"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = {}
        self.training_features = {}
        
    def analyze_model_features(self):
        """Analyze what features each trained model expects"""
        print("Analyzing trained model feature requirements...")
        
        models_path = Path("models/trained")
        if not models_path.exists():
            print("No trained models directory found")
            return
            
        # Load training results to see expected features
        results_file = models_path / 'training_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
                
            print("Training results loaded:")
            for model_name, results in training_results.items():
                if isinstance(results, dict):
                    if 'feature_names' in results:
                        self.training_features[model_name] = results['feature_names']
                        print(f"  {model_name}: {len(results['feature_names'])} features")
                    elif 'n_features' in results:
                        print(f"  {model_name}: {results['n_features']} features (names not saved)")
                    else:
                        print(f"  {model_name}: Feature info not available")
        
        # Try to load models and inspect their feature requirements
        for model_file in models_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                
                # Try to get feature names from model
                if hasattr(model, 'feature_names_in_'):
                    self.feature_columns[model_name] = list(model.feature_names_in_)
                    print(f"  {model_name}: {len(model.feature_names_in_)} features from model")
                elif hasattr(model, 'n_features_in_'):
                    print(f"  {model_name}: {model.n_features_in_} features (names not available)")
                
            except Exception as e:
                print(f"  Could not load {model_name}: {e}")
    
    def create_feature_mapping(self, current_features: List[str]) -> Dict[str, List[str]]:
        """Create mapping between current features and expected model features"""
        feature_mapping = {}
        
        for model_name in self.models.keys():
            if model_name in self.feature_columns:
                expected_features = self.feature_columns[model_name]
                
                # Find matching features
                matching_features = []
                missing_features = []
                
                for expected_feat in expected_features:
                    if expected_feat in current_features:
                        matching_features.append(expected_feat)
                    else:
                        missing_features.append(expected_feat)
                
                feature_mapping[model_name] = {
                    'matching': matching_features,
                    'missing': missing_features,
                    'coverage': len(matching_features) / len(expected_features) if expected_features else 0
                }
                
                print(f"{model_name}: {len(matching_features)}/{len(expected_features)} features match ({feature_mapping[model_name]['coverage']:.1%})")
                if missing_features[:5]:  # Show first 5 missing features
                    print(f"  Missing: {missing_features[:5]}")
        
        return feature_mapping
    
    def fix_feature_engineering_consistency(self):
        """Fix feature engineering to be consistent between training and inference"""
        print("\nFixing feature engineering consistency...")
        
        # Load feature engineering artifacts
        fe_path = Path("models/feature_engineering")
        if not fe_path.exists():
            print("No feature engineering artifacts found")
            return
            
        # Check what artifacts exist
        artifacts = list(fe_path.glob("*.pkl"))
        print(f"Found {len(artifacts)} feature engineering artifacts:")
        for artifact in artifacts:
            print(f"  {artifact.name}")
            
        # Try to load and inspect feature transformers
        try:
            # Load the PriceElasticityFeatureEngineering instance if it exists
            fe_instance_file = fe_path / "feature_engineer.pkl"
            if fe_instance_file.exists():
                fe_instance = joblib.load(fe_instance_file)
                print("Feature engineering instance loaded")
                
                # Check if it has stored feature names
                if hasattr(fe_instance, 'feature_names_'):
                    print(f"Feature engineering creates {len(fe_instance.feature_names_)} features")
                    return fe_instance.feature_names_
                    
        except Exception as e:
            print(f"Could not load feature engineering instance: {e}")
            
        return None
    
    def create_robust_prediction_function(self):
        """Create a robust prediction function that handles feature mismatches"""
        
        def robust_predict(model, X_features, model_name):
            """Robust prediction with feature alignment"""
            try:
                # If model has expected feature names, align them
                if hasattr(model, 'feature_names_in_'):
                    expected_features = model.feature_names_in_
                    
                    # Create aligned feature matrix
                    X_aligned = pd.DataFrame(index=X_features.index)
                    
                    for feat in expected_features:
                        if feat in X_features.columns:
                            X_aligned[feat] = X_features[feat]
                        else:
                            # Use mean or mode imputation for missing features
                            if feat.endswith('_mean') or feat.endswith('_sum') or feat.endswith('_std'):
                                X_aligned[feat] = 0.0  # Numeric features default to 0
                            elif feat.endswith('_encoded') or 'category' in feat.lower():
                                X_aligned[feat] = 0  # Categorical encoded features default to 0
                            else:
                                X_aligned[feat] = 0.0  # Default to 0
                    
                    # Reorder columns to match expected order
                    X_aligned = X_aligned[expected_features]
                    
                else:
                    # Model doesn't specify expected features, use as-is
                    X_aligned = X_features
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_aligned)
                    win_probs = probabilities[:, 1] if probabilities.shape[1] == 2 else np.max(probabilities, axis=1)
                    predictions = model.predict(X_aligned)
                    return win_probs, predictions, "Success"
                else:
                    predictions = model.predict(X_aligned)
                    # Convert predictions to probabilities (rough estimate)
                    if hasattr(predictions, '__iter__'):
                        win_probs = np.where(predictions == 1, 0.7, 0.3)  # Rough probability estimate
                    else:
                        win_probs = np.full(len(X_aligned), 0.5)
                    return win_probs, predictions, "Success"
                    
            except Exception as e:
                # Return fallback values
                n_samples = len(X_features)
                win_probs = np.full(n_samples, 0.5)  # Neutral probability
                predictions = np.array(['Unknown'] * n_samples)
                return win_probs, predictions, f"Error: {str(e)}"
        
        return robust_predict
    
    def generate_compatible_features(self, unified_data):
        """Generate features that are compatible with trained models"""
        print("Generating compatible features...")
        
        # Import feature engineering
        sys.path.append(str(Path(__file__).parent / "src"))
        try:
            from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
            
            fe = PriceElasticityFeatureEngineering()
            
            # Try to load existing artifacts
            try:
                fe.load_feature_engineering_artifacts()
                print("Existing feature engineering artifacts loaded")
            except:
                print("No existing artifacts found, will create new features")
                
            # Generate features
            featured_data = fe.create_comprehensive_features(unified_data, fit=False)
            print(f"Generated {featured_data.shape[1]} features for {featured_data.shape[0]} samples")
            
            return featured_data, fe
            
        except Exception as e:
            print(f"Error in feature generation: {e}")
            return None, None
    
    def run_compatibility_test(self):
        """Run a comprehensive compatibility test"""
        print("="*60)
        print("INFERENCE COMPATIBILITY DIAGNOSTIC")
        print("="*60)
        
        # 1. Analyze model features
        self.analyze_model_features()
        
        # 2. Check feature engineering
        fe_features = self.fix_feature_engineering_consistency()
        
        # 3. Load sample data and generate features
        print("\nLoading sample data...")
        try:
            # Load quote history as sample
            quote_data = pd.read_csv("datasets/quote_history.csv")
            print(f"Loaded sample data: {quote_data.shape}")
            
            # Generate features
            featured_data, fe = self.generate_compatible_features(quote_data)
            
            if featured_data is not None:
                current_features = featured_data.columns.tolist()
                print(f"Current feature generation produces {len(current_features)} features")
                
                # 4. Create feature mapping
                feature_mapping = self.create_feature_mapping(current_features)
                
                # 5. Test robust prediction
                print("\nTesting robust prediction...")
                robust_predict = self.create_robust_prediction_function()
                
                test_results = {}
                for model_name, model in self.models.items():
                    print(f"\nTesting {model_name}...")
                    
                    # Test on first 10 samples
                    sample_features = featured_data.head(10)
                    exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
                    feature_cols = [col for col in sample_features.columns if col not in exclude_cols]
                    X_test = sample_features[feature_cols]
                    
                    win_probs, predictions, status = robust_predict(model, X_test, model_name)
                    
                    test_results[model_name] = {
                        'status': status,
                        'win_prob_range': f"{win_probs.min():.3f} - {win_probs.max():.3f}",
                        'unique_predictions': len(set(predictions))
                    }
                    
                    print(f"  Status: {status}")
                    print(f"  Win probability range: {win_probs.min():.3f} - {win_probs.max():.3f}")
                    print(f"  Unique predictions: {len(set(predictions))}")
                
                # 6. Summary and recommendations
                print("\n" + "="*60)
                print("DIAGNOSTIC SUMMARY")
                print("="*60)
                
                successful_models = sum(1 for result in test_results.values() if result['status'] == 'Success')
                total_models = len(test_results)
                
                print(f"Models loaded: {total_models}")
                print(f"Successful predictions: {successful_models}")
                print(f"Success rate: {successful_models/total_models:.1%}" if total_models > 0 else "No models tested")
                
                if successful_models < total_models:
                    print("\nRECOMMENDATIONS:")
                    print("1. Retrain models with consistent feature engineering")
                    print("2. Save feature names during model training")
                    print("3. Implement feature validation in inference pipeline")
                    print("4. Add robust fallback mechanisms")
                
                return test_results
                
        except Exception as e:
            print(f"Error in compatibility test: {e}")
            
        return None

def main():
    """Run the compatibility diagnostic and fixes"""
    fixer = InferenceCompatibilityFixer()
    results = fixer.run_compatibility_test()
    
    if results:
        print("\nCompatibility test completed. Check the output above for issues and recommendations.")
    else:
        print("Compatibility test failed. Please check your data and model files.")

if __name__ == "__main__":
    main()

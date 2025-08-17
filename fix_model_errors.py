"""
Fix the model prediction errors identified in the analysis
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils.config_loader import ConfigLoader
from utils.feature_engineering import FeatureEngineering
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_feature_data_types(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data type issues that cause model prediction failures
    """
    logger.info("Fixing feature data types...")
    X_fixed = X.copy()
    
    # Convert object columns that should be numeric
    problematic_cols = ['hist_Sale_Date_max', 'Customer_Name', 'Product_Name']
    
    for col in problematic_cols:
        if col in X_fixed.columns:
            logger.info(f"Fixing column: {col}")
            if col == 'hist_Sale_Date_max':
                # Convert datetime to numeric (days since epoch)
                X_fixed[col] = pd.to_datetime(X_fixed[col], errors='coerce')
                X_fixed[col] = (X_fixed[col] - pd.Timestamp('1970-01-01')).dt.days
                X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
            
            elif col in ['Customer_Name', 'Product_Name']:
                # Convert text to target encoding or drop if not needed
                X_fixed = X_fixed.drop(columns=[col])
                logger.info(f"Dropped text column: {col}")
    
    # Ensure all remaining columns are numeric
    for col in X_fixed.columns:
        if X_fixed[col].dtype == 'object':
            logger.warning(f"Found remaining object column: {col}, attempting conversion")
            # Try to convert to numeric, replace with median if fails
            X_fixed[col] = pd.to_numeric(X_fixed[col], errors='coerce')
            if X_fixed[col].isnull().all():
                X_fixed = X_fixed.drop(columns=[col])
                logger.info(f"Dropped unconvertible column: {col}")
            else:
                X_fixed[col] = X_fixed[col].fillna(X_fixed[col].median())
    
    logger.info(f"Fixed data types. Shape: {X_fixed.shape}")
    return X_fixed

def get_model_expected_features(model_name: str) -> list:
    """
    Get the feature names that each model expects based on training
    """
    models_path = Path("models/trained")
    model_file = models_path / f"{model_name}_model.pkl"
    
    try:
        model = joblib.load(model_file)
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'feature_importances_') and hasattr(model, 'n_features_'):
            # Generate feature names if not stored
            return [f"feature_{i}" for i in range(model.n_features_)]
    except:
        pass
    
    return None

def align_features_with_model(X: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Align feature columns with what the model expects
    """
    if expected_features is None:
        return X
    
    X_aligned = pd.DataFrame(index=X.index)
    
    for feature in expected_features:
        if feature in X.columns:
            X_aligned[feature] = X[feature]
        else:
            # Missing feature - fill with zeros or median
            X_aligned[feature] = 0
            logger.warning(f"Missing expected feature '{feature}', filled with 0")
    
    logger.info(f"Aligned features. Shape: {X_aligned.shape}")
    return X_aligned

def test_fixed_prediction():
    """
    Test the fixed prediction pipeline
    """
    logger.info("Testing fixed prediction pipeline...")
    
    # Initialize components
    config = ConfigLoader()
    feature_engineer = FeatureEngineering(config)
    
    # Load test data (use smaller sample)
    data_config = config.get_data_config()
    datasets_path = Path(data_config['datasets_path'])
    
    quote_history = pd.read_csv(datasets_path / "quote_history.csv").head(100)  # Small sample
    
    logger.info(f"Testing with {len(quote_history)} samples")
    
    # Load feature engineering artifacts
    fe_path = Path("models/feature_engineering")
    if fe_path.exists():
        feature_engineer.load_feature_engineering_artifacts()
    
    # Create features
    try:
        featured_data = feature_engineer.create_comprehensive_features(quote_history, fit=False)
        logger.info(f"Features created: {featured_data.shape}")
        
        # Prepare for prediction
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
        feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
        X = featured_data[feature_cols]
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Data types: {X.dtypes.value_counts()}")
        
        # Fix data types
        X_fixed = fix_feature_data_types(X)
        
        # Test each model
        models_path = Path("models/trained")
        for model_file in models_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            
            try:
                model = joblib.load(model_file)
                
                if hasattr(model, 'predict'):
                    logger.info(f"Testing {model_name}...")
                    
                    # Get expected features
                    expected_features = get_model_expected_features(model_name)
                    if expected_features:
                        X_aligned = align_features_with_model(X_fixed, expected_features)
                    else:
                        X_aligned = X_fixed
                    
                    # Test prediction
                    predictions = model.predict(X_aligned)
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_aligned)
                        logger.info(f"✅ {model_name}: predictions shape {predictions.shape}, probabilities shape {probabilities.shape}")
                    else:
                        logger.info(f"✅ {model_name}: predictions shape {predictions.shape}")
                        
                else:
                    logger.warning(f"❌ {model_name}: Not a valid ML model (type: {type(model)})")
                    
            except Exception as e:
                logger.error(f"❌ {model_name}: {e}")
    
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    test_fixed_prediction()

import joblib
from pathlib import Path

# Check what features each model expects
models_path = Path("models/trained")

print("=== MODEL FEATURE REQUIREMENTS ===\n")

for model_file in models_path.glob("*_model.pkl"):
    model_name = model_file.stem.replace('_model', '')
    print(f"--- {model_name.upper()} ---")
    
    try:
        model = joblib.load(model_file)
        
        if hasattr(model, 'predict'):
            print(f"✓ Valid ML model: {type(model).__name__}")
            
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
                print(f"Expected features ({len(features)}): {features[:5]}...")
                if len(features) > 5:
                    print(f"...and {len(features)-5} more")
            else:
                print("No feature names stored")
                
            if hasattr(model, 'n_features_in_'):
                print(f"Expected feature count: {model.n_features_in_}")
                
        else:
            print(f"✗ Invalid model: {type(model)} (missing predict method)")
            if isinstance(model, dict):
                print(f"  Dictionary keys: {list(model.keys())}")
        
    except Exception as e:
        print(f"✗ Error loading: {e}")
    
    print()

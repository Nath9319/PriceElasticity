import joblib
import pandas as pd
from pathlib import Path

# Load and inspect each model
models_path = Path("models/trained")

for model_file in models_path.glob("*_model.pkl"):
    model_name = model_file.stem.replace('_model', '')
    print(f"\n=== {model_name.upper()} MODEL ===")
    
    try:
        model = joblib.load(model_file)
        print(f"Type: {type(model)}")
        print(f"Size: {model_file.stat().st_size} bytes")
        
        if hasattr(model, 'predict'):
            print("✓ Has predict method")
        else:
            print("✗ Missing predict method")
            
        if hasattr(model, 'predict_proba'):
            print("✓ Has predict_proba method")
        else:
            print("✗ Missing predict_proba method")
            
        # If it's a dict, show keys
        if isinstance(model, dict):
            print(f"Dictionary keys: {list(model.keys())}")
            
        # Show first few attributes/methods
        if hasattr(model, '__dict__'):
            attrs = [attr for attr in dir(model) if not attr.startswith('_')][:10]
            print(f"Methods/attributes: {attrs}")
            
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

print("\n=== FEATURE ENGINEERING ARTIFACTS ===")
fe_path = Path("models/feature_engineering")
if fe_path.exists():
    for file in fe_path.glob("*.pkl"):
        print(f"Found: {file.name}")
        try:
            artifact = joblib.load(file)
            print(f"  Type: {type(artifact)}")
            if isinstance(artifact, dict):
                print(f"  Keys: {list(artifact.keys())}")
        except Exception as e:
            print(f"  Error: {e}")
else:
    print("Feature engineering path not found")

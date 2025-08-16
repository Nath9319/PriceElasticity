#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from utils.config_loader import config_loader

# Load and merge data as per the main pipeline
config = config_loader
data = {}

# Load datasets
data_config = config.get_data_config()
datasets_path = Path(data_config['datasets_path'])
files = data_config['files']

for name, filename in files.items():
    file_path = datasets_path / filename
    if file_path.exists():
        data[name] = pd.read_csv(file_path)
        print(f"Loaded {name}: {data[name].shape}")

# Create unified dataset
unified = data['quote_history'].copy()

# Join with sales_history
if 'sales_history' in data:
    sales_agg = data['sales_history'].groupby(['Customer_ID', 'Product_ID']).agg({
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
    if table_name in data:
        unified = unified.merge(data[table_name], on='Customer_ID', how='left')

# Join with product_master
if 'product_master' in data:
    unified = unified.merge(data['product_master'], on='Product_ID', how='left')

print(f"\nUnified dataset shape: {unified.shape}")
print(f"\nUnified dataset columns: {unified.columns.tolist()}")

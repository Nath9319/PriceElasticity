#!/usr/bin/env python3
"""
Quick Data Inspector
==================

Inspect the actual data structure to understand column names and fix compatibility issues.
"""

import pandas as pd
from pathlib import Path

def inspect_datasets():
    """Inspect all datasets to understand their structure"""
    print("DATASET INSPECTION")
    print("="*50)
    
    datasets_path = Path("datasets")
    files = [
        'quote_history.csv',
        'customer_master.csv', 
        'customer_segmentation.csv',
        'product_master.csv',
        'sales_history.csv'
    ]
    
    data = {}
    for filename in files:
        file_path = datasets_path / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            data[filename.replace('.csv', '')] = df
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample values:")
            for col in df.columns[:5]:  # First 5 columns
                sample_vals = df[col].dropna().unique()[:3]
                print(f"    {col}: {sample_vals}")
        else:
            print(f"\n{filename}: NOT FOUND")
    
    # Create unified dataset to see resulting columns
    print("\nUNIFIED DATASET:")
    print("="*30)
    
    if 'quote_history' in data:
        unified = data['quote_history'].copy()
        print(f"Starting with quote_history: {unified.shape}")
        print(f"Initial columns: {list(unified.columns)}")
        
        # Join with customer tables
        if 'customer_master' in data:
            pre_merge = unified.shape
            unified = unified.merge(data['customer_master'], on='Customer_ID', how='left')
            print(f"After customer_master merge: {pre_merge} -> {unified.shape}")
            
        if 'customer_segmentation' in data:
            pre_merge = unified.shape
            unified = unified.merge(data['customer_segmentation'], on='Customer_ID', how='left')
            print(f"After customer_segmentation merge: {pre_merge} -> {unified.shape}")
        
        # Join with product_master
        if 'product_master' in data:
            pre_merge = unified.shape
            unified = unified.merge(data['product_master'], on='Product_ID', how='left')
            print(f"After product_master merge: {pre_merge} -> {unified.shape}")
        
        print(f"\nFinal unified columns: {list(unified.columns)}")
        
        # Check for List_Price and Product_Category specifically
        print(f"\nKey columns check:")
        print(f"  'List_Price' in columns: {'List_Price' in unified.columns}")
        print(f"  'Product_Category' in columns: {'Product_Category' in unified.columns}")
        
        # Look for similar column names
        price_cols = [col for col in unified.columns if 'price' in col.lower()]
        category_cols = [col for col in unified.columns if 'category' in col.lower()]
        print(f"  Price-related columns: {price_cols}")
        print(f"  Category-related columns: {category_cols}")
        
        return unified
    
    return None

if __name__ == "__main__":
    unified_df = inspect_datasets()

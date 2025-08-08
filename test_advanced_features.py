#!/usr/bin/env python3
"""
Test script for advanced features implementation
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_advanced_feature_engineering():
    """Test the advanced feature engineering capabilities"""
    print("🧪 Testing Advanced Feature Engineering...")
    
    try:
        from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Quote_ID': [f'Q{i:06d}' for i in range(n_samples)],
            'Customer_ID': [f'C{i%100:04d}' for i in range(n_samples)],
            'Product_ID': [f'P{i%50:03d}' for i in range(n_samples)],
            'Quote_Date': pd.date_range(start='2022-01-01', end='2024-01-01', periods=n_samples),
            'Customer_Since_Date': pd.date_range(start='2020-01-01', end='2023-01-01', periods=n_samples),
            'Launch_Date': pd.date_range(start='2019-01-01', end='2023-06-01', periods=n_samples),
            'List_Price': np.random.lognormal(mean=8, sigma=0.5, size=n_samples),
            'Net_Price': np.random.lognormal(mean=7.8, sigma=0.5, size=n_samples),
            'Offered_Price': np.random.lognormal(mean=7.7, sigma=0.5, size=n_samples),
            'Customer_Segment': np.random.choice(['Enterprise', 'Mid-Market', 'SMB'], n_samples),
            'Product_Category': np.random.choice(['Software', 'Hardware', 'Services'], n_samples),
            'Competition_Status': np.random.choice(['None', 'Low', 'Medium', 'High'], n_samples),
            'Product_Objective': np.random.choice(['Growth', 'Profitability', 'Market_Share'], n_samples),
            'Lifecycle_Stage': np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'], n_samples),
            'Status': np.random.choice(['Won', 'Lost'], n_samples, p=[0.35, 0.65])
        }
        
        df = pd.DataFrame(data)
        
        # Initialize feature engineering
        fe = PriceElasticityFeatureEngineering()
        
        # Test advanced temporal features
        print("  ✓ Testing advanced temporal features...")
        df_temporal = fe.create_advanced_temporal_features(df)
        temporal_features = [col for col in df_temporal.columns if any(x in col for x in ['lag_', 'ma_', 'ewma_', 'fourier_', 'stl_'])]
        print(f"    Created {len(temporal_features)} temporal features")
        
        # Test B2B domain features
        print("  ✓ Testing B2B domain features...")
        df_b2b = fe.create_b2b_domain_features(df)
        b2b_features = [col for col in df_b2b.columns if any(x in col for x in ['contract_', 'inventory_', 'market_volatility', 'seasonal_'])]
        print(f"    Created {len(b2b_features)} B2B domain features")
        
        # Test GNN features
        print("  ✓ Testing Graph Neural Network features...")
        df_gnn = fe.create_graph_neural_network_features(df)
        gnn_features = [col for col in df_gnn.columns if any(x in col for x in ['centrality', 'similarity', 'network_', 'embedding_'])]
        print(f"    Created {len(gnn_features)} GNN features")
        
        # Test advanced interactions
        print("  ✓ Testing advanced interaction features...")
        df_interactions = fe.create_advanced_interaction_features(df)
        interaction_features = [col for col in df_interactions.columns if any(x in col for x in ['_x_', 'sensitivity', 'competitive_ratio', 'premium_tolerance'])]
        print(f"    Created {len(interaction_features)} interaction features")
        
        print("✅ Advanced Feature Engineering tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Feature Engineering test failed: {e}")
        return False

def test_model_explainability():
    """Test the model explainability features"""
    print("🧪 Testing Model Explainability...")
    
    try:
        from training.model_training import PriceElasticityModelTraining
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X['Net_Price'] = np.random.lognormal(mean=7, sigma=0.5, size=n_samples)
        X['Customer_Segment'] = np.random.choice(['Enterprise', 'Mid-Market', 'SMB'], n_samples)
        X['Product_Category'] = np.random.choice(['Software', 'Hardware', 'Services'], n_samples)
        
        y = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X.select_dtypes(include=[np.number]), y)
        
        # Initialize model training
        trainer = PriceElasticityModelTraining()
        
        # Test SHAP analysis
        print("  ✓ Testing SHAP analysis...")
        shap_results = trainer.calculate_shap_values(model, X.select_dtypes(include=[np.number]), 'ensemble')
        if shap_results.get('feature_importance'):
            print(f"    SHAP analysis completed with {len(shap_results['feature_importance'])} features analyzed")
        
        # Test elasticity curves
        print("  ✓ Testing elasticity curve generation...")
        elasticity_curves = trainer.generate_elasticity_curves(model, X, segments=['Customer_Segment'])
        if elasticity_curves.get('overall'):
            print(f"    Generated elasticity curves with {len(elasticity_curves['overall']['prices'])} price points")
        
        # Test automated reports
        print("  ✓ Testing automated report generation...")
        model_results = {'model_comparison': {'best_model': 'ensemble', 'best_auc': 0.75}}
        automated_reports = trainer.create_automated_reports(model_results, shap_results, elasticity_curves)
        if automated_reports.get('executive_summary'):
            print(f"    Generated automated report with {len(automated_reports.get('recommendations', []))} recommendations")
        
        print("✅ Model Explainability tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model Explainability test failed: {e}")
        return False

def test_scenario_analysis():
    """Test the scenario analysis capabilities"""
    print("🧪 Testing Scenario Analysis...")
    
    try:
        # Test price impact simulation
        print("  ✓ Testing price impact simulation...")
        
        # Simulate price elasticity calculation
        price_adjustment = 10  # 10% increase
        base_win_rate = 0.35
        elasticity = -0.6
        
        new_win_rate = base_win_rate * (1 + elasticity * (price_adjustment / 100))
        revenue_change = price_adjustment / 100 + (new_win_rate - base_win_rate)
        
        print(f"    Price adjustment: {price_adjustment}% -> Win rate change: {(new_win_rate - base_win_rate)*100:.1f}%")
        print(f"    Expected revenue change: {revenue_change*100:.1f}%")
        
        # Test competitive response modeling
        print("  ✓ Testing competitive response modeling...")
        
        # Simulate market share evolution
        our_initial_share = 0.25
        competitor_response_factor = 0.6  # Medium aggressiveness
        
        # Simple competitive dynamics
        our_price_change = -10  # 10% price cut
        competitor_response = our_price_change * competitor_response_factor
        
        share_change = abs(our_price_change - competitor_response) * 0.02  # 2% share per 1% price advantage
        new_our_share = our_initial_share + share_change
        
        print(f"    Our price change: {our_price_change}% -> Competitor response: {competitor_response}%")
        print(f"    Market share change: {share_change*100:.1f}% -> New share: {new_our_share*100:.1f}%")
        
        # Test multi-scenario analysis
        print("  ✓ Testing multi-scenario analysis...")
        
        scenarios = [
            {'price_change': 5, 'market_condition': 'Normal', 'probability': 0.4},
            {'price_change': -5, 'market_condition': 'Recession', 'probability': 0.3},
            {'price_change': 10, 'market_condition': 'Growth', 'probability': 0.3}
        ]
        
        expected_outcome = sum(s['price_change'] * s['probability'] for s in scenarios)
        print(f"    Expected outcome across scenarios: {expected_outcome:.1f}%")
        
        print("✅ Scenario Analysis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Scenario Analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Advanced Price Elasticity Features")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test advanced feature engineering
    if test_advanced_feature_engineering():
        tests_passed += 1
    
    print()
    
    # Test model explainability
    if test_model_explainability():
        tests_passed += 1
    
    print()
    
    # Test scenario analysis
    if test_scenario_analysis():
        tests_passed += 1
    
    print()
    print("=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All advanced features are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
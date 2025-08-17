"""
Comprehensive Analysis of Enhanced Fixed Inference Results
Analyzes the CSV output from the enhanced inference runner
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_inference_results():
    """Comprehensive analysis of the inference results"""
    
    # Load the latest results
    results_file = "results/inference/enhanced_fixed_inference_20250817_213513.csv"
    df = pd.read_csv(results_file)
    
    print("=" * 80)
    print("🔍 COMPREHENSIVE INFERENCE RESULTS ANALYSIS")
    print("=" * 80)
    
    # 1. BASIC STATISTICS
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   • Total Quotes Processed: {len(df):,}")
    print(f"   • Date Range: {df['Quote_Date'].min()} to {df['Quote_Date'].max()}")
    print(f"   • Unique Customers: {df['Customer_ID'].nunique():,}")
    print(f"   • Unique Products: {df['Product_ID'].nunique():,}")
    
    # 2. MODEL PERFORMANCE ANALYSIS
    print(f"\n🤖 MODEL PERFORMANCE ANALYSIS")
    
    # Individual model success rates
    ensemble_success = (df['ensemble_prediction'] != 'Error').sum()
    gnn_success = (df['graph_neural_network_prediction'] != 'Error').sum()
    
    print(f"   • Ensemble Model Success Rate: {ensemble_success/len(df)*100:.1f}% ({ensemble_success:,}/{len(df):,})")
    print(f"   • Graph Neural Network Success Rate: {gnn_success/len(df)*100:.1f}% ({gnn_success:,}/{len(df):,})")
    
    # Win probability statistics
    print(f"\n📈 WIN PROBABILITY STATISTICS")
    print(f"   • Ensemble Win Probability:")
    print(f"     - Mean: {df['ensemble_win_probability'].mean():.3f}")
    print(f"     - Median: {df['ensemble_win_probability'].median():.3f}")
    print(f"     - Min: {df['ensemble_win_probability'].min():.3f}")
    print(f"     - Max: {df['ensemble_win_probability'].max():.3f}")
    print(f"     - Std: {df['ensemble_win_probability'].std():.3f}")
    
    print(f"   • Graph Neural Network Win Probability:")
    print(f"     - Mean: {df['graph_neural_network_win_probability'].mean():.3f}")
    print(f"     - Median: {df['graph_neural_network_win_probability'].median():.3f}")
    print(f"     - Min: {df['graph_neural_network_win_probability'].min():.3f}")
    print(f"     - Max: {df['graph_neural_network_win_probability'].max():.3f}")
    print(f"     - Std: {df['graph_neural_network_win_probability'].std():.3f}")
    
    # 3. PRICING ANALYSIS
    print(f"\n💰 DUAL PRICING ANALYSIS")
    
    # Ensemble pricing
    print(f"   • Ensemble Accurate Price (Conservative):")
    print(f"     - Mean: ${df['ensemble_accurate_price'].mean():,.2f}")
    print(f"     - Median: ${df['ensemble_accurate_price'].median():,.2f}")
    print(f"     - Range: ${df['ensemble_accurate_price'].min():,.2f} - ${df['ensemble_accurate_price'].max():,.2f}")
    
    print(f"   • Ensemble Elastic Price (Stretch):")
    print(f"     - Mean: ${df['ensemble_elastic_price'].mean():,.2f}")
    print(f"     - Median: ${df['ensemble_elastic_price'].median():,.2f}")
    print(f"     - Range: ${df['ensemble_elastic_price'].min():,.2f} - ${df['ensemble_elastic_price'].max():,.2f}")
    
    # Price elasticity calculation
    df['price_elasticity_ratio'] = df['ensemble_elastic_price'] / df['ensemble_accurate_price']
    print(f"   • Price Elasticity Ratio (Elastic/Accurate):")
    print(f"     - Mean: {df['price_elasticity_ratio'].mean():.2f}x")
    print(f"     - Range: {df['price_elasticity_ratio'].min():.2f}x - {df['price_elasticity_ratio'].max():.2f}x")
    
    # 4. BUSINESS SEGMENT ANALYSIS
    print(f"\n🏢 CUSTOMER SEGMENT ANALYSIS")
    segment_analysis = df.groupby('Customer_Segment').agg({
        'ensemble_win_probability': ['mean', 'count'],
        'ensemble_accurate_price': 'mean',
        'ensemble_elastic_price': 'mean',
        'actual_status': lambda x: (x == 'Won').mean()
    }).round(3)
    
    segment_analysis.columns = ['Avg_Win_Prob', 'Quote_Count', 'Avg_Accurate_Price', 'Avg_Elastic_Price', 'Actual_Win_Rate']
    
    for segment in segment_analysis.index:
        data = segment_analysis.loc[segment]
        print(f"   • {segment}:")
        print(f"     - Quotes: {data['Quote_Count']:,} ({data['Quote_Count']/len(df)*100:.1f}%)")
        print(f"     - Predicted Win Rate: {data['Avg_Win_Prob']:.1%}")
        print(f"     - Actual Win Rate: {data['Actual_Win_Rate']:.1%}")
        print(f"     - Avg Accurate Price: ${data['Avg_Accurate_Price']:,.2f}")
        print(f"     - Avg Elastic Price: ${data['Avg_Elastic_Price']:,.2f}")
    
    # 5. PRODUCT CATEGORY ANALYSIS
    print(f"\n📦 PRODUCT CATEGORY ANALYSIS")
    category_analysis = df.groupby('Product_Category').agg({
        'ensemble_win_probability': ['mean', 'count'],
        'ensemble_accurate_price': 'mean',
        'actual_status': lambda x: (x == 'Won').mean()
    }).round(3)
    
    category_analysis.columns = ['Avg_Win_Prob', 'Quote_Count', 'Avg_Accurate_Price', 'Actual_Win_Rate']
    
    for category in category_analysis.index:
        data = category_analysis.loc[category]
        print(f"   • {category}:")
        print(f"     - Quotes: {data['Quote_Count']:,} ({data['Quote_Count']/len(df)*100:.1f}%)")
        print(f"     - Predicted Win Rate: {data['Avg_Win_Prob']:.1%}")
        print(f"     - Actual Win Rate: {data['Actual_Win_Rate']:.1%}")
        print(f"     - Avg Price: ${data['Avg_Accurate_Price']:,.2f}")
    
    # 6. PREDICTION ACCURACY ANALYSIS
    print(f"\n🎯 PREDICTION ACCURACY ANALYSIS")
    
    # Convert predictions to binary for accuracy calculation
    df['ensemble_predicted_win'] = (df['ensemble_win_probability'] > 0.5).astype(int)
    df['gnn_predicted_win'] = (df['graph_neural_network_win_probability'] > 0.5).astype(int)
    df['actual_win'] = (df['actual_status'] == 'Won').astype(int)
    
    # Accuracy metrics
    ensemble_accuracy = (df['ensemble_predicted_win'] == df['actual_win']).mean()
    gnn_accuracy = (df['gnn_predicted_win'] == df['actual_win']).mean()
    
    print(f"   • Ensemble Model Accuracy: {ensemble_accuracy:.1%}")
    print(f"   • Graph Neural Network Accuracy: {gnn_accuracy:.1%}")
    
    # Since all predictions are "Lost" due to low win probabilities, let's analyze this
    ensemble_lost_predictions = (df['ensemble_prediction'] == 'Lost').sum()
    gnn_lost_predictions = (df['graph_neural_network_prediction'] == 'Lost').sum()
    actual_lost = (df['actual_status'] == 'Lost').sum()
    
    print(f"   • Ensemble 'Lost' Predictions: {ensemble_lost_predictions:,} ({ensemble_lost_predictions/len(df)*100:.1f}%)")
    print(f"   • GNN 'Lost' Predictions: {gnn_lost_predictions:,} ({gnn_lost_predictions/len(df)*100:.1f}%)")
    print(f"   • Actual 'Lost' Outcomes: {actual_lost:,} ({actual_lost/len(df)*100:.1f}%)")
    
    # 7. PRICE vs ORIGINAL COMPARISON
    print(f"\n💵 PRICE COMPARISON WITH ORIGINALS")
    
    # Calculate price adjustments
    df['accurate_vs_net_ratio'] = df['ensemble_accurate_price'] / df['Net_Price']
    df['elastic_vs_list_ratio'] = df['ensemble_elastic_price'] / df['List_Price']
    
    print(f"   • Accurate Price vs Net Price:")
    print(f"     - Average Ratio: {df['accurate_vs_net_ratio'].mean():.2f}x")
    print(f"     - Range: {df['accurate_vs_net_ratio'].min():.2f}x - {df['accurate_vs_net_ratio'].max():.2f}x")
    
    print(f"   • Elastic Price vs List Price:")
    print(f"     - Average Ratio: {df['elastic_vs_list_ratio'].mean():.2f}x")
    print(f"     - Range: {df['elastic_vs_list_ratio'].min():.2f}x - {df['elastic_vs_list_ratio'].max():.2f}x")
    
    # 8. HIGH VALUE OPPORTUNITIES
    print(f"\n🎯 HIGH VALUE OPPORTUNITIES")
    
    # Identify quotes with high elastic prices but low win probability (potential for improvement)
    high_value_low_prob = df[(df['ensemble_elastic_price'] > df['ensemble_elastic_price'].quantile(0.75)) & 
                            (df['ensemble_win_probability'] < 0.4)]
    
    print(f"   • High Value, Low Win Probability Quotes: {len(high_value_low_prob):,}")
    print(f"   • Average Elastic Price: ${high_value_low_prob['ensemble_elastic_price'].mean():,.2f}")
    print(f"   • Average Win Probability: {high_value_low_prob['ensemble_win_probability'].mean():.1%}")
    
    # 9. MODEL RELIABILITY INSIGHTS
    print(f"\n🔧 MODEL RELIABILITY INSIGHTS")
    
    # Check if models show consistent predictions
    df['model_agreement'] = abs(df['ensemble_win_probability'] - df['graph_neural_network_win_probability'])
    
    print(f"   • Average Model Agreement (lower = more agreement): {df['model_agreement'].mean():.3f}")
    print(f"   • High Agreement Cases (<0.02 difference): {(df['model_agreement'] < 0.02).sum():,}")
    print(f"   • Low Agreement Cases (>0.05 difference): {(df['model_agreement'] > 0.05).sum():,}")
    
    # 10. BUSINESS RECOMMENDATIONS
    print(f"\n💡 BUSINESS RECOMMENDATIONS")
    
    # Low win probability analysis
    very_low_win_prob = df[df['ensemble_win_probability'] < 0.35]
    print(f"   • Quotes with Very Low Win Probability (<35%): {len(very_low_win_prob):,}")
    print(f"     - Consider price adjustments or different strategies")
    
    # High discount analysis  
    high_discount = df[df['current_discount_percent'] > 0.3]
    print(f"   • High Discount Quotes (>30%): {len(high_discount):,}")
    print(f"     - Average Win Probability: {high_discount['ensemble_win_probability'].mean():.1%}")
    
    print(f"\n" + "=" * 80)
    print("📋 ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    df = analyze_inference_results()

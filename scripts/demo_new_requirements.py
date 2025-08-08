"""
Demonstration Script for Newly Implemented Requirements
Shows comprehensive usage of:
- REQUIREMENT 9: Graph Neural Networks
- REQUIREMENT 7: Scenario Analysis & Simulation
- REQUIREMENT 4: Advanced Model Explainability
- Enhanced Data Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime, timedelta
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import the new modules
from models.graph_neural_networks import GraphNeuralNetworks
from analysis.scenario_analysis import ScenarioAnalysisSystem, ScenarioConfig
from explainability.model_explainer import AdvancedModelExplainer
from validation.data_validator import EnhancedDataValidator
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample data for demonstration"""
    
    np.random.seed(42)
    
    # Generate base data
    data = {
        'Quote_ID': [f'Q{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Customer_ID': [f'C{str(np.random.randint(1, 201)).zfill(4)}' for _ in range(n_samples)],
        'Product_ID': [f'P{str(np.random.randint(1, 51)).zfill(3)}' for _ in range(n_samples)],
        'Quote_Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D')[:n_samples],
        'List_Price': np.random.uniform(1000, 50000, n_samples),
        'Customer_Segment': np.random.choice(['SMB', 'Mid-Market', 'Enterprise', 'Strategic'], n_samples, 
                                           p=[0.4, 0.3, 0.2, 0.1]),
        'Product_Category': np.random.choice(['Hardware', 'Software', 'Services', 'Support'], n_samples),
        'Competition_Status': np.random.choice(['None', 'Low', 'Medium', 'High'], n_samples,
                                             p=[0.2, 0.3, 0.3, 0.2])
    }
    
    # Calculate derived fields
    discount_rates = np.random.uniform(0.0, 0.4, n_samples)  # 0-40% discount
    data['Net_Price'] = data['List_Price'] * (1 - discount_rates)
    
    # Create win probability based on price and other factors
    price_factor = (data['List_Price'] - data['Net_Price']) / data['List_Price']  # Discount depth
    segment_factor = {'SMB': 0.4, 'Mid-Market': 0.5, 'Enterprise': 0.6, 'Strategic': 0.7}
    competition_factor = {'None': 0.8, 'Low': 0.6, 'Medium': 0.4, 'High': 0.3}
    
    win_probs = []
    for i in range(n_samples):
        prob = 0.3  # Base probability
        prob += price_factor[i] * 0.5  # Higher discount increases win probability
        prob += segment_factor[data['Customer_Segment'][i]] * 0.3
        prob += competition_factor[data['Competition_Status'][i]] * 0.2
        prob += np.random.normal(0, 0.1)  # Random noise
        prob = max(0.05, min(0.95, prob))  # Clamp between 5% and 95%
        win_probs.append(prob)
    
    # Generate win/loss based on probabilities
    data['Status'] = ['Won' if np.random.random() < p else 'Lost' for p in win_probs]
    
    # Add some engineered features for demonstration
    data['discount_depth'] = (data['List_Price'] - data['Net_Price']) / data['List_Price']
    data['price_per_category'] = data['Net_Price']  # Simplified
    
    df = pd.DataFrame(data)
    
    return df


def demonstrate_graph_neural_networks():
    """Demonstrate Graph Neural Networks functionality"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: REQUIREMENT 9 - Graph Neural Networks")
    print("="*60)
    
    # Initialize GNN system
    gnn = GraphNeuralNetworks()
    
    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(500)  # Smaller dataset for faster processing
    print(f"   Generated {len(df)} sample records")
    
    # Create bipartite customer-product graphs
    print("\n2. Creating bipartite customer-product graphs...")
    graph_data = gnn.create_bipartite_customer_product_graphs(df)
    print(f"   Created graph with {graph_data['stats']['num_customers']} customers, "
          f"{graph_data['stats']['num_products']} products, "
          f"and {graph_data['stats']['num_edges']} edges")
    
    # Generate graph embeddings
    print("\n3. Generating graph embeddings...")
    embeddings = gnn.generate_graph_embeddings(df)
    print(f"   Generated embeddings using methods: {list(embeddings.keys())}")
    
    # Implement GraphSAGE
    print("\n4. Implementing GraphSAGE...")
    graphsage_results = gnn.implement_graphsage(df)
    print(f"   GraphSAGE model type: {graphsage_results['model_type']}")
    print(f"   Training accuracy: {graphsage_results['training_accuracy']:.3f}")
    print(f"   Validation accuracy: {graphsage_results['validation_accuracy']:.3f}")
    
    # Implement Graph Attention Networks
    print("\n5. Implementing Graph Attention Networks...")
    gat_results = gnn.implement_graph_attention_networks(df)
    print(f"   GAT model type: {gat_results['model_type']}")
    print(f"   Training accuracy: {gat_results['training_accuracy']:.3f}")
    print(f"   Validation accuracy: {gat_results['validation_accuracy']:.3f}")
    print(f"   Number of attention heads: {gat_results['num_attention_heads']}")
    
    # Model network spillover effects
    print("\n6. Modeling network spillover effects...")
    spillover_analysis = gnn.model_network_spillover_effects(df)
    print(f"   Analyzed spillover effects: {list(spillover_analysis.keys())}")
    
    # Create graph features for traditional ML
    print("\n7. Creating graph features for ML models...")
    df_with_graph_features = gnn.create_graph_features_for_ml(df)
    graph_feature_cols = [col for col in df_with_graph_features.columns 
                         if col not in df.columns]
    print(f"   Added {len(graph_feature_cols)} graph-based features")
    print(f"   Sample graph features: {graph_feature_cols[:5]}")
    
    print("\n✅ Graph Neural Networks demonstration completed successfully!")
    return gnn, df_with_graph_features


def demonstrate_scenario_analysis():
    """Demonstrate Scenario Analysis & Simulation functionality"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: REQUIREMENT 7 - Scenario Analysis & Simulation")
    print("="*60)
    
    # Initialize scenario analysis system
    scenario_system = ScenarioAnalysisSystem()
    
    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(800)
    scenario_system.set_baseline_data(df)
    print(f"   Set baseline data with {len(df)} records")
    
    # Create mock trained models (for demonstration)
    print("\n2. Loading mock trained models...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Prepare simple model for demo
    feature_cols = ['Net_Price', 'discount_depth', 'price_per_category']
    X = df[feature_cols].fillna(0)
    y = (df['Status'] == 'Won').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    mock_model = RandomForestClassifier(n_estimators=50, random_state=42)
    mock_model.fit(X_train, y_train)
    
    # Add mock model to scenario system
    scenario_system.models['ensemble'] = mock_model
    print("   Loaded mock ensemble model")
    
    # Simulate price change impacts
    print("\n3. Simulating price change impacts...")
    price_changes = {'P001': 0.1, 'P002': -0.05, 'P003': 0.15}  # 10%, -5%, 15% changes
    simulation_results = scenario_system.simulate_price_change_impacts(price_changes)
    
    print(f"   Baseline win rate: {simulation_results['baseline_metrics']['win_rate']:.3f}")
    print(f"   Scenario win rate: {simulation_results['scenario_metrics']['win_rate']:.3f}")
    print(f"   Impact analysis metrics: {list(simulation_results['impact_analysis'].keys())}")
    
    # Optimize pricing strategies
    print("\n4. Optimizing pricing strategies...")
    products_to_optimize = ['P001', 'P002', 'P003']
    optimization_results = scenario_system.optimize_pricing_strategies(
        products_to_optimize, objective='revenue'
    )
    
    if optimization_results['optimization_successful']:
        print("   Optimization successful!")
        print(f"   Optimal price changes: {optimization_results['optimal_price_changes']}")
        print(f"   Objective value: {optimization_results['objective_value']:.2f}")
    else:
        print(f"   Optimization failed: {optimization_results.get('error', 'Unknown error')}")
    
    # Model competitive responses
    print("\n5. Modeling competitive responses...")
    competitor_scenarios = [
        {'name': 'aggressive_competitor', 'price_response_factor': 0.8, 'intensity_factor': 1.2},
        {'name': 'passive_competitor', 'price_response_factor': 0.3, 'intensity_factor': 0.9}
    ]
    
    competitive_analysis = scenario_system.model_competitive_responses(
        price_changes, competitor_scenarios
    )
    
    print(f"   Analyzed {len(competitor_scenarios)} competitive scenarios")
    print(f"   Competitive risk assessment: Revenue at risk = "
          f"{competitive_analysis['competitive_risk_assessment']['revenue_at_risk']:.1f}%")
    
    # Run sensitivity analysis
    print("\n6. Running sensitivity analysis...")
    key_parameters = ['price_change_P001', 'price_change_P002']
    parameter_ranges = {
        'price_change_P001': (-0.2, 0.2),
        'price_change_P002': (-0.15, 0.15)
    }
    base_scenario = {'P001': 0.0, 'P002': 0.0}
    
    sensitivity_results = scenario_system.run_sensitivity_analysis(
        key_parameters, parameter_ranges, base_scenario
    )
    
    print(f"   Sensitivity analysis completed for {len(key_parameters)} parameters")
    if sensitivity_results['most_sensitive_parameters']:
        most_sensitive = sensitivity_results['most_sensitive_parameters'][0]
        print(f"   Most sensitive parameter: {most_sensitive[0]} "
              f"(sensitivity: {most_sensitive[1]['total_sensitivity']:.3f})")
    
    # Create interactive dashboards
    print("\n7. Creating interactive dashboards...")
    dashboards = scenario_system.create_interactive_dashboards(simulation_results)
    print(f"   Created {len(dashboards)} interactive dashboards: {list(dashboards.keys())}")
    
    # Run Monte Carlo simulation
    print("\n8. Running Monte Carlo simulation...")
    scenario_config = ScenarioConfig(
        scenario_name="price_increase_scenario",
        price_changes={'P001': 0.1, 'P002': 0.05},
        market_conditions={'economic_factor': 'stable'},
        time_horizon=90,
        monte_carlo_samples=100  # Reduced for demo
    )
    
    uncertainty_distributions = {
        'price_uncertainty_P001': {'type': 'normal', 'mean': 0, 'std': 0.02},
        'price_uncertainty_P002': {'type': 'normal', 'mean': 0, 'std': 0.015}
    }
    
    try:
        mc_results = scenario_system.run_monte_carlo_simulation(
            scenario_config, uncertainty_distributions
        )
        print(f"   Monte Carlo simulation completed: {scenario_config.scenario_name}")
        print(f"   Overall score change: {mc_results.metric_changes.get('win_rate', 0):.2f}%")
        print(f"   Risk metrics: VaR(5%) = {mc_results.risk_metrics['value_at_risk_5pct']:.2f}")
    except Exception as e:
        print(f"   Monte Carlo simulation failed: {str(e)}")
    
    print("\n✅ Scenario Analysis & Simulation demonstration completed successfully!")
    return scenario_system, simulation_results


def demonstrate_model_explainability():
    """Demonstrate Advanced Model Explainability functionality"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: REQUIREMENT 4 - Advanced Model Explainability")
    print("="*60)
    
    # Initialize explainer
    explainer = AdvancedModelExplainer()
    
    # Generate sample data and train a simple model
    print("\n1. Preparing data and models...")
    df = generate_sample_data(600)
    
    # Prepare features
    feature_cols = ['Net_Price', 'discount_depth', 'price_per_category']
    X = df[feature_cols].fillna(0)
    y = (df['Status'] == 'Won').astype(int)
    
    # Train multiple models for comparison
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Load models and data into explainer
    explainer.load_models_and_data(models, df, feature_cols)
    print(f"   Loaded {len(models)} models with {len(feature_cols)} features")
    
    # Calculate SHAP values
    print("\n2. Calculating SHAP values...")
    shap_results = explainer.calculate_shap_values('random_forest', sample_size=200)
    print(f"   SHAP analysis completed using {shap_results['explainer_type']}")
    print(f"   Feature importance (top 3): "
          f"{list(shap_results['feature_importance'].items())[:3]}")
    
    # Generate elasticity curves
    print("\n3. Generating price elasticity curves...")
    price_features = ['Net_Price', 'discount_depth']
    elasticity_results = explainer.generate_elasticity_curves(
        price_features, 'random_forest', price_range=(-0.2, 0.2), num_points=11
    )
    
    print(f"   Generated elasticity curves for {len(price_features)} features")
    for feature in price_features:
        if feature in elasticity_results['elasticity_curves']:
            curve_data = elasticity_results['elasticity_curves'][feature]
            baseline = curve_data['baseline_prediction']
            print(f"   {feature}: baseline prediction = {baseline:.3f}")
    
    # Create automated reports
    print("\n4. Creating automated reports...")
    automated_report = explainer.create_automated_reports('random_forest')
    
    print(f"   Executive Summary: {automated_report['executive_summary'][:100]}...")
    print(f"   Model Performance: Accuracy = "
          f"{automated_report['model_performance']['accuracy']:.3f}")
    print(f"   Top drivers: {len(automated_report['feature_importance_analysis']['top_drivers'])}")
    print(f"   Business recommendations: "
          f"{len(automated_report['business_recommendations'])}")
    
    # Perform sensitivity analysis
    print("\n5. Performing sensitivity analysis...")
    key_features = feature_cols[:2]  # Analyze top 2 features
    sensitivity_results = explainer.perform_sensitivity_analysis(
        key_features, 'random_forest', perturbation_range=0.1
    )
    
    print(f"   Sensitivity analysis completed for {len(key_features)} features")
    if sensitivity_results['sensitivity_ranking']:
        most_sensitive = sensitivity_results['sensitivity_ranking'][0]
        print(f"   Most sensitive feature: {most_sensitive[0]} "
              f"(sensitivity: {abs(most_sensitive[1]):.3f})")
    
    # Explain individual predictions
    print("\n6. Explaining individual predictions...")
    sample_indices = [0, 50, 100]  # Explain 3 predictions
    individual_explanations = explainer.explain_individual_predictions(
        sample_indices, 'random_forest'
    )
    
    print(f"   Explained {individual_explanations['explained_samples']} predictions")
    for idx in sample_indices:
        if idx in individual_explanations['individual_explanations']:
            explanation = individual_explanations['individual_explanations'][idx]
            print(f"   Sample {idx}: prediction = {explanation['prediction']:.3f}")
    
    print("\n✅ Advanced Model Explainability demonstration completed successfully!")
    return explainer, automated_report


def demonstrate_data_validation():
    """Demonstrate Enhanced Data Validation functionality"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Enhanced Data Validation")
    print("="*60)
    
    # Initialize validator
    validator = EnhancedDataValidator()
    
    # Generate sample data with some quality issues
    print("\n1. Generating sample data with quality issues...")
    df = generate_sample_data(400)
    
    # Introduce some data quality issues for demonstration
    df_with_issues = df.copy()
    
    # Add missing values
    missing_indices = np.random.choice(df.index, size=20, replace=False)
    df_with_issues.loc[missing_indices, 'Customer_Segment'] = np.nan
    
    # Add invalid status values
    df_with_issues.loc[df_with_issues.index[5:8], 'Status'] = 'Pending'
    
    # Add negative prices
    df_with_issues.loc[df_with_issues.index[10:12], 'Net_Price'] = -1000
    
    # Add Net Price > List Price violations
    df_with_issues.loc[df_with_issues.index[15:18], 'Net_Price'] = \
        df_with_issues.loc[df_with_issues.index[15:18], 'List_Price'] * 1.2
    
    print(f"   Generated dataset with {len(df_with_issues)} records and introduced quality issues")
    
    # Validate dataset schema
    print("\n2. Validating dataset schema...")
    schema_results = validator.validate_dataset_schema(df_with_issues, 'quote_history')
    
    schema_errors = [r for r in schema_results if r.level.value == 'error']
    schema_warnings = [r for r in schema_results if r.level.value == 'warning']
    
    print(f"   Schema validation: {len(schema_errors)} errors, {len(schema_warnings)} warnings")
    if schema_errors:
        print(f"   Sample error: {schema_errors[0].message}")
    
    # Perform comprehensive data quality assessment
    print("\n3. Performing data quality assessment...")
    quality_metrics = validator.perform_data_quality_assessment(df_with_issues, 'demo_dataset')
    
    print(f"   Overall quality score: {quality_metrics.overall_score:.3f}")
    print(f"   Completeness score: {quality_metrics.completeness_score:.3f}")
    print(f"   Consistency score: {quality_metrics.consistency_score:.3f}")
    print(f"   Validity score: {quality_metrics.validity_score:.3f}")
    print(f"   Uniqueness score: {quality_metrics.uniqueness_score:.3f}")
    print(f"   Quality issues found: {len(quality_metrics.quality_issues)}")
    
    # Validate business rules
    print("\n4. Validating business rules...")
    business_rule_results = validator.validate_business_rules(df_with_issues)
    
    business_errors = [r for r in business_rule_results if r.level.value == 'error']
    print(f"   Business rule validation: {len(business_errors)} errors")
    
    for error in business_errors[:3]:  # Show first 3 errors
        print(f"   - {error.message}")
    
    # Detect outliers
    print("\n5. Detecting outliers...")
    outlier_results = validator.detect_outliers(df_with_issues)
    
    print(f"   Outlier detection: {len(outlier_results)} columns analyzed")
    outlier_issues = [r for r in outlier_results if not r.passed]
    if outlier_issues:
        print(f"   Found outliers in {len(outlier_issues)} columns")
        print(f"   Sample: {outlier_issues[0].message}")
    
    # Check data drift (compare with original clean data)
    print("\n6. Checking data drift...")
    drift_results = validator.check_data_drift(df_with_issues, df, threshold=0.1)
    
    drift_detected = [r for r in drift_results if not r.passed]
    print(f"   Data drift analysis: {len(drift_results)} checks performed")
    if drift_detected:
        print(f"   Drift detected in {len(drift_detected)} aspects")
    else:
        print("   No significant drift detected")
    
    # Generate comprehensive validation report
    print("\n7. Generating validation report...")
    all_validation_results = (schema_results + business_rule_results + 
                             outlier_results + drift_results)
    
    validation_report = validator.generate_validation_report(
        all_validation_results, quality_metrics
    )
    
    print(f"   Validation report generated")
    print(f"   Total checks: {validation_report['summary']['total_checks']}")
    print(f"   Pass rate: {validation_report['summary']['pass_rate_percentage']:.1f}%")
    print(f"   Recommendations: {len(validation_report['recommendations'])}")
    
    # Show top recommendations
    print("\n   Top recommendations:")
    for i, rec in enumerate(validation_report['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ Enhanced Data Validation demonstration completed successfully!")
    return validator, validation_report


def main():
    """Main demonstration function"""
    
    print("🎯 COMPREHENSIVE DEMONSTRATION OF NEW REQUIREMENTS")
    print("=" * 80)
    print("This script demonstrates the implementation of:")
    print("- REQUIREMENT 9: Graph Neural Networks")
    print("- REQUIREMENT 7: Scenario Analysis & Simulation")
    print("- REQUIREMENT 4: Advanced Model Explainability")
    print("- Enhanced Data Validation with Schema Validation")
    print("=" * 80)
    
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path(__file__).parent.parent / 'outputs' / 'demonstrations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Demonstrate Graph Neural Networks
        gnn, df_with_graph = demonstrate_graph_neural_networks()
        results['graph_neural_networks'] = {
            'status': 'completed',
            'features_added': len([c for c in df_with_graph.columns if 'embedding' in c or 'centrality' in c])
        }
        
        # 2. Demonstrate Scenario Analysis & Simulation
        scenario_system, simulation_results = demonstrate_scenario_analysis()
        results['scenario_analysis'] = {
            'status': 'completed',
            'baseline_win_rate': simulation_results['baseline_metrics']['win_rate'],
            'scenario_win_rate': simulation_results['scenario_metrics']['win_rate']
        }
        
        # 3. Demonstrate Advanced Model Explainability
        explainer, report = demonstrate_model_explainability()
        results['model_explainability'] = {
            'status': 'completed',
            'model_accuracy': report['model_performance']['accuracy'],
            'top_driver': report['feature_importance_analysis']['top_drivers'][0][0] if report['feature_importance_analysis']['top_drivers'] else 'N/A'
        }
        
        # 4. Demonstrate Enhanced Data Validation
        validator, validation_report = demonstrate_data_validation()
        results['data_validation'] = {
            'status': 'completed',
            'overall_quality_score': validation_report['quality_metrics']['overall_score'],
            'pass_rate': validation_report['summary']['pass_rate_percentage']
        }
        
        # Save results summary
        results_file = output_dir / f'demonstration_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📊 SUMMARY OF RESULTS:")
        print("-" * 40)
        
        for requirement, result in results.items():
            print(f"✅ {requirement.replace('_', ' ').title()}: {result['status']}")
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        
        print("\n🔍 KEY FINDINGS:")
        print(f"- Graph features added: {results['graph_neural_networks']['features_added']}")
        print(f"- Scenario analysis impact: Win rate changed from "
              f"{results['scenario_analysis']['baseline_win_rate']:.3f} to "
              f"{results['scenario_analysis']['scenario_win_rate']:.3f}")
        print(f"- Model explainability: Top driver is '{results['model_explainability']['top_driver']}'")
        print(f"- Data validation: Overall quality score = "
              f"{results['data_validation']['overall_quality_score']:.3f}")
        
        print("\n✨ The Price Elasticity system now includes all requested enhancements!")
        print("All four requirements have been successfully implemented and demonstrated.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\n❌ Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

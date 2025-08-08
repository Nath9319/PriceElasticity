"""
Complete Price Elasticity Modeling Pipeline
This script runs the entire modeling pipeline from EDA to final models.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# Add src to Python path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "src"))

from utils.config_loader import config_loader, logger, create_directory_structure
from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
from training.model_training import PriceElasticityModelTraining

warnings.filterwarnings('ignore')


class PriceElasticityPipeline:
    """
    Complete pipeline for B2B Price Elasticity Modeling
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.config = config_loader
        self.logger = logger
        self.data = {}
        self.processed_data = None
        self.models = {}
        self.results = {}
        
        # Initialize components
        self.feature_engineer = PriceElasticityFeatureEngineering()
        self.model_trainer = PriceElasticityModelTraining()
        
        self.logger.info("Price Elasticity Pipeline initialized")
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        self.logger.info("Creating sample datasets...")
        
        np.random.seed(42)
        
        # Create datasets directory
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        
        # Quote History - Central table with realistic correlations
        n_quotes = 10000
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start_date, end_date, periods=n_quotes)
        
        # Generate correlated features
        base_list_price = np.random.normal(1000, 300, n_quotes).clip(100, 5000)
        product_categories = np.random.choice(['Hardware', 'Software', 'Services', 'Support'], n_quotes)
        competition_status = np.random.choice(['High', 'Medium', 'Low', 'None'], n_quotes)
        regions = np.random.choice(['North America', 'Europe', 'APAC', 'Latin America'], n_quotes)
        product_objectives = np.random.choice(['Growth', 'Profitability', 'Market Share', 'Defensive'], n_quotes)
        
        # Create realistic discount patterns based on business logic
        base_discount = np.random.beta(2, 8, n_quotes) * 0.6
        
        # Adjust discounts based on competition and product category
        discount_adjustments = np.zeros(n_quotes)
        discount_adjustments += np.where(competition_status == 'High', 0.15, 0)  # Higher discounts for high competition
        discount_adjustments += np.where(competition_status == 'Medium', 0.08, 0)
        discount_adjustments += np.where(product_categories == 'Software', 0.05, 0)  # Software can have higher margins
        discount_adjustments += np.where(product_objectives == 'Market Share', 0.10, 0)  # Market share objectives = higher discounts
        
        # Final discount with some randomness
        discount_percent = (base_discount + discount_adjustments + np.random.normal(0, 0.03, n_quotes)).clip(0, 0.7)
        
        # Create realistic win probability based on business factors
        # Base win rate around 35%
        win_logits = np.random.normal(-0.6, 0.8, n_quotes)  # Base logit for ~35% win rate
        
        # Discount effect: higher discounts increase win probability (price elasticity!)
        win_logits += (discount_percent - 0.12) * 4  # Centered around mean discount, positive correlation
        
        # Competition effect: more competition reduces win probability
        win_logits -= np.where(competition_status == 'High', 0.8, 0)
        win_logits -= np.where(competition_status == 'Medium', 0.3, 0)
        
        # Price effect: very high prices reduce win probability
        price_percentile = (base_list_price - base_list_price.mean()) / base_list_price.std()
        win_logits -= np.where(price_percentile > 1.5, 0.5, 0)  # Penalize very high prices
        
        # Product category effects
        win_logits += np.where(product_categories == 'Services', 0.3, 0)  # Services easier to win
        win_logits -= np.where(product_categories == 'Hardware', 0.2, 0)  # Hardware more competitive
        
        # Regional effects
        win_logits += np.where(regions == 'North America', 0.2, 0)  # Home market advantage
        
        # Convert logits to probabilities and generate outcomes
        win_probabilities = 1 / (1 + np.exp(-win_logits))
        status = np.where(np.random.random(n_quotes) < win_probabilities, 'Won', 'Lost')
        
        quote_history = pd.DataFrame({
            'Quote_ID': [f'Q{i:06d}' for i in range(1, n_quotes + 1)],
            'Customer_ID': [f'C{np.random.randint(1, 2000):04d}' for _ in range(n_quotes)],
            'Product_ID': [f'P{np.random.randint(1, 500):03d}' for _ in range(n_quotes)],
            'Quote_Date': np.random.choice(dates, n_quotes),
            'List_Price': base_list_price,
            'Discount_Percent': discount_percent,
            'Status': status,
            'Product_Category': product_categories,
            'Region': regions,
            'Competition_Status': competition_status,
            'Product_Objective': product_objectives
        })
        
        # Calculate derived fields
        quote_history['Net_Price'] = quote_history['List_Price'] * (1 - quote_history['Discount_Percent'])
        quote_history['Offered_Price'] = quote_history['Net_Price'] * np.random.normal(1.0, 0.05, n_quotes).clip(0.8, 1.2)
        quote_history['Offered_Discount'] = (quote_history['List_Price'] - quote_history['Offered_Price']) / quote_history['List_Price']
        quote_history['Quote_Date'] = pd.to_datetime(quote_history['Quote_Date'])
        
        quote_history.to_csv(datasets_dir / 'quote_history.csv', index=False)
        
        # Sales History
        n_sales = 8000
        sales_history = pd.DataFrame({
            'Sale_ID': [f'S{i:06d}' for i in range(1, n_sales + 1)],
            'Customer_ID': [f'C{np.random.randint(1, 2000):04d}' for _ in range(n_sales)],
            'Product_ID': [f'P{np.random.randint(1, 500):03d}' for _ in range(n_sales)],
            'Sale_Date': pd.date_range('2022-01-01', '2024-12-31', periods=n_sales),
            'Quantity': np.random.poisson(5, n_sales) + 1,
            'Unit_Price': np.random.normal(950, 250, n_sales).clip(100, 4500)
        })
        
        sales_history['Total_Revenue'] = sales_history['Quantity'] * sales_history['Unit_Price']
        sales_history['COGS'] = sales_history['Unit_Price'] * 0.6
        sales_history.to_csv(datasets_dir / 'sales_history.csv', index=False)
        
        # Customer Master
        customer_ids = [f'C{i:04d}' for i in range(1, 2001)]
        customer_master = pd.DataFrame({
            'Customer_ID': customer_ids,
            'Customer_Name': [f'Customer {i}' for i in range(1, 2001)],
            'Industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing'], 2000),
            'Company_Size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], 2000),
            'Customer_Since_Date': pd.date_range('2018-01-01', '2023-12-31', periods=2000),
            'Credit_Rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB'], 2000),
            'Annual_Revenue': np.random.lognormal(15, 1.5, 2000).clip(100000, 1e9)
        })
        customer_master.to_csv(datasets_dir / 'customer_master.csv', index=False)
        
        # Customer Segmentation
        customer_segmentation = pd.DataFrame({
            'Customer_ID': customer_ids,
            'Customer_Segment': np.random.choice(['SMB', 'Mid-Market', 'Enterprise', 'Strategic'], 
                                               2000, p=[0.4, 0.3, 0.2, 0.1]),
            'RFM_Score': np.random.randint(111, 556, 2000),
            'CLV_Score': np.random.normal(5000, 2000, 2000).clip(500, 20000),
            'Price_Sensitivity': np.random.choice(['High', 'Medium', 'Low'], 2000),
            'Negotiation_Style': np.random.choice(['Aggressive', 'Moderate', 'Passive'], 2000)
        })
        customer_segmentation.to_csv(datasets_dir / 'customer_segmentation.csv', index=False)
        
        # Product Master
        product_ids = [f'P{i:03d}' for i in range(1, 501)]
        product_master = pd.DataFrame({
            'Product_ID': product_ids,
            'Product_Name': [f'Product {i}' for i in range(1, 501)],
            'Product_Category': np.random.choice(['Hardware', 'Software', 'Services', 'Support'], 500),
            'Product_Line': np.random.choice(['Core', 'Premium', 'Basic', 'Enterprise'], 500),
            'Launch_Date': pd.date_range('2015-01-01', '2024-01-01', periods=500),
            'Standard_Cost': np.random.normal(500, 150, 500).clip(50, 2000),
            'List_Price': np.random.normal(800, 200, 500).clip(100, 3000),
            'Lifecycle_Stage': np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'], 500)
        })
        product_master.to_csv(datasets_dir / 'product_master.csv', index=False)
        
        self.logger.info("Sample datasets created successfully")
    
    def load_data(self):
        """Load all datasets"""
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
    
    def run_eda(self):
        """Run Exploratory Data Analysis"""
        self.logger.info("Running EDA...")
        
        # Create unified dataset
        unified_data = self.create_unified_dataset()
        
        # Basic EDA metrics
        eda_results = {
            'dataset_overview': {
                'total_records': len(unified_data),
                'total_customers': unified_data['Customer_ID'].nunique(),
                'total_products': unified_data['Product_ID'].nunique(),
                'date_range': {
                    'start': unified_data['Quote_Date'].min(),
                    'end': unified_data['Quote_Date'].max()
                }
            },
            'data_quality': {
                'missing_values': unified_data.isnull().sum().to_dict(),
                'missing_percentages': (unified_data.isnull().sum() / len(unified_data) * 100).round(2).to_dict()
            },
            'business_metrics': {
                'overall_win_rate': (unified_data['Status'] == 'Won').mean(),
                'avg_discount': unified_data['Discount_Percent'].mean(),
                'avg_net_price': unified_data['Net_Price'].mean()
            }
        }
        
        # Win rates by segment
        if 'Customer_Segment' in unified_data.columns:
            segment_win_rates = unified_data.groupby('Customer_Segment')['Status'].apply(
                lambda x: (x == 'Won').mean()
            ).to_dict()
            eda_results['segment_win_rates'] = segment_win_rates
        
        # Product category analysis
        if 'Product_Category' in unified_data.columns:
            category_stats = unified_data.groupby('Product_Category').agg({
                'Status': lambda x: (x == 'Won').mean(),
                'Net_Price': 'mean',
                'Discount_Percent': 'mean'
            }).round(3).to_dict('index')
            eda_results['category_analysis'] = category_stats
        
        self.results['eda'] = eda_results
        
        # Save EDA results
        results_dir = Path("results/eda")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_dir / 'eda_summary.json', 'w') as f:
            json.dump(eda_results, f, indent=2, default=str)
        
        self.logger.info("EDA completed successfully")
        
        return unified_data, eda_results
    
    def run_feature_engineering(self, unified_data):
        """Run feature engineering pipeline"""
        self.logger.info("Running feature engineering...")
        
        # Create comprehensive features
        featured_data = self.feature_engineer.create_comprehensive_features(unified_data, fit=True)
        
        # Save feature engineering artifacts
        self.feature_engineer.save_feature_engineering_artifacts()
        
        self.logger.info(f"Feature engineering completed: {featured_data.shape}")
        
        return featured_data
    
    def run_model_training(self, featured_data):
        """Run model training pipeline"""
        self.logger.info("Running model training...")
        
        # Prepare training data
        X, y = self.model_trainer.prepare_training_data(featured_data, target_col='Status')
        
        # Train all models
        training_results = self.model_trainer.train_all_models(X, y)
        
        # Save models
        self.model_trainer.save_models()
        
        self.results['model_training'] = training_results
        self.logger.info("Model training completed successfully")
        
        return training_results
    
    def create_business_insights(self, training_results):
        """Create business insights and recommendations"""
        self.logger.info("Generating business insights...")
        
        insights = {
            'model_performance': {},
            'price_elasticity_insights': {},
            'business_recommendations': []
        }
        
        # Extract model performance
        if 'model_comparison' in training_results:
            comparison = training_results['model_comparison']
            insights['model_performance'] = {
                'best_model': comparison.get('best_model', 'Unknown'),
                'best_auc': comparison.get('best_auc', 0),
                'models_trained': comparison.get('models_trained', [])
            }
        
        # Extract price elasticity insights
        if 'x_learner' in training_results:
            xl_results = training_results['x_learner']
            if 'treatment_analysis' in xl_results:
                insights['price_elasticity_insights'] = {
                    'average_treatment_effect': xl_results['treatment_analysis'].get('average_treatment_effect', 0),
                    'positive_effects_ratio': xl_results['treatment_analysis'].get('positive_effects', 0),
                    'treatment_effect_range': xl_results['treatment_analysis'].get('treatment_effect_range', [0, 0])
                }
        
        # Business recommendations
        recommendations = []
        
        if insights['model_performance']['best_auc'] > 0.7:
            recommendations.append("Model performance is good (AUC > 0.7). Consider deploying for pricing optimization.")
        else:
            recommendations.append("Model performance needs improvement. Consider more feature engineering or data collection.")
        
        if abs(insights['price_elasticity_insights'].get('average_treatment_effect', 0)) > 0.05:
            recommendations.append("Significant price elasticity detected. Price changes can materially impact win rates.")
        else:
            recommendations.append("Low price elasticity detected. Consider focusing on value proposition rather than pricing.")
        
        insights['business_recommendations'] = recommendations
        
        self.results['business_insights'] = insights
        
        # Save insights
        results_dir = Path("results")
        with open(results_dir / 'business_insights.json', 'w') as f:
            import json
            json.dump(insights, f, indent=2, default=str)
        
        return insights
    
    def run_complete_pipeline(self):
        """Run the complete price elasticity modeling pipeline"""
        print("🚀 Starting Complete Price Elasticity Modeling Pipeline...")
        
        try:
            # Step 1: Setup
            create_directory_structure()
            
            # Step 2: Create sample data (if needed)
            self.create_sample_datasets()
            
            # Step 3: Load data
            self.load_data()
            
            # Step 4: Run EDA
            unified_data, eda_results = self.run_eda()
            
            # Step 5: Feature Engineering
            featured_data = self.run_feature_engineering(unified_data)
            
            # Step 6: Model Training
            training_results = self.run_model_training(featured_data)
            
            # Step 7: Business Insights
            business_insights = self.create_business_insights(training_results)
            
            # Step 8: Generate final summary
            self._generate_final_summary()
            
            print("✅ Complete pipeline executed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"❌ Pipeline failed: {e}")
            return False
    
    def _generate_final_summary(self):
        """Generate final pipeline summary"""
        summary = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'stages_completed': ['data_loading', 'eda', 'feature_engineering', 'model_training', 'insights']
            },
            'data_summary': self.results.get('eda', {}).get('dataset_overview', {}),
            'model_performance': self.results.get('business_insights', {}).get('model_performance', {}),
            'key_insights': self.results.get('business_insights', {}).get('business_recommendations', [])
        }
        
        # Save final summary
        with open('pipeline_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary to console
        print("\n📊 Pipeline Summary:")
        print(f"  - Data processed: {summary['data_summary'].get('total_records', 'N/A'):,} records")
        print(f"  - Customers: {summary['data_summary'].get('total_customers', 'N/A'):,}")
        print(f"  - Products: {summary['data_summary'].get('total_products', 'N/A'):,}")
        print(f"  - Best model: {summary['model_performance'].get('best_model', 'N/A')}")
        print(f"  - Best AUC: {summary['model_performance'].get('best_auc', 'N/A'):.3f}")
        
        print("\n💡 Key Insights:")
        for insight in summary['key_insights']:
            print(f"  - {insight}")
        
        print(f"\n📁 Results saved to: results/")
        print(f"📁 Models saved to: models/trained/")
        print(f"📄 Summary saved to: pipeline_summary.json")


def main():
    """Main function to run the complete pipeline"""
    pipeline = PriceElasticityPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Price Elasticity Modeling Pipeline completed successfully!")
        print("Check the results/ directory for detailed outputs.")
    else:
        print("\n💥 Pipeline execution failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

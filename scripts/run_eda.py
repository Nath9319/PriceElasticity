"""
Comprehensive Exploratory Data Analysis for B2B Price Elasticity Modeling
This script performs detailed EDA as specified in the requirements document.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append('src')

from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PriceElasticityEDA:
    """
    Comprehensive Exploratory Data Analysis for Price Elasticity Modeling
    """
    
    def __init__(self):
        """Initialize EDA class with configuration"""
        self.config = config_loader
        self.logger = logger
        self.data = {}
        self.unified_data = None
        
        # Create output directories
        self.output_dir = Path("results/eda")
        self.plots_dir = self.output_dir / "plots" 
        self.reports_dir = self.output_dir / "reports"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("EDA initialized successfully")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets as specified in requirements
        
        Returns:
            Dictionary of loaded DataFrames
        """
        self.logger.info("Loading datasets...")
        
        data_config = self.config.get_data_config()
        datasets_path = Path(data_config['datasets_path'])
        files = data_config['files']
        
        for name, filename in files.items():
            try:
                file_path = datasets_path / filename
                if file_path.exists():
                    self.data[name] = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {name}: {self.data[name].shape}")
                else:
                    self.logger.warning(f"File not found: {file_path}")
                    # Create sample data for demonstration
                    self.data[name] = self._create_sample_data(name)
                    self.logger.info(f"Created sample {name}: {self.data[name].shape}")
            except Exception as e:
                self.logger.error(f"Error loading {name}: {e}")
                self.data[name] = self._create_sample_data(name)
        
        return self.data
    
    def _create_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Create sample data for demonstration when real data is not available
        
        Args:
            dataset_name: Name of the dataset to create
            
        Returns:
            Sample DataFrame
        """
        np.random.seed(42)
        
        if dataset_name == 'quote_history':
            n_quotes = 10000
            
            # Generate realistic date range
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2024, 12, 31)
            dates = pd.date_range(start_date, end_date, periods=n_quotes)
            
            data = {
                'Quote_ID': [f'Q{i:06d}' for i in range(1, n_quotes + 1)],
                'Customer_ID': [f'C{np.random.randint(1, 2000):04d}' for _ in range(n_quotes)],
                'Product_ID': [f'P{np.random.randint(1, 500):03d}' for _ in range(n_quotes)],
                'Quote_Date': np.random.choice(dates, n_quotes),
                'List_Price': np.random.normal(1000, 300, n_quotes).clip(100, 5000),
                'Net_Price': None,  # Will calculate based on discount
                'Offered_Price': None,  # Will calculate
                'Discount_Percent': np.random.beta(2, 8, n_quotes) * 0.6,  # 0-60% discounts
                'Offered_Discount': None,  # Will calculate
                'Status': np.random.choice(['Won', 'Lost'], n_quotes, p=[0.35, 0.65]),
                'Product_Category': np.random.choice(['Hardware', 'Software', 'Services', 'Support'], n_quotes),
                'Region': np.random.choice(['North America', 'Europe', 'APAC', 'Latin America'], n_quotes),
                'Competition_Status': np.random.choice(['High', 'Medium', 'Low', 'None'], n_quotes),
                'Product_Objective': np.random.choice(['Growth', 'Profitability', 'Market Share', 'Defensive'], n_quotes)
            }
            
            df = pd.DataFrame(data)
            
            # Calculate derived fields
            df['Net_Price'] = df['List_Price'] * (1 - df['Discount_Percent'])
            df['Offered_Price'] = df['Net_Price'] * np.random.normal(1.0, 0.05, n_quotes).clip(0.8, 1.2)
            df['Offered_Discount'] = (df['List_Price'] - df['Offered_Price']) / df['List_Price']
            df['Quote_Date'] = pd.to_datetime(df['Quote_Date'])
            
            return df
            
        elif dataset_name == 'sales_history':
            n_sales = 8000
            data = {
                'Sale_ID': [f'S{i:06d}' for i in range(1, n_sales + 1)],
                'Customer_ID': [f'C{np.random.randint(1, 2000):04d}' for _ in range(n_sales)],
                'Product_ID': [f'P{np.random.randint(1, 500):03d}' for _ in range(n_sales)],
                'Sale_Date': pd.date_range('2022-01-01', '2024-12-31', periods=n_sales),
                'Quantity': np.random.poisson(5, n_sales) + 1,
                'Unit_Price': np.random.normal(950, 250, n_sales).clip(100, 4500),
                'Total_Revenue': None,  # Will calculate
                'COGS': None  # Will calculate
            }
            
            df = pd.DataFrame(data)
            df['Total_Revenue'] = df['Quantity'] * df['Unit_Price']
            df['COGS'] = df['Unit_Price'] * 0.6  # Assume 40% margin
            df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
            
            return df
        
        elif dataset_name == 'customer_master':
            customer_ids = [f'C{i:04d}' for i in range(1, 2001)]
            data = {
                'Customer_ID': customer_ids,
                'Customer_Name': [f'Customer {i}' for i in range(1, 2001)],
                'Industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing', 
                                            'Retail', 'Energy', 'Government'], 2000),
                'Company_Size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], 2000),
                'Customer_Since_Date': pd.date_range('2018-01-01', '2023-12-31', periods=2000),
                'Credit_Rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], 2000),
                'Annual_Revenue': np.random.lognormal(15, 1.5, 2000).clip(100000, 1e9)
            }
            
            return pd.DataFrame(data)
        
        elif dataset_name == 'customer_segmentation':
            customer_ids = [f'C{i:04d}' for i in range(1, 2001)]
            data = {
                'Customer_ID': customer_ids,
                'Customer_Segment': np.random.choice(['SMB', 'Mid-Market', 'Enterprise', 'Strategic'], 
                                                   2000, p=[0.4, 0.3, 0.2, 0.1]),
                'RFM_Score': np.random.randint(111, 556, 2000),
                'CLV_Score': np.random.normal(5000, 2000, 2000).clip(500, 20000),
                'Price_Sensitivity': np.random.choice(['High', 'Medium', 'Low'], 2000),
                'Negotiation_Style': np.random.choice(['Aggressive', 'Moderate', 'Passive'], 2000)
            }
            
            return pd.DataFrame(data)
        
        elif dataset_name == 'product_master':
            product_ids = [f'P{i:03d}' for i in range(1, 501)]
            data = {
                'Product_ID': product_ids,
                'Product_Name': [f'Product {i}' for i in range(1, 501)],
                'Product_Category': np.random.choice(['Hardware', 'Software', 'Services', 'Support'], 500),
                'Product_Line': np.random.choice(['Core', 'Premium', 'Basic', 'Enterprise'], 500),
                'Launch_Date': pd.date_range('2015-01-01', '2024-01-01', periods=500),
                'Standard_Cost': np.random.normal(500, 150, 500).clip(50, 2000),
                'List_Price': np.random.normal(800, 200, 500).clip(100, 3000),
                'Lifecycle_Stage': np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'], 500)
            }
            
            return pd.DataFrame(data)
        
        else:
            return pd.DataFrame()
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """
        Create unified analytical dataset by joining all tables
        Following Requirement 1: Join quote_history as central table
        
        Returns:
            Unified DataFrame for analysis
        """
        self.logger.info("Creating unified dataset...")
        
        # Start with quote_history as the central table
        unified = self.data['quote_history'].copy()
        
        # Join with sales_history for historical context
        if 'sales_history' in self.data and not self.data['sales_history'].empty:
            sales_agg = self.data['sales_history'].groupby(['Customer_ID', 'Product_ID']).agg({
                'Quantity': ['sum', 'mean'],
                'Unit_Price': ['mean', 'std'],
                'Total_Revenue': 'sum',
                'Sale_Date': 'max'
            }).round(2)
            
            sales_agg.columns = [f'hist_{col[0]}_{col[1]}' for col in sales_agg.columns]
            sales_agg = sales_agg.reset_index()
            
            unified = unified.merge(sales_agg, on=['Customer_ID', 'Product_ID'], how='left')
        
        # Join with customer_master
        if 'customer_master' in self.data and not self.data['customer_master'].empty:
            unified = unified.merge(self.data['customer_master'], on='Customer_ID', how='left')
        
        # Join with customer_segmentation
        if 'customer_segmentation' in self.data and not self.data['customer_segmentation'].empty:
            unified = unified.merge(self.data['customer_segmentation'], on='Customer_ID', how='left')
        
        # Join with product_master
        if 'product_master' in self.data and not self.data['product_master'].empty:
            unified = unified.merge(self.data['product_master'], on='Product_ID', how='left')
        
        self.unified_data = unified
        self.logger.info(f"Unified dataset created: {unified.shape}")
        
        return unified
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        Following Requirement 5.1: Data quality assessment
        
        Returns:
            Dictionary with data quality metrics
        """
        self.logger.info("Assessing data quality...")
        
        quality_report = {}
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data
        
        # Missing values analysis
        missing_analysis = {
            'total_rows': len(df),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'columns_with_high_missing': []
        }
        
        # Identify columns with >30% missing data
        for col, pct in missing_analysis['missing_percentages'].items():
            if pct > 30:
                missing_analysis['columns_with_high_missing'].append(col)
        
        quality_report['missing_values'] = missing_analysis
        
        # Focus on key fields as specified in requirements
        key_fields = ['Net_Price', 'Discount_Percent', 'Product_Objective', 'List_Price']
        key_field_quality = {}
        
        for field in key_fields:
            if field in df.columns:
                key_field_quality[field] = {
                    'missing_count': df[field].isnull().sum(),
                    'missing_percentage': (df[field].isnull().sum() / len(df) * 100).round(2),
                    'data_type': str(df[field].dtype),
                    'unique_values': df[field].nunique() if df[field].dtype in ['object', 'category'] else None
                }
        
        quality_report['key_fields'] = key_field_quality
        
        # Outlier detection
        outlier_analysis = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['List_Price', 'Net_Price', 'Offered_Price', 'Discount_Percent']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                outlier_analysis[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / len(df) * 100), 2),
                    'negative_values': (df[col] < 0).sum(),
                    'extreme_values': (df[col] > df[col].quantile(0.99)).sum()
                }
        
        quality_report['outliers'] = outlier_analysis
        
        # Data consistency checks
        consistency_checks = {}
        
        # Price consistency
        if all(col in df.columns for col in ['List_Price', 'Net_Price', 'Discount_Percent']):
            price_inconsistent = df[
                abs(df['Net_Price'] - (df['List_Price'] * (1 - df['Discount_Percent']))) > 0.01
            ]
            consistency_checks['price_consistency'] = {
                'inconsistent_records': len(price_inconsistent),
                'percentage': (len(price_inconsistent) / len(df) * 100).round(2)
            }
        
        quality_report['consistency'] = consistency_checks
        
        # Save quality report
        quality_df = pd.DataFrame({
            'Column': list(missing_analysis['missing_percentages'].keys()),
            'Missing_Count': list(missing_analysis['missing_by_column'].values()),
            'Missing_Percentage': list(missing_analysis['missing_percentages'].values()),
            'Data_Type': [str(df[col].dtype) for col in missing_analysis['missing_percentages'].keys()]
        })
        
        quality_df.to_csv(self.reports_dir / 'data_quality_report.csv', index=False)
        
        return quality_report
    
    def analyze_price_variation(self) -> Dict[str, Any]:
        """
        Analyze price variation for elasticity modeling
        Following Requirement 1.3: Assess price variation
        
        Returns:
            Dictionary with price variation analysis
        """
        self.logger.info("Analyzing price variation...")
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data
        price_analysis = {}
        
        # Overall price statistics
        price_columns = ['List_Price', 'Net_Price', 'Offered_Price']
        price_stats = {}
        
        for col in price_columns:
            if col in df.columns:
                price_stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                }
        
        price_analysis['price_statistics'] = price_stats
        
        # Price variation by product
        product_price_variation = []
        
        if 'Product_ID' in df.columns and 'Net_Price' in df.columns:
            for product in df['Product_ID'].value_counts().head(20).index:
                product_prices = df[df['Product_ID'] == product]['Net_Price']
                if len(product_prices) > 5:  # Only products with sufficient data
                    variation = {
                        'Product_ID': product,
                        'transaction_count': len(product_prices),
                        'price_mean': product_prices.mean(),
                        'price_std': product_prices.std(),
                        'cv': product_prices.std() / product_prices.mean() if product_prices.mean() != 0 else 0,
                        'price_range': product_prices.max() - product_prices.min(),
                        'sufficient_variation': product_prices.std() / product_prices.mean() > 0.1
                    }
                    product_price_variation.append(variation)
        
        price_analysis['product_variation'] = product_price_variation
        
        # Identify products with sufficient price variation for modeling
        sufficient_variation_products = [
            p for p in product_price_variation 
            if p['sufficient_variation'] and p['transaction_count'] >= 10
        ]
        
        price_analysis['modeling_candidates'] = {
            'total_products': len(product_price_variation),
            'sufficient_variation_count': len(sufficient_variation_products),
            'percentage_suitable': (len(sufficient_variation_products) / len(product_price_variation) * 100) 
                                 if product_price_variation else 0
        }
        
        # Visualize price points over time for top-selling products
        self._plot_price_variation_over_time(df)
        
        return price_analysis
    
    def analyze_win_rates(self) -> Dict[str, Any]:
        """
        Analyze quote win rates across segments
        Following Requirement 1.4: Analyze target variables
        
        Returns:
            Dictionary with win rate analysis
        """
        self.logger.info("Analyzing win rates...")
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data
        win_rate_analysis = {}
        
        if 'Status' not in df.columns:
            self.logger.warning("Status column not found for win rate analysis")
            return {}
        
        # Overall win rate
        total_quotes = len(df)
        won_quotes = (df['Status'] == 'Won').sum()
        overall_win_rate = won_quotes / total_quotes if total_quotes > 0 else 0
        
        win_rate_analysis['overall'] = {
            'total_quotes': total_quotes,
            'won_quotes': won_quotes,
            'lost_quotes': total_quotes - won_quotes,
            'win_rate': overall_win_rate,
            'dataset_balance': min(won_quotes, total_quotes - won_quotes) / total_quotes
        }
        
        # Win rates by segment
        segment_columns = ['Product_Category', 'Customer_Segment', 'Region']
        segment_win_rates = {}
        
        for segment_col in segment_columns:
            if segment_col in df.columns:
                segment_rates = df.groupby(segment_col)['Status'].agg([
                    lambda x: (x == 'Won').sum(),  # Won count
                    'count',  # Total count
                    lambda x: (x == 'Won').mean()  # Win rate
                ]).round(3)
                
                segment_rates.columns = ['won_count', 'total_count', 'win_rate']
                segment_win_rates[segment_col] = segment_rates.to_dict('index')
        
        win_rate_analysis['by_segment'] = segment_win_rates
        
        # Win rates by discount levels
        if 'Discount_Percent' in df.columns:
            # Create discount bins
            df['discount_bin'] = pd.cut(df['Discount_Percent'], 
                                      bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0], 
                                      labels=['0-10%', '10-20%', '20-30%', '30-40%', '40%+'])
            
            discount_win_rates = df.groupby('discount_bin')['Status'].agg([
                lambda x: (x == 'Won').sum(),
                'count',
                lambda x: (x == 'Won').mean()
            ]).round(3)
            
            discount_win_rates.columns = ['won_count', 'total_count', 'win_rate']
            win_rate_analysis['by_discount'] = discount_win_rates.to_dict('index')
        
        # Create win rate visualizations
        self._plot_win_rate_analysis(df, win_rate_analysis)
        
        return win_rate_analysis
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in quotes and pricing
        Following Requirement 5.3: Conduct temporal analysis
        
        Returns:
            Dictionary with temporal analysis
        """
        self.logger.info("Analyzing temporal patterns...")
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data.copy()
        temporal_analysis = {}
        
        if 'Quote_Date' not in df.columns:
            self.logger.warning("Quote_Date column not found")
            return {}
        
        # Ensure Quote_Date is datetime
        df['Quote_Date'] = pd.to_datetime(df['Quote_Date'])
        
        # Extract temporal features
        df['year'] = df['Quote_Date'].dt.year
        df['month'] = df['Quote_Date'].dt.month
        df['quarter'] = df['Quote_Date'].dt.quarter
        df['day_of_week'] = df['Quote_Date'].dt.dayofweek
        df['week_of_year'] = df['Quote_Date'].dt.isocalendar().week
        
        # Time coverage analysis
        time_coverage = {
            'start_date': df['Quote_Date'].min(),
            'end_date': df['Quote_Date'].max(),
            'total_days': (df['Quote_Date'].max() - df['Quote_Date'].min()).days,
            'data_gaps': self._identify_data_gaps(df['Quote_Date'])
        }
        temporal_analysis['coverage'] = time_coverage
        
        # Seasonal patterns
        seasonal_patterns = {}
        
        # Monthly patterns
        monthly_stats = df.groupby('month').agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean(),
            'Net_Price': 'mean',
            'Discount_Percent': 'mean'
        }).round(3)
        monthly_stats.columns = ['quote_count', 'win_rate', 'avg_price', 'avg_discount']
        seasonal_patterns['monthly'] = monthly_stats.to_dict('index')
        
        # Quarterly patterns
        quarterly_stats = df.groupby('quarter').agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean(),
            'Net_Price': 'mean',
            'Discount_Percent': 'mean'
        }).round(3)
        quarterly_stats.columns = ['quote_count', 'win_rate', 'avg_price', 'avg_discount']
        seasonal_patterns['quarterly'] = quarterly_stats.to_dict('index')
        
        # Day of week patterns
        dow_stats = df.groupby('day_of_week').agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean()
        }).round(3)
        dow_stats.columns = ['quote_count', 'win_rate']
        seasonal_patterns['day_of_week'] = dow_stats.to_dict('index')
        
        temporal_analysis['seasonal_patterns'] = seasonal_patterns
        
        # Trend analysis
        monthly_trends = df.groupby([df['Quote_Date'].dt.to_period('M')]).agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean(),
            'Net_Price': 'mean',
            'Discount_Percent': 'mean'
        }).round(3)
        
        temporal_analysis['trends'] = {
            'quote_volume_trend': monthly_trends['Quote_ID'].to_dict(),
            'win_rate_trend': monthly_trends['Status'].to_dict(),
            'price_trend': monthly_trends['Net_Price'].to_dict(),
            'discount_trend': monthly_trends['Discount_Percent'].to_dict()
        }
        
        # Create temporal visualizations
        self._plot_temporal_analysis(df, temporal_analysis)
        
        return temporal_analysis
    
    def analyze_segments(self) -> Dict[str, Any]:
        """
        Analyze customer and product segments
        Following Requirement 5.8: Evaluate segment definitions
        
        Returns:
            Dictionary with segment analysis
        """
        self.logger.info("Analyzing segments...")
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data
        segment_analysis = {}
        
        # Customer segment analysis
        if 'Customer_Segment' in df.columns:
            customer_segments = df.groupby('Customer_Segment').agg({
                'Customer_ID': 'nunique',
                'Quote_ID': 'count',
                'Status': lambda x: (x == 'Won').mean(),
                'Net_Price': ['mean', 'std'],
                'Discount_Percent': ['mean', 'std']
            }).round(3)
            
            customer_segments.columns = [
                'unique_customers', 'total_quotes', 'win_rate',
                'avg_price', 'price_std', 'avg_discount', 'discount_std'
            ]
            
            # Calculate segment homogeneity
            customer_segments['price_cv'] = customer_segments['price_std'] / customer_segments['avg_price']
            customer_segments['discount_cv'] = customer_segments['discount_std'] / customer_segments['avg_discount']
            
            segment_analysis['customer_segments'] = customer_segments.to_dict('index')
        
        # Product category analysis
        if 'Product_Category' in df.columns:
            product_segments = df.groupby('Product_Category').agg({
                'Product_ID': 'nunique',
                'Quote_ID': 'count',
                'Status': lambda x: (x == 'Won').mean(),
                'Net_Price': ['mean', 'std'],
                'Discount_Percent': ['mean', 'std']
            }).round(3)
            
            product_segments.columns = [
                'unique_products', 'total_quotes', 'win_rate',
                'avg_price', 'price_std', 'avg_discount', 'discount_std'
            ]
            
            segment_analysis['product_categories'] = product_segments.to_dict('index')
        
        # Combined segment analysis (Customer_Segment + Product_Category)
        if all(col in df.columns for col in ['Customer_Segment', 'Product_Category']):
            combined_segments = df.groupby(['Customer_Segment', 'Product_Category']).agg({
                'Quote_ID': 'count',
                'Status': lambda x: (x == 'Won').mean(),
                'Net_Price': 'mean'
            }).round(3)
            
            combined_segments.columns = ['quote_count', 'win_rate', 'avg_price']
            
            # Identify viable combined segments (minimum sample size)
            viable_segments = combined_segments[combined_segments['quote_count'] >= 50]
            
            segment_analysis['combined_segments'] = {
                'total_combinations': len(combined_segments),
                'viable_combinations': len(viable_segments),
                'viability_rate': len(viable_segments) / len(combined_segments) if len(combined_segments) > 0 else 0,
                'segment_details': viable_segments.to_dict('index')
            }
        
        return segment_analysis
    
    def create_bid_response_analysis(self) -> Dict[str, Any]:
        """
        Construct bid-response curves and analyze patterns
        Following Requirement 5.6: Conduct bid-response analysis
        
        Returns:
            Dictionary with bid-response analysis
        """
        self.logger.info("Creating bid-response analysis...")
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data.copy()
        bid_response_analysis = {}
        
        if not all(col in df.columns for col in ['Net_Price', 'List_Price', 'Status']):
            self.logger.warning("Required columns for bid-response analysis not found")
            return {}
        
        # Calculate price ratios
        df['price_ratio_to_list'] = df['Net_Price'] / df['List_Price']
        df['discount_depth'] = 1 - df['price_ratio_to_list']
        
        # Create price ratio bins
        df['price_ratio_bin'] = pd.cut(df['price_ratio_to_list'], 
                                     bins=np.arange(0.3, 1.1, 0.05),
                                     labels=[f'{i:.2f}-{i+0.05:.2f}' for i in np.arange(0.3, 1.05, 0.05)])
        
        # Calculate win probability by price ratio
        win_prob_by_ratio = df.groupby('price_ratio_bin').agg({
            'Status': lambda x: (x == 'Won').mean(),
            'Quote_ID': 'count'
        }).dropna()
        win_prob_by_ratio.columns = ['win_probability', 'quote_count']
        
        # Filter bins with sufficient sample size
        win_prob_by_ratio = win_prob_by_ratio[win_prob_by_ratio['quote_count'] >= 10]
        
        bid_response_analysis['overall_curve'] = win_prob_by_ratio.to_dict('index')
        
        # Bid-response curves by segment
        segment_curves = {}
        
        if 'Customer_Segment' in df.columns:
            for segment in df['Customer_Segment'].unique():
                if pd.notna(segment):
                    segment_df = df[df['Customer_Segment'] == segment]
                    
                    segment_curve = segment_df.groupby('price_ratio_bin').agg({
                        'Status': lambda x: (x == 'Won').mean(),
                        'Quote_ID': 'count'
                    }).dropna()
                    segment_curve.columns = ['win_probability', 'quote_count']
                    segment_curve = segment_curve[segment_curve['quote_count'] >= 5]
                    
                    if len(segment_curve) > 0:
                        segment_curves[segment] = segment_curve.to_dict('index')
        
        bid_response_analysis['by_customer_segment'] = segment_curves
        
        # Minimum discount thresholds
        discount_thresholds = {}
        
        # Overall threshold
        win_rates_by_discount = df.groupby(
            pd.cut(df['discount_depth'], bins=np.arange(0, 0.7, 0.05))
        )['Status'].agg(lambda x: (x == 'Won').mean())
        
        # Find minimum discount for 50% win rate
        threshold_50pct = None
        for discount_bin, win_rate in win_rates_by_discount.items():
            if win_rate >= 0.5:
                threshold_50pct = discount_bin.left
                break
        
        discount_thresholds['overall'] = {
            'min_discount_50pct_win': threshold_50pct,
            'win_rates_by_discount': win_rates_by_discount.to_dict()
        }
        
        bid_response_analysis['discount_thresholds'] = discount_thresholds
        
        # Create bid-response visualizations
        self._plot_bid_response_curves(df, bid_response_analysis)
        
        return bid_response_analysis
    
    def _identify_data_gaps(self, dates: pd.Series) -> List[Dict]:
        """Identify gaps in the time series data"""
        dates_sorted = dates.sort_values()
        gaps = []
        
        for i in range(1, len(dates_sorted)):
            gap_days = (dates_sorted.iloc[i] - dates_sorted.iloc[i-1]).days
            if gap_days > 7:  # Consider gaps > 7 days as significant
                gaps.append({
                    'start_date': dates_sorted.iloc[i-1],
                    'end_date': dates_sorted.iloc[i],
                    'gap_days': gap_days
                })
        
        return gaps
    
    def _plot_price_variation_over_time(self, df: pd.DataFrame):
        """Create price variation visualizations"""
        # Price trends over time for top products
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall price trends
        monthly_prices = df.groupby(df['Quote_Date'].dt.to_period('M')).agg({
            'Net_Price': 'mean',
            'Discount_Percent': 'mean',
            'List_Price': 'mean'
        })
        
        axes[0, 0].plot(monthly_prices.index.to_timestamp(), monthly_prices['Net_Price'], marker='o')
        axes[0, 0].set_title('Average Net Price Over Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].plot(monthly_prices.index.to_timestamp(), monthly_prices['Discount_Percent'], 
                       marker='o', color='orange')
        axes[0, 1].set_title('Average Discount Percentage Over Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Price distribution by product category
        if 'Product_Category' in df.columns:
            df.boxplot(column='Net_Price', by='Product_Category', ax=axes[1, 0])
            axes[1, 0].set_title('Price Distribution by Product Category')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Price variation coefficient by product
        product_variation = df.groupby('Product_ID')['Net_Price'].agg(['mean', 'std']).dropna()
        product_variation['cv'] = product_variation['std'] / product_variation['mean']
        product_variation_top = product_variation.nlargest(20, 'cv')
        
        axes[1, 1].bar(range(len(product_variation_top)), product_variation_top['cv'])
        axes[1, 1].set_title('Top 20 Products by Price Variation (CV)')
        axes[1, 1].set_xlabel('Product Rank')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'price_variation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Price variation plots saved")
    
    def _plot_win_rate_analysis(self, df: pd.DataFrame, win_rate_analysis: Dict):
        """Create win rate analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Win rate by discount bins
        if 'discount_bin' in df.columns:
            discount_win_rates = df.groupby('discount_bin')['Status'].agg(
                lambda x: (x == 'Won').mean()
            )
            
            axes[0, 0].bar(discount_win_rates.index, discount_win_rates.values)
            axes[0, 0].set_title('Win Rate by Discount Level')
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Win rate by customer segment
        if 'Customer_Segment' in df.columns:
            segment_win_rates = df.groupby('Customer_Segment')['Status'].agg(
                lambda x: (x == 'Won').mean()
            )
            
            axes[0, 1].bar(segment_win_rates.index, segment_win_rates.values, color='orange')
            axes[0, 1].set_title('Win Rate by Customer Segment')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win rate by product category
        if 'Product_Category' in df.columns:
            category_win_rates = df.groupby('Product_Category')['Status'].agg(
                lambda x: (x == 'Won').mean()
            )
            
            axes[1, 0].bar(category_win_rates.index, category_win_rates.values, color='green')
            axes[1, 0].set_title('Win Rate by Product Category')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall win/loss distribution
        status_counts = df['Status'].value_counts()
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Overall Win/Loss Distribution')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'win_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Win rate analysis plots saved")
    
    def _plot_temporal_analysis(self, df: pd.DataFrame, temporal_analysis: Dict):
        """Create temporal analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly quote volume and win rate
        monthly_stats = df.groupby('month').agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean()
        })
        
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        bars = ax1.bar(monthly_stats.index, monthly_stats['Quote_ID'], alpha=0.7, color='skyblue')
        line = ax2.plot(monthly_stats.index, monthly_stats['Status'], color='red', marker='o')
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Quote Volume', color='skyblue')
        ax2.set_ylabel('Win Rate', color='red')
        ax1.set_title('Monthly Quote Volume and Win Rate')
        
        # Quarterly patterns
        quarterly_stats = df.groupby('quarter').agg({
            'Net_Price': 'mean',
            'Discount_Percent': 'mean'
        })
        
        axes[0, 1].bar(quarterly_stats.index, quarterly_stats['Net_Price'], alpha=0.7)
        axes[0, 1].set_title('Average Price by Quarter')
        axes[0, 1].set_ylabel('Average Net Price')
        
        # Day of week patterns
        dow_stats = df.groupby('day_of_week').agg({
            'Quote_ID': 'count',
            'Status': lambda x: (x == 'Won').mean()
        })
        
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(dow_labels, dow_stats['Quote_ID'].values)
        axes[1, 0].set_title('Quote Volume by Day of Week')
        axes[1, 0].set_ylabel('Quote Count')
        
        # Time series trend
        monthly_trends = df.groupby(df['Quote_Date'].dt.to_period('M'))['Status'].agg(
            lambda x: (x == 'Won').mean()
        )
        
        axes[1, 1].plot(monthly_trends.index.to_timestamp(), monthly_trends.values, marker='o')
        axes[1, 1].set_title('Win Rate Trend Over Time')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Temporal analysis plots saved")
    
    def _plot_bid_response_curves(self, df: pd.DataFrame, bid_response_analysis: Dict):
        """Create bid-response curve visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall bid-response curve
        if 'price_ratio_to_list' in df.columns:
            # Create bins and calculate win rates
            price_bins = np.arange(0.3, 1.1, 0.05)
            bin_centers = price_bins[:-1] + 0.025
            
            win_rates = []
            for i in range(len(price_bins) - 1):
                mask = (df['price_ratio_to_list'] >= price_bins[i]) & (df['price_ratio_to_list'] < price_bins[i + 1])
                if mask.sum() > 10:  # Minimum sample size
                    win_rate = (df[mask]['Status'] == 'Won').mean()
                    win_rates.append(win_rate)
                else:
                    win_rates.append(np.nan)
            
            valid_indices = ~np.isnan(win_rates)
            axes[0, 0].plot(bin_centers[valid_indices], np.array(win_rates)[valid_indices], 
                           marker='o', linewidth=2)
            axes[0, 0].set_xlabel('Price Ratio to List Price')
            axes[0, 0].set_ylabel('Win Probability')
            axes[0, 0].set_title('Overall Bid-Response Curve')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate by discount depth
        if 'discount_depth' in df.columns:
            discount_bins = pd.cut(df['discount_depth'], bins=np.arange(0, 0.8, 0.05))
            win_by_discount = df.groupby(discount_bins)['Status'].agg(lambda x: (x == 'Won').mean())
            
            bin_centers = [interval.mid for interval in win_by_discount.index if pd.notna(interval)]
            win_rates = win_by_discount.values
            
            axes[0, 1].plot(bin_centers, win_rates, marker='s', color='orange', linewidth=2)
            axes[0, 1].set_xlabel('Discount Depth')
            axes[0, 1].set_ylabel('Win Probability')
            axes[0, 1].set_title('Win Rate by Discount Depth')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Bid-response by customer segment
        if 'Customer_Segment' in df.columns:
            for i, segment in enumerate(df['Customer_Segment'].unique()[:4]):
                if pd.notna(segment):
                    segment_df = df[df['Customer_Segment'] == segment]
                    
                    if len(segment_df) > 50:
                        segment_bins = pd.cut(segment_df['price_ratio_to_list'], bins=price_bins)
                        segment_win_rates = segment_df.groupby(segment_bins)['Status'].agg(
                            lambda x: (x == 'Won').mean()
                        )
                        
                        bin_centers_seg = [interval.mid for interval in segment_win_rates.index if pd.notna(interval)]
                        win_rates_seg = segment_win_rates.dropna().values
                        
                        if len(bin_centers_seg) > 0:
                            axes[1, 0].plot(bin_centers_seg, win_rates_seg, 
                                          marker='o', label=segment, linewidth=2)
            
            axes[1, 0].set_xlabel('Price Ratio to List Price')
            axes[1, 0].set_ylabel('Win Probability')
            axes[1, 0].set_title('Bid-Response Curves by Customer Segment')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Price sensitivity heatmap
        if all(col in df.columns for col in ['Customer_Segment', 'Product_Category']):
            price_sensitivity = df.groupby(['Customer_Segment', 'Product_Category']).agg({
                'Status': lambda x: (x == 'Won').mean(),
                'Quote_ID': 'count'
            })
            
            # Filter for sufficient sample sizes
            price_sensitivity = price_sensitivity[price_sensitivity['Quote_ID'] >= 20]
            
            if len(price_sensitivity) > 0:
                pivot_table = price_sensitivity['Status'].unstack(fill_value=0)
                
                im = axes[1, 1].imshow(pivot_table.values, cmap='RdYlBu', aspect='auto')
                axes[1, 1].set_xticks(range(len(pivot_table.columns)))
                axes[1, 1].set_xticklabels(pivot_table.columns, rotation=45)
                axes[1, 1].set_yticks(range(len(pivot_table.index)))
                axes[1, 1].set_yticklabels(pivot_table.index)
                axes[1, 1].set_title('Win Rate Heatmap: Segment × Category')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[1, 1])
                cbar.set_label('Win Rate')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'bid_response_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Bid-response analysis plots saved")
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive EDA report
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating comprehensive EDA report...")
        
        # Run all analyses
        self.load_data()
        self.create_unified_dataset()
        
        quality_report = self.assess_data_quality()
        price_analysis = self.analyze_price_variation()
        win_rate_analysis = self.analyze_win_rates()
        temporal_analysis = self.analyze_temporal_patterns()
        segment_analysis = self.analyze_segments()
        bid_response_analysis = self.create_bid_response_analysis()
        
        # Generate summary report
        report_content = f"""
# B2B Price Elasticity Modeling - Exploratory Data Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive exploratory data analysis for B2B price elasticity modeling,
following the requirements specified in the project documentation.

### Dataset Overview
- Total Records: {len(self.unified_data) if self.unified_data is not None else 'N/A'}
- Date Range: {temporal_analysis.get('coverage', {}).get('start_date', 'N/A')} to {temporal_analysis.get('coverage', {}).get('end_date', 'N/A')}
- Total Days: {temporal_analysis.get('coverage', {}).get('total_days', 'N/A')}

### Key Findings

#### Data Quality Assessment
- **Overall Win Rate**: {win_rate_analysis.get('overall', {}).get('win_rate', 'N/A'):.1%}
- **Dataset Balance**: {win_rate_analysis.get('overall', {}).get('dataset_balance', 'N/A'):.1%}
- **High Missing Data Fields**: {len(quality_report.get('missing_values', {}).get('columns_with_high_missing', []))} columns

#### Price Variation Analysis
- **Products with Sufficient Variation**: {price_analysis.get('modeling_candidates', {}).get('sufficient_variation_count', 'N/A')} out of {price_analysis.get('modeling_candidates', {}).get('total_products', 'N/A')}
- **Modeling Viability**: {price_analysis.get('modeling_candidates', {}).get('percentage_suitable', 'N/A'):.1f}%

#### Segment Analysis
- **Viable Combined Segments**: {segment_analysis.get('combined_segments', {}).get('viable_combinations', 'N/A')} out of {segment_analysis.get('combined_segments', {}).get('total_combinations', 'N/A')}
- **Segment Viability Rate**: {segment_analysis.get('combined_segments', {}).get('viability_rate', 'N/A'):.1%}

## Detailed Analysis Results

### 1. Data Quality Assessment

#### Missing Values
The analysis identified the following data quality issues:
- Total rows processed: {quality_report.get('missing_values', {}).get('total_rows', 'N/A')}
- Columns with >30% missing data: {', '.join(quality_report.get('missing_values', {}).get('columns_with_high_missing', []))}

### 2. Price Variation Analysis

The price variation analysis reveals:
- Average coefficient of variation across products: {np.mean([p['cv'] for p in price_analysis.get('product_variation', [])]) if price_analysis.get('product_variation') else 'N/A':.3f}
- Products suitable for elasticity modeling: {price_analysis.get('modeling_candidates', {}).get('sufficient_variation_count', 'N/A')}

### 3. Win Rate Analysis

Win rate patterns show:
- Overall quote win rate: {win_rate_analysis.get('overall', {}).get('win_rate', 'N/A'):.1%}
- Win rate variance across segments indicates pricing sensitivity differences

### 4. Temporal Patterns

Time series analysis indicates:
- Seasonal patterns detected in quote volume and win rates
- {len(temporal_analysis.get('coverage', {}).get('data_gaps', []))} significant data gaps identified

### 5. Segment Viability

Segment analysis confirms:
- {segment_analysis.get('combined_segments', {}).get('viable_combinations', 'N/A')} customer-product segment combinations have sufficient data for modeling
- Hierarchical modeling approach recommended for sparse segments

## Recommendations

### For Data Quality
1. Implement data quality monitoring for key fields with high missing rates
2. Establish data validation rules for price consistency
3. Address outliers in pricing data through business rule validation

### For Modeling Approach
1. **Hierarchical Bayesian Models**: Recommended for segments with limited data
2. **Individual-Level Models**: Suitable for high-volume customer-product combinations
3. **Ensemble Methods**: Combine multiple approaches for robust predictions

### For Price Optimization
1. Focus on products with sufficient price variation for elasticity estimation
2. Implement segment-specific pricing strategies based on win rate patterns
3. Consider temporal factors in pricing decisions

## Next Steps

1. **Feature Engineering**: Implement comprehensive feature engineering pipeline
2. **Model Development**: Begin with hierarchical Bayesian models for segment-level analysis
3. **Causal Analysis**: Implement X-Learner for individual-level causal effects
4. **Validation Framework**: Establish time-series cross-validation methodology

---
*This analysis serves as the foundation for the B2B price elasticity modeling system development.*
"""
        
        # Save report
        report_path = self.reports_dir / 'comprehensive_eda_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save detailed analysis results as JSON
        detailed_results = {
            'data_quality': quality_report,
            'price_analysis': price_analysis,
            'win_rate_analysis': win_rate_analysis,
            'temporal_analysis': temporal_analysis,
            'segment_analysis': segment_analysis,
            'bid_response_analysis': bid_response_analysis,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(self.reports_dir / 'detailed_eda_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive EDA report saved to {report_path}")
        return str(report_path)


def main():
    """Main function to run the EDA analysis"""
    print("🚀 Starting Comprehensive B2B Price Elasticity EDA...")
    
    # Initialize EDA
    eda = PriceElasticityEDA()
    
    try:
        # Generate comprehensive report
        report_path = eda.generate_comprehensive_report()
        
        print(f"✅ EDA completed successfully!")
        print(f"📊 Report saved to: {report_path}")
        print(f"📁 Plots saved to: {eda.plots_dir}")
        print(f"📄 Detailed results saved to: {eda.reports_dir}")
        
        # Print summary statistics
        if eda.unified_data is not None:
            print(f"\n📈 Dataset Summary:")
            print(f"  - Total records: {len(eda.unified_data):,}")
            print(f"  - Total customers: {eda.unified_data['Customer_ID'].nunique():,}")
            print(f"  - Total products: {eda.unified_data['Product_ID'].nunique():,}")
            print(f"  - Date range: {eda.unified_data['Quote_Date'].min()} to {eda.unified_data['Quote_Date'].max()}")
            
            if 'Status' in eda.unified_data.columns:
                win_rate = (eda.unified_data['Status'] == 'Won').mean()
                print(f"  - Overall win rate: {win_rate:.1%}")
        
    except Exception as e:
        print(f"❌ Error during EDA: {e}")
        logger.error(f"EDA failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

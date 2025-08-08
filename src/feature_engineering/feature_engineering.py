"""
Comprehensive Feature Engineering for B2B Price Elasticity Modeling
This script implements all feature engineering requirements from the specifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')


class PriceElasticityFeatureEngineering:
    """
    Comprehensive Feature Engineering for Price Elasticity Modeling
    Implements all requirements from the specification document
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineering with configuration
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config_loader if config is None else config
        self.logger = logger
        self.scalers = {}
        self.encoders = {}
        self.feature_metadata = {}
        
        # Get feature engineering config
        self.fe_config = self.config.get('feature_engineering', {})
        
        self.logger.info("Feature Engineering initialized")
    
    def create_price_dynamics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price dynamics features as per Requirement 3.1
        
        Args:
            df: Input DataFrame with price and date information
            
        Returns:
            DataFrame with price dynamics features
        """
        self.logger.info("Creating price dynamics features...")
        
        df_features = df.copy()
        
        # Ensure Quote_Date is datetime and properly formatted
        has_quote_date = 'Quote_Date' in df_features.columns
        if has_quote_date:
            df_features['Quote_Date'] = pd.to_datetime(df_features['Quote_Date'])
            # Sort by date for rolling calculations
            df_features = df_features.sort_values(['Quote_Date']).reset_index(drop=True)
        else:
            # Sort by index if no date column
            df_features = df_features.reset_index(drop=True)
        
        # Get configuration
        windows = self.fe_config.get('price_dynamics', {}).get('windows', [7, 30, 90])
        volatility_window = self.fe_config.get('price_dynamics', {}).get('volatility_window', 90)
        
        # Price ratio to category average (simplified approach)
        if all(col in df_features.columns for col in ['Net_Price', 'Product_Category']):
            # Calculate simple category average
            category_avg = df_features.groupby('Product_Category')['Net_Price'].transform('mean')
            df_features['price_ratio_to_category_avg'] = (
                df_features['Net_Price'] / category_avg
            ).fillna(1.0)
        
        # Discount depth calculation
        if all(col in df_features.columns for col in ['List_Price', 'Net_Price']):
            df_features['discount_depth'] = (
                df_features['List_Price'] - df_features['Net_Price']
            ) / df_features['List_Price']
            df_features['discount_depth'] = df_features['discount_depth'].clip(0, 1)
        
        # Price volatility measures (simplified approach)
        if 'Net_Price' in df_features.columns:
            # Calculate price volatility per product (standard deviation)
            product_price_stats = df_features.groupby('Product_ID')['Net_Price'].agg(['std', 'mean']).fillna(0)
            product_price_stats.columns = ['price_volatility', 'price_mean']
            
            # Merge back to main dataframe
            df_features = df_features.merge(
                product_price_stats,
                left_on='Product_ID',
                right_index=True,
                how='left'
            )
            
            # Price coefficient of variation
            df_features['price_cv'] = (
                df_features['price_volatility'] / df_features['price_mean']
            ).fillna(0)
        
        # Days since last price change
        if all(col in df_features.columns for col in ['Product_ID', 'Net_Price', 'Quote_Date']):
            price_change_threshold = self.fe_config.get('price_dynamics', {}).get('price_change_threshold', 0.05)
            
            def calculate_days_since_price_change(group):
                group = group.sort_values('Quote_Date')
                group['price_pct_change'] = group['Net_Price'].pct_change().fillna(0)
                group['significant_change'] = abs(group['price_pct_change']) > price_change_threshold
                
                days_since = []
                last_change_date = None
                
                for idx, row in group.iterrows():
                    current_date = row['Quote_Date']
                    
                    if row['significant_change'] or last_change_date is None:
                        last_change_date = current_date
                        days_since.append(0)
                    else:
                        days_since.append((current_date - last_change_date).days)
                
                group['days_since_last_price_change'] = days_since
                return group
            
            df_features = df_features.groupby('Product_ID').apply(
                calculate_days_since_price_change
            ).reset_index(drop=True)
        
        self.logger.info(f"Created {len([col for col in df_features.columns if col.startswith(('price_ratio', 'discount_depth', 'price_volatility', 'price_cv', 'days_since'))])} price dynamics features")
        
        return df_features
    
    def create_competitive_positioning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create competitive positioning features as per Requirement 3.2
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with competitive positioning features
        """
        self.logger.info("Creating competitive positioning features...")
        
        df_features = df.copy()
        
        # Competition status index mapping
        if 'Competition_Status' in df_features.columns:
            competition_mapping = {
                'None': 1.0,
                'Low': 2.0,
                'Medium': 3.0,
                'High': 4.0
            }
            df_features['competition_status_index'] = df_features['Competition_Status'].map(
                competition_mapping
            ).fillna(2.5)
        
        # Competitive premium estimate
        if all(col in df_features.columns for col in ['Net_Price', 'Product_Category']):
            # Calculate market median by category
            market_median = df_features.groupby('Product_Category')['Net_Price'].transform('median')
            df_features['competitive_premium'] = (
                df_features['Net_Price'] - market_median
            ) / market_median
        
        # Product lifecycle price index
        if all(col in df_features.columns for col in ['Product_ID', 'Net_Price', 'Quote_Date']):
            def calculate_lifecycle_price_index(group):
                group = group.sort_values('Quote_Date')
                
                # Calculate price trend over time
                if len(group) > 3:
                    # Simple linear trend
                    x = np.arange(len(group))
                    prices = group['Net_Price'].values
                    
                    # Calculate slope
                    slope = np.polyfit(x, prices, 1)[0]
                    
                    # Normalize by mean price
                    mean_price = prices.mean()
                    normalized_slope = slope / mean_price if mean_price > 0 else 0
                    
                    group['product_lifecycle_price_index'] = normalized_slope
                else:
                    group['product_lifecycle_price_index'] = 0
                
                return group
            
            df_features = df_features.groupby('Product_ID').apply(
                calculate_lifecycle_price_index
            ).reset_index(drop=True)
        
        # Market position relative to category
        if all(col in df_features.columns for col in ['Net_Price', 'Product_Category']):
            category_stats = df_features.groupby('Product_Category')['Net_Price'].agg(['min', 'max', 'mean'])
            df_features = df_features.merge(
                category_stats.add_suffix('_category'), 
                left_on='Product_Category', 
                right_index=True, 
                how='left'
            )
            
            # Market position (0 = bottom, 1 = top of category)
            df_features['market_position_in_category'] = (
                (df_features['Net_Price'] - df_features['min_category']) / 
                (df_features['max_category'] - df_features['min_category'])
            ).fillna(0.5).clip(0, 1)
        
        self.logger.info("Created competitive positioning features")
        return df_features
    
    def create_customer_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer value features as per Requirement 3.3
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with customer value features
        """
        self.logger.info("Creating customer value features...")
        
        df_features = df.copy()
        
        # RFM metrics calculation
        if all(col in df_features.columns for col in ['Customer_ID', 'Quote_Date', 'Net_Price']):
            current_date = df_features['Quote_Date'].max()
            
            # Customer-level RFM calculation
            customer_rfm = df_features.groupby('Customer_ID').agg({
                'Quote_Date': ['max', 'count'],
                'Net_Price': ['sum', 'mean']
            }).round(2)
            
            customer_rfm.columns = ['last_quote_date', 'quote_frequency', 'total_monetary', 'avg_monetary']
            
            # Recency (days since last quote)
            customer_rfm['recency_days'] = (current_date - customer_rfm['last_quote_date']).dt.days
            
            # Calculate RFM scores
            n_quantiles = self.fe_config.get('customer', {}).get('rfm_quantiles', 5)
            
            customer_rfm['recency_score'] = pd.qcut(
                customer_rfm['recency_days'], 
                q=n_quantiles, 
                labels=range(n_quantiles, 0, -1)
            ).astype(float)
            
            customer_rfm['frequency_score'] = pd.qcut(
                customer_rfm['quote_frequency'].rank(method='first'), 
                q=n_quantiles, 
                labels=range(1, n_quantiles + 1)
            ).astype(float)
            
            customer_rfm['monetary_score'] = pd.qcut(
                customer_rfm['total_monetary'].rank(method='first'), 
                q=n_quantiles, 
                labels=range(1, n_quantiles + 1)
            ).astype(float)
            
            # Combined RFM score
            customer_rfm['rfm_combined_score'] = (
                customer_rfm['recency_score'] * 100 + 
                customer_rfm['frequency_score'] * 10 + 
                customer_rfm['monetary_score']
            )
            
            # Merge back to main dataframe
            df_features = df_features.merge(
                customer_rfm[['recency_days', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_combined_score']], 
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
        
        # Customer tenure calculation
        if all(col in df_features.columns for col in ['Customer_Since_Date', 'Quote_Date']):
            df_features['Customer_Since_Date'] = pd.to_datetime(df_features['Customer_Since_Date'])
            df_features['customer_tenure_days'] = (
                df_features['Quote_Date'] - df_features['Customer_Since_Date']
            ).dt.days
            
            # Tenure bins
            tenure_bins = self.fe_config.get('customer', {}).get('tenure_bins', [0, 90, 365, 730, 1825])
            df_features['customer_tenure_bin'] = pd.cut(
                df_features['customer_tenure_days'], 
                bins=tenure_bins + [float('inf')], 
                labels=['New', 'Short', 'Medium', 'Long', 'VeryLong']
            )
        
        # Customer Lifetime Value (CLV) calculation
        if all(col in df_features.columns for col in ['Customer_ID', 'Net_Price', 'customer_tenure_days']):
            clv_method = self.fe_config.get('customer', {}).get('clv_method', 'traditional')
            
            if clv_method == 'traditional':
                # Traditional CLV calculation
                customer_clv = df_features.groupby('Customer_ID').agg({
                    'Net_Price': ['mean', 'sum', 'count'],
                    'customer_tenure_days': 'first'
                })
                
                customer_clv.columns = ['avg_order_value', 'total_value', 'purchase_frequency', 'tenure_days']
                
                # Estimate purchase frequency per year
                customer_clv['annual_frequency'] = (
                    customer_clv['purchase_frequency'] * 365 / customer_clv['tenure_days'].clip(lower=1)
                )
                
                # Estimate customer lifespan (simple heuristic)
                customer_clv['estimated_lifespan_years'] = (customer_clv['tenure_days'] / 365).clip(lower=0.1)
                
                # CLV calculation (simplified)
                customer_clv['clv_estimate'] = (
                    customer_clv['avg_order_value'] * 
                    customer_clv['annual_frequency'] * 
                    customer_clv['estimated_lifespan_years'] * 
                    0.1  # Assumed profit margin
                )
                
                # Merge back
                df_features = df_features.merge(
                    customer_clv[['clv_estimate']], 
                    left_on='Customer_ID', 
                    right_index=True, 
                    how='left'
                )
        
        # Quote to order conversion rate (if we have sales data)
        if 'Status' in df_features.columns:
            customer_conversion = df_features.groupby('Customer_ID')['Status'].agg([
                lambda x: (x == 'Won').sum(),
                'count',
                lambda x: (x == 'Won').mean()
            ])
            customer_conversion.columns = ['won_quotes', 'total_quotes', 'conversion_rate']
            
            df_features = df_features.merge(
                customer_conversion[['conversion_rate']], 
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
        
        self.logger.info("Created customer value features")
        return df_features
    
    def create_product_hierarchy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product hierarchy features as per Requirement 3.5
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with product hierarchy features
        """
        self.logger.info("Creating product hierarchy features...")
        
        df_features = df.copy()
        
        # Category sales velocity
        if all(col in df_features.columns for col in ['Product_Category', 'Quote_Date']):
            velocity_periods = self.fe_config.get('product', {}).get('velocity_periods', [30, 90, 365])
            
            for period in velocity_periods:
                cutoff_date = df_features['Quote_Date'].max() - timedelta(days=period)
                recent_data = df_features[df_features['Quote_Date'] >= cutoff_date]
                
                category_velocity = recent_data.groupby('Product_Category')['Quote_ID'].count() / period
                category_velocity.name = f'category_sales_velocity_{period}d'
                
                df_features = df_features.merge(
                    category_velocity, 
                    left_on='Product_Category', 
                    right_index=True, 
                    how='left'
                )
                df_features[f'category_sales_velocity_{period}d'] = df_features[f'category_sales_velocity_{period}d'].fillna(0)
        
        # Category price elasticity proxy
        if all(col in df_features.columns for col in ['Product_Category', 'Net_Price', 'Status']):
            # Calculate simple price elasticity proxy for each category
            category_elasticity = []
            
            for category in df_features['Product_Category'].unique():
                if pd.notna(category):
                    cat_data = df_features[df_features['Product_Category'] == category]
                    
                    if len(cat_data) > 10:
                        # Simple correlation between price and win rate
                        win_rate_by_price = cat_data.groupby(
                            pd.cut(cat_data['Net_Price'], bins=5)
                        )['Status'].apply(lambda x: (x == 'Won').mean())
                        
                        if len(win_rate_by_price.dropna()) > 2:
                            # Calculate elasticity as correlation
                            price_midpoints = [interval.mid for interval in win_rate_by_price.index if pd.notna(interval)]
                            win_rates = win_rate_by_price.dropna().values
                            
                            if len(price_midpoints) > 1:
                                elasticity = np.corrcoef(price_midpoints, win_rates)[0, 1]
                                elasticity = np.nan_to_num(elasticity, 0)
                            else:
                                elasticity = 0
                        else:
                            elasticity = 0
                    else:
                        elasticity = 0
                    
                    category_elasticity.append({
                        'Product_Category': category,
                        'category_price_elasticity': elasticity
                    })
            
            category_elasticity_df = pd.DataFrame(category_elasticity)
            df_features = df_features.merge(
                category_elasticity_df, 
                on='Product_Category', 
                how='left'
            )
            df_features['category_price_elasticity'] = df_features['category_price_elasticity'].fillna(0)
        
        # Lifecycle stage index
        if 'Lifecycle_Stage' in df_features.columns:
            lifecycle_stages = self.fe_config.get('product', {}).get('lifecycle_stages', 
                                                                   ['Introduction', 'Growth', 'Maturity', 'Decline'])
            stage_mapping = {stage: i+1 for i, stage in enumerate(lifecycle_stages)}
            df_features['lifecycle_stage_index'] = df_features['Lifecycle_Stage'].map(stage_mapping).fillna(2)
        
        # Product newness metrics
        if all(col in df_features.columns for col in ['Launch_Date', 'Quote_Date']):
            df_features['Launch_Date'] = pd.to_datetime(df_features['Launch_Date'])
            df_features['product_age_days'] = (df_features['Quote_Date'] - df_features['Launch_Date']).dt.days
            
            # Product newness score (higher for newer products)
            max_age = df_features['product_age_days'].max()
            df_features['product_newness_score'] = 1 - (df_features['product_age_days'] / max_age)
            df_features['product_newness_score'] = df_features['product_newness_score'].clip(0, 1)
        
        # Product performance metrics
        if all(col in df_features.columns for col in ['Product_ID', 'Status']):
            product_performance = df_features.groupby('Product_ID').agg({
                'Status': [lambda x: (x == 'Won').mean(), 'count'],
                'Net_Price': 'mean'
            }).round(3)
            
            product_performance.columns = ['product_win_rate', 'product_quote_volume', 'product_avg_price']
            
            df_features = df_features.merge(
                product_performance, 
                left_on='Product_ID', 
                right_index=True, 
                how='left'
            )
        
        self.logger.info("Created product hierarchy features")
        return df_features
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features as per Requirement 3.7
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Creating temporal features...")
        
        df_features = df.copy()
        
        if 'Quote_Date' not in df_features.columns:
            self.logger.warning("Quote_Date column not found, skipping temporal features")
            return df_features
        
        # Ensure Quote_Date is datetime
        df_features['Quote_Date'] = pd.to_datetime(df_features['Quote_Date'])
        
        # Basic temporal features
        df_features['year'] = df_features['Quote_Date'].dt.year
        df_features['month'] = df_features['Quote_Date'].dt.month
        df_features['quarter'] = df_features['Quote_Date'].dt.quarter
        df_features['day_of_week'] = df_features['Quote_Date'].dt.dayofweek
        df_features['day_of_month'] = df_features['Quote_Date'].dt.day
        df_features['week_of_year'] = df_features['Quote_Date'].dt.isocalendar().week
        
        # Seasonal components with sine/cosine encoding
        seasonal_components = self.fe_config.get('temporal', {}).get('seasonal_components', ['month', 'quarter', 'day_of_week'])
        
        if 'month' in seasonal_components:
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        if 'quarter' in seasonal_components:
            df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
            df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
        
        if 'day_of_week' in seasonal_components:
            df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Holiday effects (simplified - US holidays)
        if self.fe_config.get('temporal', {}).get('holiday_effects', True):
            def is_holiday_period(date):
                """Simple holiday detection for major US holidays"""
                month, day = date.month, date.day
                
                # Major holidays affecting business
                holidays = [
                    (1, 1),   # New Year
                    (7, 4),   # Independence Day
                    (11, 11), # Veterans Day
                    (12, 25), # Christmas
                ]
                
                # Check if within 2 days of holiday
                for h_month, h_day in holidays:
                    if month == h_month and abs(day - h_day) <= 2:
                        return 1
                
                # Thanksgiving (4th Thursday of November)
                if month == 11 and date.weekday() == 3:  # Thursday
                    # Simple approximation for 4th Thursday
                    if 22 <= day <= 28:
                        return 1
                
                return 0
            
            df_features['is_holiday_period'] = df_features['Quote_Date'].apply(is_holiday_period)
            
            # Days to/from nearest holiday
            # This is a simplified version - could be enhanced with actual holiday calendar
            df_features['days_to_quarter_end'] = 90 - (df_features['Quote_Date'].dt.dayofyear % 90)
            df_features['days_to_year_end'] = (
                pd.to_datetime(df_features['Quote_Date'].dt.year.astype(str) + '-12-31') - 
                df_features['Quote_Date']
            ).dt.days
        
        # Trend components
        if self.fe_config.get('temporal', {}).get('trend_components', True):
            # Days since dataset start
            min_date = df_features['Quote_Date'].min()
            df_features['days_since_start'] = (df_features['Quote_Date'] - min_date).dt.days
            
            # Normalized time trend (0 to 1)
            max_days = df_features['days_since_start'].max()
            df_features['time_trend'] = df_features['days_since_start'] / max_days if max_days > 0 else 0
        
        # Business day indicators
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_month_end'] = (df_features['Quote_Date'].dt.day >= 25).astype(int)
        df_features['is_quarter_end'] = (df_features['month'] % 3 == 0).astype(int) & df_features['is_month_end']
        df_features['is_year_end'] = (df_features['month'] == 12).astype(int) & df_features['is_month_end']
        
        self.logger.info("Created temporal features")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and polynomial features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features...")
        
        df_features = df.copy()
        
        # Price-customer interactions
        if all(col in df_features.columns for col in ['Net_Price', 'Customer_Segment']):
            for segment in df_features['Customer_Segment'].unique():
                if pd.notna(segment):
                    df_features[f'price_x_{segment.lower().replace("-", "_")}_segment'] = (
                        df_features['Net_Price'] * (df_features['Customer_Segment'] == segment).astype(int)
                    )
        
        # Price-product category interactions
        if all(col in df_features.columns for col in ['Net_Price', 'Product_Category']):
            for category in df_features['Product_Category'].unique():
                if pd.notna(category):
                    df_features[f'price_x_{category.lower()}_category'] = (
                        df_features['Net_Price'] * (df_features['Product_Category'] == category).astype(int)
                    )
        
        # Discount-competition interactions
        if all(col in df_features.columns for col in ['discount_depth', 'Competition_Status']):
            for comp_status in df_features['Competition_Status'].unique():
                if pd.notna(comp_status):
                    df_features[f'discount_x_{comp_status.lower()}_competition'] = (
                        df_features['discount_depth'] * (df_features['Competition_Status'] == comp_status).astype(int)
                    )
        
        # Polynomial features for key variables
        key_numeric_features = ['Net_Price', 'discount_depth', 'customer_tenure_days']
        
        for feature in key_numeric_features:
            if feature in df_features.columns:
                # Squared terms
                df_features[f'{feature}_squared'] = df_features[feature] ** 2
                
                # Cubic terms (only for certain features)
                if feature in ['Net_Price', 'discount_depth']:
                    df_features[f'{feature}_cubed'] = df_features[feature] ** 3
        
        # Cross-product interactions for key pairs
        feature_pairs = [
            ('Net_Price', 'discount_depth'),
            ('customer_tenure_days', 'Net_Price'),
            ('rfm_combined_score', 'Net_Price') if 'rfm_combined_score' in df_features.columns else None
        ]
        
        for pair in feature_pairs:
            if pair and all(col in df_features.columns for col in pair):
                feature1, feature2 = pair
                df_features[f'{feature1}_x_{feature2}'] = df_features[feature1] * df_features[feature2]
        
        self.logger.info("Created interaction features")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features with various methods
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        self.logger.info("Encoding categorical features...")
        
        df_features = df.copy()
        categorical_config = self.fe_config.get('categorical', {})
        
        # Identify categorical columns
        categorical_columns = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude certain columns from encoding
        exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
        categorical_columns = [col for col in categorical_columns if col not in exclude_cols]
        
        for col in categorical_columns:
            if col in df_features.columns:
                unique_values = df_features[col].nunique()
                high_cardinality_threshold = categorical_config.get('high_cardinality_threshold', 50)
                
                if unique_values <= 10:
                    # One-hot encoding for low cardinality
                    if fit:
                        # Create dummy variables
                        dummies = pd.get_dummies(df_features[col], prefix=col, dummy_na=True)
                        self.encoders[f'{col}_dummies'] = dummies.columns.tolist()
                    else:
                        # Use stored columns
                        if f'{col}_dummies' in self.encoders:
                            dummies = pd.get_dummies(df_features[col], prefix=col, dummy_na=True)
                            # Ensure all expected columns exist
                            for expected_col in self.encoders[f'{col}_dummies']:
                                if expected_col not in dummies.columns:
                                    dummies[expected_col] = 0
                            dummies = dummies[self.encoders[f'{col}_dummies']]
                        else:
                            continue
                    
                    df_features = pd.concat([df_features.drop(columns=[col]), dummies], axis=1)
                
                elif unique_values <= high_cardinality_threshold:
                    # Label encoding for medium cardinality
                    if fit:
                        encoder = LabelEncoder()
                        df_features[f'{col}_encoded'] = encoder.fit_transform(df_features[col].fillna('Unknown'))
                        self.encoders[f'{col}_label'] = encoder
                    else:
                        if f'{col}_label' in self.encoders:
                            encoder = self.encoders[f'{col}_label']
                            # Handle unseen categories
                            df_features[f'{col}_encoded'] = df_features[col].fillna('Unknown').apply(
                                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                            )
                        else:
                            df_features[f'{col}_encoded'] = 0
                
                else:
                    # Target encoding for high cardinality (if target is available)
                    if 'Status' in df_features.columns and fit:
                        target_encoding_smoothing = categorical_config.get('target_encoding_smoothing', 10)
                        
                        # Calculate target encoding
                        target_mean = (df_features['Status'] == 'Won').mean()
                        category_stats = df_features.groupby(col).agg({
                            'Status': [lambda x: (x == 'Won').sum(), 'count']
                        })
                        category_stats.columns = ['wins', 'count']
                        
                        # Smoothed target encoding
                        category_stats['target_encoded'] = (
                            (category_stats['wins'] + target_encoding_smoothing * target_mean) / 
                            (category_stats['count'] + target_encoding_smoothing)
                        )
                        
                        self.encoders[f'{col}_target'] = category_stats['target_encoded'].to_dict()
                        df_features[f'{col}_target_encoded'] = df_features[col].map(
                            self.encoders[f'{col}_target']
                        ).fillna(target_mean)
                    
                    elif f'{col}_target' in self.encoders and not fit:
                        df_features[f'{col}_target_encoded'] = df_features[col].map(
                            self.encoders[f'{col}_target']
                        ).fillna(0.35)  # Default win rate
                    
                    else:
                        # Frequency encoding as fallback
                        if fit:
                            freq_encoding = df_features[col].value_counts().to_dict()
                            self.encoders[f'{col}_freq'] = freq_encoding
                        
                        if f'{col}_freq' in self.encoders:
                            df_features[f'{col}_freq_encoded'] = df_features[col].map(
                                self.encoders[f'{col}_freq']
                            ).fillna(0)
        
        self.logger.info("Categorical encoding completed")
        return df_features
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True, method: str = 'robust') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        self.logger.info(f"Scaling features using {method} method...")
        
        df_features = df.copy()
        
        # Identify numerical columns to scale
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns from scaling
        exclude_from_scaling = [
            'Quote_ID', 'Customer_ID', 'Product_ID', 'year', 'month', 'quarter', 
            'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 'is_holiday_period'
        ]
        
        numeric_columns = [col for col in numeric_columns 
                          if col not in exclude_from_scaling and col in df_features.columns]
        
        if numeric_columns:
            if method == 'standard':
                scaler_class = StandardScaler
            elif method == 'robust':
                scaler_class = RobustScaler
            else:
                from sklearn.preprocessing import MinMaxScaler
                scaler_class = MinMaxScaler
            
            if fit:
                scaler = scaler_class()
                df_features[numeric_columns] = scaler.fit_transform(df_features[numeric_columns])
                self.scalers['feature_scaler'] = scaler
            else:
                if 'feature_scaler' in self.scalers:
                    scaler = self.scalers['feature_scaler']
                    df_features[numeric_columns] = scaler.transform(df_features[numeric_columns])
        
        self.logger.info(f"Scaled {len(numeric_columns)} numerical features")
        return df_features
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values with various imputation strategies
        
        Args:
            df: Input DataFrame
            fit: Whether to fit imputers
            
        Returns:
            DataFrame with imputed missing values
        """
        self.logger.info("Handling missing values...")
        
        df_features = df.copy()
        
        # Strategy 1: Fill with business logic defaults
        business_defaults = {
            'Competition_Status': 'Medium',
            'Product_Objective': 'Profitability',
            'customer_tenure_days': 365,  # Default 1 year
            'conversion_rate': 0.35,  # Default win rate
        }
        
        for col, default_val in business_defaults.items():
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna(default_val)
        
        # Strategy 2: Forward fill for time-ordered data
        time_ordered_cols = [col for col in df_features.columns 
                           if any(x in col.lower() for x in ['price', 'discount', 'volatility'])]
        
        if 'Quote_Date' in df_features.columns and time_ordered_cols:
            df_features = df_features.sort_values('Quote_Date')
            df_features[time_ordered_cols] = df_features[time_ordered_cols].fillna(method='ffill')
        
        # Strategy 3: KNN imputation for remaining numerical columns
        numerical_cols_with_missing = df_features.select_dtypes(include=[np.number]).columns[
            df_features.select_dtypes(include=[np.number]).isnull().any()
        ].tolist()
        
        if numerical_cols_with_missing:
            if fit:
                imputer = KNNImputer(n_neighbors=5)
                df_features[numerical_cols_with_missing] = imputer.fit_transform(
                    df_features[numerical_cols_with_missing]
                )
                self.encoders['knn_imputer'] = imputer
            else:
                if 'knn_imputer' in self.encoders:
                    imputer = self.encoders['knn_imputer']
                    df_features[numerical_cols_with_missing] = imputer.transform(
                        df_features[numerical_cols_with_missing]
                    )
        
        # Strategy 4: Mode imputation for remaining categorical columns
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_features[col].isnull().any():
                if fit:
                    mode_val = df_features[col].mode().iloc[0] if len(df_features[col].mode()) > 0 else 'Unknown'
                    self.encoders[f'{col}_mode'] = mode_val
                    df_features[col] = df_features[col].fillna(mode_val)
                else:
                    if f'{col}_mode' in self.encoders:
                        df_features[col] = df_features[col].fillna(self.encoders[f'{col}_mode'])
        
        self.logger.info("Missing value imputation completed")
        return df_features
    
    def create_comprehensive_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create all features using the comprehensive pipeline
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Starting comprehensive feature engineering pipeline...")
        
        # Store original columns for reference
        original_columns = df.columns.tolist()
        
        # Step 1: Handle missing values first
        df_features = self.handle_missing_values(df, fit=fit)
        
        # Step 2: Create all feature groups
        df_features = self.create_price_dynamics_features(df_features)
        df_features = self.create_competitive_positioning_features(df_features)
        df_features = self.create_customer_value_features(df_features)
        df_features = self.create_product_hierarchy_features(df_features)
        df_features = self.create_temporal_features(df_features)
        
        # Step 3: Create interaction features
        df_features = self.create_interaction_features(df_features)
        
        # Step 4: Encode categorical features
        df_features = self.encode_categorical_features(df_features, fit=fit)
        
        # Step 4.5: Handle datetime columns
        df_features = self._handle_datetime_columns(df_features)
        
        # Step 5: Scale numerical features
        df_features = self.scale_features(df_features, fit=fit, method='robust')
        
        # Step 6: Feature selection and cleanup
        df_features = self._feature_cleanup(df_features)
        
        # Store feature metadata
        if fit:
            self.feature_metadata = {
                'original_columns': original_columns,
                'final_columns': df_features.columns.tolist(),
                'feature_count': len(df_features.columns),
                'added_features': len(df_features.columns) - len(original_columns),
                'processing_timestamp': datetime.now().isoformat()
            }
        
        self.logger.info(f"Feature engineering completed. Features: {len(original_columns)} → {len(df_features.columns)}")
        return df_features
    
    def _feature_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up features - remove highly correlated, constant, etc.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Performing feature cleanup...")
        
        df_clean = df.copy()
        
        # Remove constant features
        constant_features = []
        for col in df_clean.columns:
            if df_clean[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            df_clean = df_clean.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove features with too many missing values (>95%)
        high_missing_features = []
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() / len(df_clean) > 0.95:
                high_missing_features.append(col)
        
        if high_missing_features:
            df_clean = df_clean.drop(columns=high_missing_features)
            self.logger.info(f"Removed {len(high_missing_features)} features with >95% missing values")
        
        # Handle infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        return df_clean
    
    def _handle_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert datetime columns to numerical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with datetime columns converted to numerical features
        """
        df_features = df.copy()
        
        # Find datetime columns
        datetime_columns = [col for col in df_features.columns 
                          if 'datetime64' in str(df_features[col].dtype)]
        
        if datetime_columns:
            self.logger.info(f"Converting {len(datetime_columns)} datetime columns to numerical features")
            
            # Reference date for calculations (use dataset minimum)
            reference_date = None
            for col in datetime_columns:
                if reference_date is None:
                    reference_date = df_features[col].min()
                else:
                    reference_date = min(reference_date, df_features[col].min())
            
            for col in datetime_columns:
                # Convert to days since reference date
                df_features[f'{col}_days_since_ref'] = (
                    df_features[col] - reference_date
                ).dt.days.fillna(0)
                
                # Convert to unix timestamp
                df_features[f'{col}_timestamp'] = (
                    df_features[col].astype('int64') // 10**9
                ).fillna(0)  # Convert to seconds since epoch
                
                # Extract year and month as additional features
                df_features[f'{col}_year'] = df_features[col].dt.year.fillna(reference_date.year)
                df_features[f'{col}_month'] = df_features[col].dt.month.fillna(reference_date.month)
                
                # Remove original datetime column
                df_features = df_features.drop(columns=[col])
        
        return df_features
    
    def save_feature_engineering_artifacts(self, output_dir: str = "models/feature_engineering"):
        """
        Save feature engineering artifacts for reuse
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, output_path / 'scalers.pkl')
        
        # Save encoders
        if self.encoders:
            joblib.dump(self.encoders, output_path / 'encoders.pkl')
        
        # Save metadata
        if self.feature_metadata:
            import json
            with open(output_path / 'feature_metadata.json', 'w') as f:
                json.dump(self.feature_metadata, f, indent=2)
        
        self.logger.info(f"Feature engineering artifacts saved to {output_path}")
    
    def load_feature_engineering_artifacts(self, input_dir: str = "models/feature_engineering"):
        """
        Load feature engineering artifacts
        
        Args:
            input_dir: Directory to load artifacts from
        """
        input_path = Path(input_dir)
        
        # Load scalers
        scalers_path = input_path / 'scalers.pkl'
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
        
        # Load encoders
        encoders_path = input_path / 'encoders.pkl'
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
        
        # Load metadata
        metadata_path = input_path / 'feature_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.feature_metadata = json.load(f)
        
        self.logger.info(f"Feature engineering artifacts loaded from {input_path}")


def main():
    """Main function to demonstrate feature engineering"""
    print("🚀 Starting Feature Engineering Pipeline...")
    
    # Initialize feature engineering
    fe = PriceElasticityFeatureEngineering()
    
    # This would typically be called with real data from the EDA pipeline
    print("Feature engineering pipeline is ready!")
    print("Use fe.create_comprehensive_features(df) to process your data.")


if __name__ == "__main__":
    main()

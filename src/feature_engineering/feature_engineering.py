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
from statsmodels.tsa.seasonal import STL
from scipy import stats
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    def create_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced temporal features including lag features, rolling windows, and STL decomposition
        Following Requirements 23 and 12.1
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with advanced temporal features
        """
        self.logger.info("Creating advanced temporal features...")
        
        df_features = df.copy()
        
        if 'Quote_Date' not in df_features.columns:
            self.logger.warning("Quote_Date column not found, skipping advanced temporal features")
            return df_features
        
        # Ensure Quote_Date is datetime and sort
        df_features['Quote_Date'] = pd.to_datetime(df_features['Quote_Date'])
        df_features = df_features.sort_values(['Product_ID', 'Customer_ID', 'Quote_Date']).reset_index(drop=True)
        
        # 1. Lag Features (Requirement 23.1)
        lag_periods = self.fe_config.get('temporal', {}).get('lag_periods', [1, 2, 3, 7, 30])
        
        for lag in lag_periods:
            if 'Net_Price' in df_features.columns:
                # Simple lag features by product
                df_features[f'price_lag_{lag}'] = df_features.groupby('Product_ID')['Net_Price'].shift(lag)
                
                # Nested lag features by customer-product combination
                df_features[f'price_lag_{lag}_customer_product'] = df_features.groupby(['Customer_ID', 'Product_ID'])['Net_Price'].shift(lag)
        
        # 2. Rolling Window Features (Requirement 23.2)
        window_sizes = self.fe_config.get('temporal', {}).get('window_sizes', [7, 14, 30, 90])
        
        for window in window_sizes:
            if 'Net_Price' in df_features.columns:
                # Moving averages: MA_n = (1/n) × Σ(Price_{t-i})
                df_features[f'price_ma_{window}'] = df_features.groupby('Product_ID')['Net_Price'].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # Exponentially weighted moving averages: EWMA_t = α × Price_t + (1-α) × EWMA_{t-1}
                alpha = 2 / (window + 1)  # Standard EWMA alpha
                df_features[f'price_ewma_{window}'] = df_features.groupby('Product_ID')['Net_Price'].ewm(
                    alpha=alpha, adjust=False
                ).mean().reset_index(0, drop=True)
                
                # Rolling standard deviations
                df_features[f'price_std_{window}'] = df_features.groupby('Product_ID')['Net_Price'].rolling(
                    window=window, min_periods=1
                ).std().reset_index(0, drop=True)
                
                # Price momentum (current vs moving average)
                df_features[f'price_momentum_{window}'] = (
                    df_features['Net_Price'] / df_features[f'price_ma_{window}'] - 1
                ).fillna(0)
        
        # 3. STL Decomposition (Requirement 23.4)
        try:
            # Group by product and perform STL decomposition for products with sufficient data
            def perform_stl_decomposition(group):
                if len(group) >= 24:  # Need at least 2 years of monthly data
                    group = group.set_index('Quote_Date').resample('M')['Net_Price'].mean().dropna()
                    if len(group) >= 12:
                        try:
                            stl = STL(group, seasonal=7, robust=True)
                            result = stl.fit()
                            
                            # Create features from decomposition
                            trend_strength = 1 - np.var(result.resid) / np.var(result.trend + result.resid)
                            seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
                            
                            return pd.Series({
                                'stl_trend_strength': trend_strength,
                                'stl_seasonal_strength': seasonal_strength,
                                'stl_trend_slope': np.polyfit(range(len(result.trend)), result.trend, 1)[0]
                            })
                        except:
                            pass
                
                return pd.Series({
                    'stl_trend_strength': 0,
                    'stl_seasonal_strength': 0,
                    'stl_trend_slope': 0
                })
            
            stl_features = df_features.groupby('Product_ID').apply(perform_stl_decomposition).reset_index()
            df_features = df_features.merge(stl_features, on='Product_ID', how='left')
            
        except Exception as e:
            self.logger.warning(f"STL decomposition failed: {e}")
            df_features['stl_trend_strength'] = 0
            df_features['stl_seasonal_strength'] = 0
            df_features['stl_trend_slope'] = 0
        
        # 4. Advanced Seasonal Features (Requirement 23.3)
        if 'Quote_Date' in df_features.columns:
            # Fourier transform features for cyclical patterns
            for period in [12, 4, 52]:  # Monthly, quarterly, weekly cycles
                df_features[f'fourier_sin_{period}'] = np.sin(2 * np.pi * df_features['Quote_Date'].dt.dayofyear / period)
                df_features[f'fourier_cos_{period}'] = np.cos(2 * np.pi * df_features['Quote_Date'].dt.dayofyear / period)
        
        # 5. Temporal Consistency Checks (Requirement 23.5)
        # Ensure no future data leakage by forward-filling missing lag features
        lag_columns = [col for col in df_features.columns if 'lag_' in col]
        for col in lag_columns:
            df_features[col] = df_features.groupby(['Product_ID', 'Customer_ID'])[col].fillna(method='ffill')
        
        self.logger.info("Created advanced temporal features")
        return df_features
    
    def create_advanced_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced interaction features including customer price sensitivity and competitive features
        Following Requirements 24 and 3.4
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with advanced interaction features
        """
        self.logger.info("Creating advanced interaction features...")
        
        df_features = df.copy()
        
        # 1. Customer Price Sensitivity Analysis (Requirement 24.1)
        if all(col in df_features.columns for col in ['Customer_ID', 'Net_Price', 'Quote_Date']):
            def calculate_customer_price_sensitivity(group):
                if len(group) >= 5:  # Need minimum transactions
                    # Sort by date
                    group = group.sort_values('Quote_Date')
                    
                    # Calculate price changes and volume changes (using count as proxy)
                    group['price_change'] = group['Net_Price'].pct_change()
                    group['volume_proxy'] = 1  # Each quote is a volume unit
                    
                    # Calculate rolling volume (quote frequency)
                    group['volume_change'] = group['volume_proxy'].rolling(window=3, min_periods=1).sum().pct_change()
                    
                    # Calculate correlation between price and volume changes
                    valid_data = group.dropna(subset=['price_change', 'volume_change'])
                    if len(valid_data) >= 3:
                        correlation = valid_data['price_change'].corr(valid_data['volume_change'])
                        premium_tolerance = group['Net_Price'].max() / group['Net_Price'].median() if group['Net_Price'].median() > 0 else 1
                        
                        return pd.Series({
                            'customer_price_sensitivity': -correlation if not np.isnan(correlation) else 0,
                            'premium_tolerance': premium_tolerance,
                            'price_volatility_tolerance': group['Net_Price'].std() / group['Net_Price'].mean() if group['Net_Price'].mean() > 0 else 0
                        })
                
                return pd.Series({
                    'customer_price_sensitivity': 0,
                    'premium_tolerance': 1,
                    'price_volatility_tolerance': 0
                })
            
            customer_sensitivity = df_features.groupby('Customer_ID').apply(calculate_customer_price_sensitivity).reset_index()
            df_features = df_features.merge(customer_sensitivity, on='Customer_ID', how='left')
        
        # 2. Competitive Features (Requirement 24.2)
        if all(col in df_features.columns for col in ['Net_Price', 'Product_Category']):
            # Calculate competitive ratios and market position
            category_stats = df_features.groupby('Product_Category')['Net_Price'].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).add_suffix('_category')
            
            df_features = df_features.merge(category_stats, left_on='Product_Category', right_index=True, how='left')
            
            # Competitive ratio = Our_Price / Category_Average_Price
            df_features['competitive_ratio'] = df_features['Net_Price'] / df_features['mean_category']
            
            # Price position index = normalized position within market range
            df_features['price_position_index'] = (
                (df_features['Net_Price'] - df_features['min_category']) / 
                (df_features['max_category'] - df_features['min_category'])
            ).fillna(0.5).clip(0, 1)
            
            # Market volatility index
            df_features['market_volatility_index'] = df_features['std_category'] / df_features['mean_category']
        
        # 3. Advanced Polynomial Features (Requirement 24.3)
        key_features = ['Net_Price', 'discount_depth', 'customer_tenure_days']
        
        for feature in key_features:
            if feature in df_features.columns:
                # Squared and cubed terms
                df_features[f'{feature}_squared'] = df_features[feature] ** 2
                df_features[f'{feature}_cubed'] = df_features[feature] ** 3
                
                # Log transformation for skewed features
                if feature in ['Net_Price', 'customer_tenure_days']:
                    df_features[f'{feature}_log'] = np.log1p(df_features[feature])
        
        # 4. Cross-product interactions
        interaction_pairs = [
            ('Net_Price', 'discount_depth'),
            ('customer_price_sensitivity', 'Net_Price'),
            ('competitive_ratio', 'discount_depth'),
            ('premium_tolerance', 'Net_Price')
        ]
        
        for feature1, feature2 in interaction_pairs:
            if all(col in df_features.columns for col in [feature1, feature2]):
                df_features[f'{feature1}_x_{feature2}'] = df_features[feature1] * df_features[feature2]
        
        # 5. Price-customer segment interactions
        if all(col in df_features.columns for col in ['Net_Price', 'Customer_Segment']):
            for segment in df_features['Customer_Segment'].unique():
                if pd.notna(segment):
                    df_features[f'price_x_{segment.lower().replace("-", "_")}_segment'] = (
                        df_features['Net_Price'] * (df_features['Customer_Segment'] == segment).astype(int)
                    )
        
        self.logger.info("Created advanced interaction features")
        return df_features
    
    def create_b2b_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create B2B domain-specific features including contract structures and supply chain factors
        Following Requirement 25
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with B2B domain features
        """
        self.logger.info("Creating B2B domain-specific features...")
        
        df_features = df.copy()
        
        # 1. Contract Features (Requirement 25.1)
        if 'Customer_ID' in df_features.columns:
            # Calculate customer relationship metrics as proxy for contract features
            customer_metrics = df_features.groupby('Customer_ID').agg({
                'Quote_Date': ['min', 'max', 'count'],
                'Net_Price': ['mean', 'sum']
            })
            
            customer_metrics.columns = ['first_quote_date', 'last_quote_date', 'total_quotes', 'avg_deal_size', 'total_value']
            
            # Contract length factor (relationship duration)
            customer_metrics['relationship_duration_days'] = (
                customer_metrics['last_quote_date'] - customer_metrics['first_quote_date']
            ).dt.days
            
            # Average contract duration (using quote frequency as proxy)
            avg_relationship_duration = customer_metrics['relationship_duration_days'].mean()
            customer_metrics['contract_length_factor'] = (
                customer_metrics['relationship_duration_days'] / avg_relationship_duration
            ).fillna(1.0)
            
            # Deal structure indicators
            customer_metrics['deal_frequency'] = (
                customer_metrics['total_quotes'] / (customer_metrics['relationship_duration_days'] + 1) * 365
            )
            customer_metrics['deal_size_consistency'] = (
                df_features.groupby('Customer_ID')['Net_Price'].std() / 
                df_features.groupby('Customer_ID')['Net_Price'].mean()
            ).fillna(0)
            
            # Merge back to main dataframe
            df_features = df_features.merge(
                customer_metrics[['contract_length_factor', 'deal_frequency', 'deal_size_consistency']], 
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
        
        # 2. Supply Chain Features (Requirement 25.2)
        if all(col in df_features.columns for col in ['Product_ID', 'Net_Price', 'Quote_Date']):
            # Calculate product-level supply chain proxies
            product_metrics = df_features.groupby('Product_ID').agg({
                'Net_Price': ['mean', 'std', 'count'],
                'Quote_Date': ['min', 'max']
            })
            
            product_metrics.columns = ['avg_price', 'price_volatility', 'demand_frequency', 'first_quote', 'last_quote']
            
            # Inventory turnover proxy (using demand frequency and price volatility)
            product_metrics['inventory_turnover_proxy'] = (
                product_metrics['demand_frequency'] / (product_metrics['price_volatility'] + 1)
            )
            
            # Stock-out risk score (based on demand variability)
            product_metrics['stock_out_risk_score'] = (
                product_metrics['price_volatility'] / product_metrics['avg_price']
            ).fillna(0).clip(0, 1)
            
            # Supply chain disruption indicator (price volatility spikes)
            product_metrics['supply_disruption_indicator'] = (
                product_metrics['price_volatility'] > product_metrics['price_volatility'].quantile(0.8)
            ).astype(int)
            
            # Merge back to main dataframe
            df_features = df_features.merge(
                product_metrics[['inventory_turnover_proxy', 'stock_out_risk_score', 'supply_disruption_indicator']], 
                left_on='Product_ID', 
                right_index=True, 
                how='left'
            )
        
        # 3. Economic Indicators (Requirement 25.3)
        if 'Quote_Date' in df_features.columns:
            # Create time-based economic proxies
            df_features['year'] = df_features['Quote_Date'].dt.year
            df_features['quarter'] = df_features['Quote_Date'].dt.quarter
            
            # Market volatility as standard deviation of prices over time
            quarterly_volatility = df_features.groupby(['year', 'quarter'])['Net_Price'].std().reset_index()
            quarterly_volatility.columns = ['year', 'quarter', 'market_volatility']
            
            df_features = df_features.merge(quarterly_volatility, on=['year', 'quarter'], how='left')
            df_features['market_volatility'] = df_features['market_volatility'].fillna(df_features['market_volatility'].mean())
            
            # Seasonal demand patterns
            monthly_demand = df_features.groupby(df_features['Quote_Date'].dt.month)['Quote_ID'].count()
            monthly_demand_norm = (monthly_demand - monthly_demand.mean()) / monthly_demand.std()
            
            df_features['seasonal_demand_index'] = df_features['Quote_Date'].dt.month.map(monthly_demand_norm.to_dict())
        
        # 4. Business Impact Features (Requirement 25.4)
        if all(col in df_features.columns for col in ['Customer_ID', 'Net_Price']):
            # Customer acquisition cost proxy (inverse of deal frequency)
            df_features['customer_acquisition_cost_proxy'] = 1 / (df_features['deal_frequency'] + 0.1)
            
            # Customer lifetime value multiples
            if 'clv_estimate' in df_features.columns:
                df_features['clv_multiple'] = df_features['clv_estimate'] / df_features['Net_Price']
            else:
                # Simple CLV proxy
                customer_value = df_features.groupby('Customer_ID')['Net_Price'].sum()
                df_features['clv_multiple'] = df_features['Customer_ID'].map(customer_value) / df_features['Net_Price']
        
        # 5. Market Context Features (Requirement 25.5)
        if 'Quote_Date' in df_features.columns:
            # Rolling market volatility
            df_features = df_features.sort_values('Quote_Date')
            
            for window in [30, 90, 180]:
                df_features[f'market_volatility_{window}d'] = df_features['Net_Price'].rolling(
                    window=window, min_periods=1
                ).std()
            
            # Industry-specific cyclical indicators (using product category as proxy)
            if 'Product_Category' in df_features.columns:
                category_cycles = df_features.groupby(['Product_Category', df_features['Quote_Date'].dt.month])['Net_Price'].mean().reset_index()
                category_cycles['month'] = category_cycles['Quote_Date']
                category_cycles = category_cycles.pivot(index='month', columns='Product_Category', values='Net_Price')
                
                # Calculate cyclical strength for each category
                for category in category_cycles.columns:
                    if category_cycles[category].notna().sum() >= 6:  # Need at least 6 months of data
                        cyclical_strength = category_cycles[category].std() / category_cycles[category].mean()
                        df_features.loc[df_features['Product_Category'] == category, 'category_cyclical_strength'] = cyclical_strength
                
                df_features['category_cyclical_strength'] = df_features['category_cyclical_strength'].fillna(0)
        
        self.logger.info("Created B2B domain-specific features")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic interaction and polynomial features (legacy method)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating basic interaction features...")
        
        # Call the advanced interaction features method
        return self.create_advanced_interaction_features(df)
    
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
    
    def create_graph_neural_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Graph Neural Network features including customer-product networks and embeddings
        Following Requirement 9 and 12.5
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with GNN-based features
        """
        self.logger.info("Creating Graph Neural Network features...")
        
        df_features = df.copy()
        
        try:
            # 1. Create bipartite customer-product graphs (Requirement 9.1)
            if all(col in df_features.columns for col in ['Customer_ID', 'Product_ID', 'Net_Price']):
                # Build customer-product interaction network
                G = nx.Graph()
                
                # Add customer nodes
                customers = df_features['Customer_ID'].unique()
                G.add_nodes_from([(f"C_{c}", {"type": "customer"}) for c in customers])
                
                # Add product nodes
                products = df_features['Product_ID'].unique()
                G.add_nodes_from([(f"P_{p}", {"type": "product"}) for p in products])
                
                # Add edges with weights (transaction value)
                for _, row in df_features.iterrows():
                    customer_node = f"C_{row['Customer_ID']}"
                    product_node = f"P_{row['Product_ID']}"
                    weight = row['Net_Price']
                    
                    if G.has_edge(customer_node, product_node):
                        G[customer_node][product_node]['weight'] += weight
                        G[customer_node][product_node]['count'] += 1
                    else:
                        G.add_edge(customer_node, product_node, weight=weight, count=1)
                
                # 2. Calculate node centrality measures (Requirement 9.4)
                try:
                    # Degree centrality
                    degree_centrality = nx.degree_centrality(G)
                    
                    # Betweenness centrality (for smaller graphs)
                    if len(G.nodes()) < 1000:
                        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
                    else:
                        betweenness_centrality = {node: 0 for node in G.nodes()}
                    
                    # Eigenvector centrality
                    try:
                        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
                    except:
                        eigenvector_centrality = {node: 0 for node in G.nodes()}
                    
                    # Map centrality measures back to dataframe
                    df_features['customer_degree_centrality'] = df_features['Customer_ID'].apply(
                        lambda x: degree_centrality.get(f"C_{x}", 0)
                    )
                    df_features['customer_betweenness_centrality'] = df_features['Customer_ID'].apply(
                        lambda x: betweenness_centrality.get(f"C_{x}", 0)
                    )
                    df_features['customer_eigenvector_centrality'] = df_features['Customer_ID'].apply(
                        lambda x: eigenvector_centrality.get(f"C_{x}", 0)
                    )
                    
                    df_features['product_degree_centrality'] = df_features['Product_ID'].apply(
                        lambda x: degree_centrality.get(f"P_{x}", 0)
                    )
                    df_features['product_betweenness_centrality'] = df_features['Product_ID'].apply(
                        lambda x: betweenness_centrality.get(f"P_{x}", 0)
                    )
                    df_features['product_eigenvector_centrality'] = df_features['Product_ID'].apply(
                        lambda x: eigenvector_centrality.get(f"P_{x}", 0)
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Centrality calculation failed: {e}")
                    # Set default values
                    for centrality_type in ['degree', 'betweenness', 'eigenvector']:
                        df_features[f'customer_{centrality_type}_centrality'] = 0
                        df_features[f'product_{centrality_type}_centrality'] = 0
                
                # 3. Generate graph-based similarity scores (Requirement 9.4)
                # Customer similarity based on product preferences
                customer_product_matrix = df_features.pivot_table(
                    index='Customer_ID', 
                    columns='Product_ID', 
                    values='Net_Price', 
                    aggfunc='mean'
                ).fillna(0)
                
                if customer_product_matrix.shape[0] > 1 and customer_product_matrix.shape[1] > 1:
                    # Calculate cosine similarity between customers
                    customer_similarity = cosine_similarity(customer_product_matrix)
                    
                    # For each customer, find their average similarity to others
                    customer_avg_similarity = {}
                    for i, customer in enumerate(customer_product_matrix.index):
                        similarities = customer_similarity[i]
                        # Exclude self-similarity
                        other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
                        customer_avg_similarity[customer] = np.mean(other_similarities) if len(other_similarities) > 0 else 0
                    
                    df_features['customer_similarity_score'] = df_features['Customer_ID'].map(customer_avg_similarity).fillna(0)
                else:
                    df_features['customer_similarity_score'] = 0
                
                # Product similarity based on customer base
                product_customer_matrix = df_features.pivot_table(
                    index='Product_ID', 
                    columns='Customer_ID', 
                    values='Net_Price', 
                    aggfunc='mean'
                ).fillna(0)
                
                if product_customer_matrix.shape[0] > 1 and product_customer_matrix.shape[1] > 1:
                    product_similarity = cosine_similarity(product_customer_matrix)
                    
                    product_avg_similarity = {}
                    for i, product in enumerate(product_customer_matrix.index):
                        similarities = product_similarity[i]
                        other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
                        product_avg_similarity[product] = np.mean(other_similarities) if len(other_similarities) > 0 else 0
                    
                    df_features['product_similarity_score'] = df_features['Product_ID'].map(product_avg_similarity).fillna(0)
                else:
                    df_features['product_similarity_score'] = 0
                
                # 4. Network effect features (Requirement 9.3)
                # Calculate network density around each customer and product
                customer_network_density = {}
                product_network_density = {}
                
                for customer in customers:
                    customer_node = f"C_{customer}"
                    if customer_node in G:
                        neighbors = list(G.neighbors(customer_node))
                        if len(neighbors) > 1:
                            # Create subgraph of customer's neighborhood
                            subgraph = G.subgraph([customer_node] + neighbors)
                            density = nx.density(subgraph)
                            customer_network_density[customer] = density
                        else:
                            customer_network_density[customer] = 0
                    else:
                        customer_network_density[customer] = 0
                
                for product in products:
                    product_node = f"P_{product}"
                    if product_node in G:
                        neighbors = list(G.neighbors(product_node))
                        if len(neighbors) > 1:
                            subgraph = G.subgraph([product_node] + neighbors)
                            density = nx.density(subgraph)
                            product_network_density[product] = density
                        else:
                            product_network_density[product] = 0
                    else:
                        product_network_density[product] = 0
                
                df_features['customer_network_density'] = df_features['Customer_ID'].map(customer_network_density).fillna(0)
                df_features['product_network_density'] = df_features['Product_ID'].map(product_network_density).fillna(0)
                
                # 5. Graph embeddings using Word2Vec-style approach (Requirement 12.5)
                # Create random walks for embedding generation
                def generate_random_walks(graph, num_walks=10, walk_length=5):
                    walks = []
                    nodes = list(graph.nodes())
                    
                    for _ in range(num_walks):
                        for node in nodes:
                            walk = [node]
                            current_node = node
                            
                            for _ in range(walk_length - 1):
                                neighbors = list(graph.neighbors(current_node))
                                if neighbors:
                                    # Weighted random selection based on edge weights
                                    weights = [graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
                                    total_weight = sum(weights)
                                    if total_weight > 0:
                                        probabilities = [w / total_weight for w in weights]
                                        current_node = np.random.choice(neighbors, p=probabilities)
                                    else:
                                        current_node = np.random.choice(neighbors)
                                    walk.append(current_node)
                                else:
                                    break
                            
                            walks.append(walk)
                    
                    return walks
                
                # Generate walks and create simple embeddings
                if len(G.nodes()) > 0:
                    walks = generate_random_walks(G, num_walks=5, walk_length=3)
                    
                    # Create co-occurrence matrix for simple embedding
                    node_cooccurrence = {}
                    for walk in walks:
                        for i, node in enumerate(walk):
                            if node not in node_cooccurrence:
                                node_cooccurrence[node] = {}
                            
                            # Look at context window
                            for j in range(max(0, i-1), min(len(walk), i+2)):
                                if i != j:
                                    context_node = walk[j]
                                    if context_node not in node_cooccurrence[node]:
                                        node_cooccurrence[node][context_node] = 0
                                    node_cooccurrence[node][context_node] += 1
                    
                    # Create simple embedding features (sum of co-occurrence counts)
                    customer_embedding_strength = {}
                    product_embedding_strength = {}
                    
                    for node, cooccur in node_cooccurrence.items():
                        strength = sum(cooccur.values())
                        if node.startswith('C_'):
                            customer_id = node[2:]  # Remove 'C_' prefix
                            customer_embedding_strength[customer_id] = strength
                        elif node.startswith('P_'):
                            product_id = node[2:]  # Remove 'P_' prefix
                            product_embedding_strength[product_id] = strength
                    
                    df_features['customer_embedding_strength'] = df_features['Customer_ID'].map(customer_embedding_strength).fillna(0)
                    df_features['product_embedding_strength'] = df_features['Product_ID'].map(product_embedding_strength).fillna(0)
                else:
                    df_features['customer_embedding_strength'] = 0
                    df_features['product_embedding_strength'] = 0
                
        except Exception as e:
            self.logger.warning(f"Graph neural network feature creation failed: {e}")
            # Set default values for all GNN features
            gnn_features = [
                'customer_degree_centrality', 'customer_betweenness_centrality', 'customer_eigenvector_centrality',
                'product_degree_centrality', 'product_betweenness_centrality', 'product_eigenvector_centrality',
                'customer_similarity_score', 'product_similarity_score',
                'customer_network_density', 'product_network_density',
                'customer_embedding_strength', 'product_embedding_strength'
            ]
            
            for feature in gnn_features:
                df_features[feature] = 0
        
        self.logger.info("Created Graph Neural Network features")
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
        
        # Step 2.5: Create advanced feature groups
        df_features = self.create_advanced_temporal_features(df_features)
        df_features = self.create_b2b_domain_features(df_features)
        df_features = self.create_graph_neural_network_features(df_features)
        
        # Step 3: Create interaction features (now includes advanced interactions)
        df_features = self.create_interaction_features(df_features)
        
        # Step 4: Encode categorical features
        df_features = self.encode_categorical_features(df_features, fit=fit)
        
        # Step 4.5: Handle datetime columns
        df_features = self._handle_datetime_columns(df_features)
        
        # Step 4.75: Handle infinite values before scaling
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        
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

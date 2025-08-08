#!/usr/bin/env python3
"""
Price Elasticity Model Inference & Analytics Dashboard
=====================================================

Interactive Streamlit application for:
- Model inference and predictions
- Comprehensive EDA visualizations
- Real-time analytics and insights
- Model performance monitoring
- Price sensitivity analysis

Usage:
    streamlit run inference.py
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import yaml

# Advanced UI components
try:
    from streamlit_option_menu import option_menu
    from streamlit_plotly_events import plotly_events
    from streamlit_aggrid import AgGrid, GridOptionsBuilder, JsCode
    HAS_ADVANCED_UI = True
except ImportError:
    HAS_ADVANCED_UI = False

# Import our modules
try:
    from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
    from training.model_training import PriceElasticityModelTraining
    from utils.config_loader import config_loader
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Price Elasticity Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


class PriceElasticityDashboard:
    """
    Comprehensive dashboard for price elasticity analysis and model inference
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.config = config_loader
        self.feature_engineer = None
        self.models = {}
        self.training_results = {}
        self.sample_data = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load models and artifacts
        self._load_artifacts()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'ensemble'
    
    def _load_artifacts(self):
        """Load trained models and feature engineering artifacts"""
        try:
            # Load feature engineering artifacts
            fe_path = Path("models/feature_engineering")
            if fe_path.exists():
                self.feature_engineer = PriceElasticityFeatureEngineering()
                self.feature_engineer.load_feature_engineering_artifacts()
                st.session_state.fe_loaded = True
            
            # Load trained models
            models_path = Path("models/trained")
            if models_path.exists():
                for model_file in models_path.glob("*_model.pkl"):
                    model_name = model_file.stem.replace('_model', '')
                    try:
                        self.models[model_name] = joblib.load(model_file)
                    except Exception as e:
                        st.warning(f"Could not load {model_name}: {e}")
                
                # Load training results
                results_file = models_path / 'training_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.training_results = json.load(f)
                
                st.session_state.models_loaded = len(self.models) > 0
            
        except Exception as e:
            st.warning(f"Could not load all artifacts: {e}")
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🎯 Price Elasticity Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Available", len(self.models), help="Number of trained models ready for inference")
        
        with col2:
            status = "✅ Ready" if st.session_state.models_loaded else "⚠️ Not Ready"
            st.metric("System Status", status, help="Overall system readiness")
        
        with col3:
            data_status = "✅ Loaded" if st.session_state.data_loaded else "📥 Load Data"
            st.metric("Data Status", data_status, help="Data loading status")
        
        with col4:
            st.metric("Features Engineered", 
                     len(self.feature_engineer.feature_metadata.get('final_columns', [])) if self.feature_engineer else 0,
                     help="Number of features available for modeling")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## 🔧 Control Panel")
        
        # Navigation
        if HAS_ADVANCED_UI:
            page = option_menu(
                "Navigation",
                ["📊 EDA & Insights", "🤖 Model Inference", "📈 Performance Analytics", "🔍 Training Monitor"],
                icons=['bar-chart', 'cpu', 'graph-up', 'search'],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
            )
        else:
            page = st.sidebar.selectbox(
                "Select Page",
                ["📊 EDA & Insights", "🤖 Model Inference", "📈 Performance Analytics", "🔍 Training Monitor"]
            )
        
        st.sidebar.markdown("---")
        
        # Data upload
        st.sidebar.markdown("### 📁 Data Management")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your data", 
            type=['csv', 'xlsx', 'parquet'],
            help="Upload your B2B pricing data for analysis"
        )
        
        # Sample data option
        if st.sidebar.button("🎲 Generate Sample Data", help="Create sample data for demonstration"):
            self.sample_data = self._generate_sample_data()
            st.session_state.data_loaded = True
            st.success("Sample data generated successfully!")
        
        # Model selection
        if self.models:
            st.sidebar.markdown("### 🤖 Model Selection")
            st.session_state.selected_model = st.sidebar.selectbox(
                "Choose Model",
                list(self.models.keys()),
                index=list(self.models.keys()).index(st.session_state.selected_model) 
                if st.session_state.selected_model in self.models else 0,
                help="Select which trained model to use for inference"
            )
        
        # Settings
        st.sidebar.markdown("### ⚙️ Settings")
        show_advanced = st.sidebar.checkbox("Show Advanced Options", False)
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", False)
        
        if auto_refresh:
            st.rerun()
        
        return page, uploaded_file, show_advanced
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample B2B pricing data"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic B2B pricing data
        data = {
            'Quote_ID': [f'Q{i:06d}' for i in range(n_samples)],
            'Customer_ID': [f'C{i%500:04d}' for i in range(n_samples)],
            'Product_ID': [f'P{i%100:03d}' for i in range(n_samples)],
            'Quote_Date': pd.date_range(start='2020-01-01', end='2024-01-01', periods=n_samples),
            'Customer_Since_Date': pd.date_range(start='2018-01-01', end='2023-01-01', periods=n_samples),
            'Launch_Date': pd.date_range(start='2019-01-01', end='2023-06-01', periods=n_samples),
            
            # Price features
            'List_Price': np.random.lognormal(mean=8, sigma=0.5, size=n_samples),
            'Net_Price': np.random.lognormal(mean=7.8, sigma=0.5, size=n_samples),
            'Offered_Price': np.random.lognormal(mean=7.7, sigma=0.5, size=n_samples),
            
            # Categorical features
            'Customer_Segment': np.random.choice(['Enterprise', 'Mid-Market', 'SMB'], n_samples, p=[0.3, 0.4, 0.3]),
            'Product_Category': np.random.choice(['Software', 'Hardware', 'Services'], n_samples, p=[0.5, 0.3, 0.2]),
            'Competition_Status': np.random.choice(['None', 'Low', 'Medium', 'High'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'Product_Objective': np.random.choice(['Growth', 'Profitability', 'Market_Share'], n_samples, p=[0.4, 0.4, 0.2]),
            'Lifecycle_Stage': np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'], n_samples, p=[0.2, 0.3, 0.4, 0.1]),
            
            # Target variable (Win/Loss)
            'Status': np.random.choice(['Won', 'Lost'], n_samples, p=[0.35, 0.65])
        }
        
        df = pd.DataFrame(data)
        
        # Add logical relationships
        discount_depth = (df['List_Price'] - df['Net_Price']) / df['List_Price']
        win_prob = 0.2 + 0.3 * discount_depth + 0.1 * np.random.random(n_samples)
        df.loc[np.random.random(n_samples) < win_prob, 'Status'] = 'Won'
        
        return df
    
    def render_eda_page(self):
        """Render comprehensive EDA page"""
        st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded and self.sample_data is None:
            st.info("👆 Please upload data or generate sample data using the sidebar.")
            return
        
        df = self.sample_data if self.sample_data is not None else pd.DataFrame()
        
        if df.empty:
            st.error("No data available for analysis.")
            return
        
        # Data overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Dataset Overview")
            
            # Dataset statistics
            st.write(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.write(f"**Date Range:** {df['Quote_Date'].min().strftime('%Y-%m-%d')} to {df['Quote_Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Win Rate:** {(df['Status'] == 'Won').mean():.1%}")
            
            # Display sample data
            with st.expander("View Sample Data"):
                st.dataframe(df.head(10))
        
        with col2:
            st.subheader("🎯 Key Metrics")
            
            avg_list_price = df['List_Price'].mean()
            avg_net_price = df['Net_Price'].mean()
            avg_discount = ((df['List_Price'] - df['Net_Price']) / df['List_Price']).mean()
            
            st.metric("Average List Price", f"${avg_list_price:,.0f}")
            st.metric("Average Net Price", f"${avg_net_price:,.0f}")
            st.metric("Average Discount", f"{avg_discount:.1%}")
        
        # Visualizations
        st.subheader("📊 Interactive Visualizations")
        
        # Create tabs for different analysis
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Price Analysis", "🎯 Win Rate Analysis", "👥 Customer Insights", "📅 Temporal Trends"])
        
        with tab1:
            self._render_price_analysis(df)
        
        with tab2:
            self._render_win_rate_analysis(df)
        
        with tab3:
            self._render_customer_analysis(df)
        
        with tab4:
            self._render_temporal_analysis(df)
    
    def _render_price_analysis(self, df: pd.DataFrame):
        """Render price analysis visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(
                df, x='Net_Price', nbins=50,
                title='Net Price Distribution',
                labels={'Net_Price': 'Net Price ($)', 'count': 'Frequency'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Discount vs Win Rate
            df['Discount_Depth'] = (df['List_Price'] - df['Net_Price']) / df['List_Price']
            df['Discount_Bin'] = pd.cut(df['Discount_Depth'], bins=10)
            
            discount_win_rate = df.groupby('Discount_Bin')['Status'].apply(lambda x: (x == 'Won').mean()).reset_index()
            discount_win_rate['Discount_Mid'] = discount_win_rate['Discount_Bin'].apply(lambda x: x.mid)
            
            fig = px.line(
                discount_win_rate, x='Discount_Mid', y='Status',
                title='Win Rate vs Discount Depth',
                labels={'Discount_Mid': 'Discount Depth', 'Status': 'Win Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price by category
        fig = px.box(
            df, x='Product_Category', y='Net_Price',
            title='Price Distribution by Product Category',
            labels={'Net_Price': 'Net Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_win_rate_analysis(self, df: pd.DataFrame):
        """Render win rate analysis visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Win rate by segment
            segment_win_rate = df.groupby('Customer_Segment')['Status'].apply(lambda x: (x == 'Won').mean()).reset_index()
            
            fig = px.bar(
                segment_win_rate, x='Customer_Segment', y='Status',
                title='Win Rate by Customer Segment',
                labels={'Status': 'Win Rate'},
                color='Status'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Win rate by competition
            comp_win_rate = df.groupby('Competition_Status')['Status'].apply(lambda x: (x == 'Won').mean()).reset_index()
            
            fig = px.bar(
                comp_win_rate, x='Competition_Status', y='Status',
                title='Win Rate by Competition Status',
                labels={'Status': 'Win Rate'},
                color='Status'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Win rate heatmap
        pivot_data = df.pivot_table(
            values='Status', 
            index='Customer_Segment', 
            columns='Product_Category',
            aggfunc=lambda x: (x == 'Won').mean()
        )
        
        fig = px.imshow(
            pivot_data, 
            title='Win Rate Heatmap: Customer Segment vs Product Category',
            aspect='auto',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_customer_analysis(self, df: pd.DataFrame):
        """Render customer analysis visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segment distribution
            segment_counts = df['Customer_Segment'].value_counts()
            
            fig = px.pie(
                values=segment_counts.values, 
                names=segment_counts.index,
                title='Customer Segment Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average deal size by segment
            avg_deal_size = df.groupby('Customer_Segment')['Net_Price'].mean().reset_index()
            
            fig = px.bar(
                avg_deal_size, x='Customer_Segment', y='Net_Price',
                title='Average Deal Size by Segment',
                labels={'Net_Price': 'Average Net Price ($)'},
                color='Net_Price'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer performance over time
        monthly_metrics = df.set_index('Quote_Date').resample('M').agg({
            'Status': lambda x: (x == 'Won').mean(),
            'Net_Price': 'mean',
            'Quote_ID': 'count'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Win Rate', 'Monthly Quote Volume'),
            shared_xaxes=True
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_metrics['Quote_Date'], y=monthly_metrics['Status'], 
                      mode='lines+markers', name='Win Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_metrics['Quote_Date'], y=monthly_metrics['Quote_ID'], 
                      mode='lines+markers', name='Quote Volume', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=500, title_text="Customer Performance Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_temporal_analysis(self, df: pd.DataFrame):
        """Render temporal analysis visualizations"""
        # Time series analysis
        df['YearMonth'] = df['Quote_Date'].dt.to_period('M')
        monthly_trends = df.groupby('YearMonth').agg({
            'Status': lambda x: (x == 'Won').mean(),
            'Net_Price': 'mean',
            'Quote_ID': 'count'
        }).reset_index()
        monthly_trends['YearMonth'] = monthly_trends['YearMonth'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal patterns
            df['Month'] = df['Quote_Date'].dt.month
            seasonal_win_rate = df.groupby('Month')['Status'].apply(lambda x: (x == 'Won').mean()).reset_index()
            
            fig = px.line(
                seasonal_win_rate, x='Month', y='Status',
                title='Seasonal Win Rate Pattern',
                labels={'Status': 'Win Rate', 'Month': 'Month of Year'}
            )
            fig.update_xaxis(tickmode='linear', tick0=1, dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week patterns
            df['DayOfWeek'] = df['Quote_Date'].dt.day_name()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_win_rate = df.groupby('DayOfWeek')['Status'].apply(lambda x: (x == 'Won').mean()).reindex(dow_order).reset_index()
            
            fig = px.bar(
                dow_win_rate, x='DayOfWeek', y='Status',
                title='Win Rate by Day of Week',
                labels={'Status': 'Win Rate', 'DayOfWeek': 'Day of Week'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Long-term trends
        fig = px.line(
            monthly_trends, x='YearMonth', y='Status',
            title='Win Rate Trend Over Time',
            labels={'Status': 'Win Rate', 'YearMonth': 'Month-Year'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_inference_page(self):
        """Render model inference page"""
        st.markdown('<h2 class="sub-header">🤖 Model Inference & Predictions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ No trained models found. Please run the training pipeline first.")
            st.code("python model_training.py")
            return
        
        # Model selection and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Selected Model: {st.session_state.selected_model}")
            
            # Model performance info
            if st.session_state.selected_model in self.training_results:
                perf = self.training_results[st.session_state.selected_model].get('performance', {})
                
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.metric("AUC Score", f"{perf.get('auc', 0):.3f}")
                with col1b:
                    st.metric("Accuracy", f"{perf.get('accuracy', 0):.3f}")
                with col1c:
                    st.metric("F1 Score", f"{perf.get('f1', 0):.3f}")
        
        with col2:
            st.subheader("Inference Options")
            inference_mode = st.radio(
                "Choose Mode",
                ["Single Prediction", "Batch Prediction", "What-If Analysis"]
            )
        
        # Render inference interface based on mode
        if inference_mode == "Single Prediction":
            self._render_single_prediction()
        elif inference_mode == "Batch Prediction":
            self._render_batch_prediction()
        else:
            self._render_whatif_analysis()
    
    def _render_single_prediction(self):
        """Render single prediction interface"""
        st.subheader("🎯 Single Quote Prediction")
        
        # Input form
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Quote Information**")
                list_price = st.number_input("List Price ($)", min_value=0.0, value=10000.0)
                net_price = st.number_input("Net Price ($)", min_value=0.0, value=8500.0)
                offered_price = st.number_input("Offered Price ($)", min_value=0.0, value=8000.0)
            
            with col2:
                st.markdown("**Customer Information**")
                customer_segment = st.selectbox("Customer Segment", ['Enterprise', 'Mid-Market', 'SMB'])
                customer_tenure = st.slider("Customer Tenure (days)", 0, 2000, 365)
            
            with col3:
                st.markdown("**Product Information**")
                product_category = st.selectbox("Product Category", ['Software', 'Hardware', 'Services'])
                competition_status = st.selectbox("Competition Status", ['None', 'Low', 'Medium', 'High'])
                lifecycle_stage = st.selectbox("Lifecycle Stage", ['Introduction', 'Growth', 'Maturity', 'Decline'])
            
            submitted = st.form_submit_button("🔮 Predict Win Probability")
            
            if submitted:
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'List_Price': [list_price],
                    'Net_Price': [net_price],
                    'Offered_Price': [offered_price],
                    'Customer_Segment': [customer_segment],
                    'Product_Category': [product_category],
                    'Competition_Status': [competition_status],
                    'Lifecycle_Stage': [lifecycle_stage],
                    'Quote_Date': [datetime.now()],
                    'Customer_Since_Date': [datetime.now() - timedelta(days=customer_tenure)],
                    'Launch_Date': [datetime.now() - timedelta(days=500)],
                    # Add dummy IDs
                    'Quote_ID': ['PRED_001'],
                    'Customer_ID': ['CUST_PRED'],
                    'Product_ID': ['PROD_PRED'],
                    'Product_Objective': ['Profitability']
                })
                
                # Make prediction
                prediction = self._make_prediction(input_data)
                
                if prediction is not None:
                    prob = prediction[0]
                    
                    # Display results
                    st.success("✅ Prediction completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Win Probability", f"{prob:.1%}")
                    with col2:
                        prediction_text = "🎉 Likely to WIN" if prob > 0.5 else "❌ Likely to LOSE"
                        st.metric("Prediction", prediction_text)
                    with col3:
                        confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.1 else "Low"
                        st.metric("Confidence", confidence)
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Win Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_batch_prediction(self):
        """Render batch prediction interface"""
        st.subheader("📋 Batch Predictions")
        
        if not st.session_state.data_loaded and self.sample_data is None:
            st.info("Please load data first using the sidebar.")
            return
        
        df = self.sample_data if self.sample_data is not None else pd.DataFrame()
        
        if df.empty:
            st.error("No data available for batch prediction.")
            return
        
        # Batch prediction options
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Batch Options**")
            sample_size = st.slider("Sample Size", 10, min(1000, len(df)), 100)
            include_confidence = st.checkbox("Include Confidence Intervals", True)
            
            if st.button("🚀 Run Batch Predictions"):
                # Sample data
                sample_df = df.sample(n=sample_size, random_state=42)
                
                # Make predictions
                with st.spinner("Making predictions..."):
                    predictions = self._make_prediction(sample_df)
                
                if predictions is not None:
                    # Add predictions to dataframe
                    sample_df = sample_df.copy()
                    sample_df['Predicted_Win_Prob'] = predictions
                    sample_df['Predicted_Status'] = (predictions > 0.5).map({True: 'Won', False: 'Lost'})
                    sample_df['Actual_Status'] = sample_df['Status']
                    
                    # Store in session state
                    st.session_state.predictions = sample_df
                    st.success(f"✅ Completed predictions for {len(sample_df)} quotes")
        
        with col2:
            if st.session_state.predictions is not None:
                pred_df = st.session_state.predictions
                
                # Performance metrics
                accuracy = (pred_df['Predicted_Status'] == pred_df['Actual_Status']).mean()
                st.metric("Batch Accuracy", f"{accuracy:.1%}")
                
                # Show predictions
                display_cols = ['Quote_ID', 'Customer_Segment', 'Product_Category', 
                              'Net_Price', 'Predicted_Win_Prob', 'Predicted_Status', 'Actual_Status']
                st.dataframe(pred_df[display_cols].head(20))
                
                # Download option
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "batch_predictions.csv",
                    "text/csv"
                )
    
    def _render_whatif_analysis(self):
        """Render what-if analysis interface"""
        st.subheader("🔍 What-If Scenario Analysis")
        
        st.markdown("Analyze how different pricing strategies affect win probability.")
        
        # Base scenario input
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Base Scenario**")
            base_price = st.number_input("Base Net Price ($)", min_value=0.0, value=10000.0, key="base")
            customer_seg = st.selectbox("Customer Segment", ['Enterprise', 'Mid-Market', 'SMB'], key="base_seg")
            product_cat = st.selectbox("Product Category", ['Software', 'Hardware', 'Services'], key="base_cat")
        
        with col2:
            st.markdown("**Analysis Parameters**")
            price_range = st.slider("Price Range (% of base)", -50, 50, (-20, 20), key="price_range")
            num_scenarios = st.slider("Number of Scenarios", 5, 20, 10)
        
        if st.button("🔬 Run What-If Analysis"):
            # Generate price scenarios
            min_price = base_price * (1 + price_range[0]/100)
            max_price = base_price * (1 + price_range[1]/100)
            price_scenarios = np.linspace(min_price, max_price, num_scenarios)
            
            # Create scenario data
            scenario_data = []
            for price in price_scenarios:
                scenario = pd.DataFrame({
                    'List_Price': [price * 1.2],  # Assume 20% markup
                    'Net_Price': [price],
                    'Offered_Price': [price],
                    'Customer_Segment': [customer_seg],
                    'Product_Category': [product_cat],
                    'Competition_Status': ['Medium'],
                    'Lifecycle_Stage': ['Maturity'],
                    'Quote_Date': [datetime.now()],
                    'Customer_Since_Date': [datetime.now() - timedelta(days=365)],
                    'Launch_Date': [datetime.now() - timedelta(days=500)],
                    'Quote_ID': [f'SCENARIO_{len(scenario_data)}'],
                    'Customer_ID': ['SCENARIO_CUST'],
                    'Product_ID': ['SCENARIO_PROD'],
                    'Product_Objective': ['Profitability']
                })
                scenario_data.append(scenario)
            
            # Combine scenarios
            all_scenarios = pd.concat(scenario_data, ignore_index=True)
            
            # Make predictions
            with st.spinner("Analyzing scenarios..."):
                predictions = self._make_prediction(all_scenarios)
            
            if predictions is not None:
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Price': price_scenarios,
                    'Win_Probability': predictions,
                    'Price_Change': ((price_scenarios - base_price) / base_price * 100)
                })
                
                # Visualization
                fig = px.line(
                    results_df, x='Price', y='Win_Probability',
                    title='Win Probability vs Price',
                    labels={'Price': 'Net Price ($)', 'Win_Probability': 'Win Probability'}
                )
                fig.add_vline(x=base_price, line_dash="dash", line_color="red", 
                             annotation_text="Base Price")
                st.plotly_chart(fig, use_container_width=True)
                
                # Optimal pricing suggestion
                optimal_idx = results_df['Win_Probability'].idxmax()
                optimal_price = results_df.iloc[optimal_idx]['Price']
                optimal_prob = results_df.iloc[optimal_idx]['Win_Probability']
                
                st.success(f"🎯 **Optimal Price:** ${optimal_price:,.0f} (Win Prob: {optimal_prob:.1%})")
                
                # Results table
                st.dataframe(results_df.round(3))
    
    def _make_prediction(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Make predictions using the selected model"""
        try:
            if not self.feature_engineer:
                st.error("Feature engineering pipeline not loaded.")
                return None
            
            if st.session_state.selected_model not in self.models:
                st.error(f"Model '{st.session_state.selected_model}' not found.")
                return None
            
            # Feature engineering
            df_features = self.feature_engineer.create_comprehensive_features(df, fit=False)
            
            # Prepare features (exclude target and ID columns)
            exclude_cols = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Sale_ID', 'Status']
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            X = df_features[feature_cols]
            
            # Make predictions
            model = self.models[st.session_state.selected_model]
            
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)[:, 1]  # Get probability of positive class
            else:
                predictions = model.predict(X)
            
            return predictions
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None
    
    def render_performance_page(self):
        """Render model performance analytics page"""
        st.markdown('<h2 class="sub-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
        
        if not self.training_results:
            st.warning("No training results found. Please run the training pipeline first.")
            return
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        # Performance metrics table
        perf_data = []
        for model_name, results in self.training_results.items():
            if 'performance' in results and model_name != 'model_comparison':
                perf = results['performance']
                perf_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'AUC': perf.get('auc', 0),
                    'Accuracy': perf.get('accuracy', 0),
                    'Precision': perf.get('precision', 0),
                    'Recall': perf.get('recall', 0),
                    'F1 Score': perf.get('f1', 0)
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # Performance comparison chart
            fig = go.Figure()
            
            metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            for metric in metrics:
                fig.add_trace(go.Scatter(
                    x=perf_df['Model'], 
                    y=perf_df[metric],
                    mode='markers+lines',
                    name=metric,
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(perf_df.round(3))
        
        # Feature importance (if available)
        st.subheader("🎯 Feature Importance Analysis")
        
        if st.session_state.selected_model in self.training_results:
            model_results = self.training_results[st.session_state.selected_model]
            
            if 'feature_importance' in model_results:
                feat_imp = model_results['feature_importance']
                if feat_imp:
                    # Convert to DataFrame and sort
                    feat_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'Importance'])
                    feat_df = feat_df.sort_values('Importance', ascending=True).tail(20)
                    
                    # Plot feature importance
                    fig = px.bar(
                        feat_df, x='Importance', y='Feature', orientation='h',
                        title=f'Top 20 Feature Importance - {st.session_state.selected_model}',
                        labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")
            else:
                st.info("Feature importance not available for this model.")
    
    def render_training_monitor_page(self):
        """Render training monitoring page"""
        st.markdown('<h2 class="sub-header">🔍 Training Monitor</h2>', unsafe_allow_html=True)
        
        # Check for training state
        training_state_file = Path("logs/training_state.json")
        
        if training_state_file.exists():
            with open(training_state_file, 'r') as f:
                training_state = json.load(f)
            
            # Training status overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = training_state.get('status', 'Unknown')
                status_color = {'completed': '🟢', 'training': '🟡', 'failed': '🔴', 'initialized': '⚪'}.get(status, '⚫')
                st.metric("Status", f"{status_color} {status.title()}")
            
            with col2:
                progress = training_state.get('progress', 0)
                st.metric("Progress", f"{progress}%")
            
            with col3:
                models_trained = len(training_state.get('models_trained', []))
                st.metric("Models Trained", models_trained)
            
            with col4:
                errors = len(training_state.get('errors', []))
                st.metric("Errors", errors, delta=None if errors == 0 else "⚠️")
            
            # Progress bar
            st.progress(progress / 100.0)
            
            # Current step
            current_step = training_state.get('current_step', 'Unknown')
            st.info(f"**Current Step:** {current_step}")
            
            # Performance metrics timeline
            if 'performance_metrics' in training_state and training_state['performance_metrics']:
                st.subheader("📊 Training Performance")
                
                perf_metrics = training_state['performance_metrics']
                models = list(perf_metrics.keys())
                
                if models:
                    # Create performance comparison
                    metrics_data = []
                    for model, metrics in perf_metrics.items():
                        for metric, value in metrics.items():
                            metrics_data.append({
                                'Model': model,
                                'Metric': metric,
                                'Value': value
                            })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    fig = px.bar(
                        metrics_df, x='Model', y='Value', color='Metric', barmode='group',
                        title='Training Performance by Model'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Errors and warnings
            if training_state.get('errors'):
                st.subheader("⚠️ Training Errors")
                for error in training_state['errors']:
                    with st.expander(f"Error in {error.get('step', 'Unknown')}"):
                        st.error(f"**Error:** {error.get('error', 'Unknown error')}")
                        st.text(f"**Time:** {error.get('timestamp', 'Unknown time')}")
            
            # Training logs
            log_file = Path("logs/model_training.log")
            if log_file.exists():
                st.subheader("📋 Training Logs")
                
                # Read last N lines of log file
                with st.expander("View Recent Log Entries"):
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            recent_lines = lines[-50:]  # Last 50 lines
                            st.code('\n'.join(recent_lines), language='text')
                    except Exception as e:
                        st.error(f"Could not read log file: {e}")
        
        else:
            st.info("No training state found. Start training to see monitoring information.")
            
            if st.button("🚀 Start Training"):
                st.code("python model_training.py")
                st.info("Run the above command in your terminal to start training.")
    
    def run(self):
        """Main dashboard runner"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get navigation
            page, uploaded_file, show_advanced = self.render_sidebar()
            
            # Handle file upload
            if uploaded_file is not None:
                if uploaded_file.type == "text/csv":
                    self.sample_data = pd.read_csv(uploaded_file)
                else:
                    st.error("Please upload a CSV file.")
                    return
                
                st.session_state.data_loaded = True
                st.success(f"✅ Data uploaded successfully! Shape: {self.sample_data.shape}")
            
            # Route to appropriate page
            if page == "📊 EDA & Insights":
                self.render_eda_page()
            elif page == "🤖 Model Inference":
                self.render_inference_page()
            elif page == "📈 Performance Analytics":
                self.render_performance_page()
            elif page == "🔍 Training Monitor":
                self.render_training_monitor_page()
            
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.exception(e)


def main():
    """Main function"""
    dashboard = PriceElasticityDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

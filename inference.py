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
    import shap
    from scipy.optimize import minimize
    HAS_ADVANCED_FEATURES = True
except ImportError:
    HAS_ADVANCED_FEATURES = False
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
                ["📊 EDA & Insights", "🤖 Model Inference", "🔍 Scenario Analysis", "📈 Performance Analytics", "🔍 Training Monitor"],
                icons=['bar-chart', 'cpu', 'target', 'graph-up', 'search'],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
            )
        else:
            page = st.sidebar.selectbox(
                "Select Page",
                ["📊 EDA & Insights", "🤖 Model Inference", "🔍 Scenario Analysis", "📈 Performance Analytics", "🔍 Training Monitor"]
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
    
    def render_scenario_analysis_page(self):
        """
        Render advanced scenario analysis page
        Following Requirement 7
        """
        st.markdown('<h2 class="sub-header">🔍 Scenario Analysis & Simulation</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ No trained models found. Please run the training pipeline first.")
            return
        
        # Scenario Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Price Impact Simulation", 
            "🎯 Strategy Optimization", 
            "⚔️ Competitive Response", 
            "📊 Multi-Scenario Analysis"
        ])
        
        with tab1:
            self._render_price_impact_simulation()
        
        with tab2:
            self._render_strategy_optimization()
        
        with tab3:
            self._render_competitive_response_modeling()
        
        with tab4:
            self._render_multi_scenario_analysis()
    
    def _render_price_impact_simulation(self):
        """Render price impact simulation interface"""
        st.subheader("💰 Price Change Impact Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Simulation Parameters**")
            
            # Price adjustment range
            price_adjustment = st.slider(
                "Price Adjustment (%)", 
                min_value=-50, max_value=50, value=0, step=5,
                help="Percentage change in pricing"
            )
            
            # Segment selection
            segments = ['All', 'Enterprise', 'Mid-Market', 'SMB']
            selected_segment = st.selectbox("Customer Segment", segments)
            
            # Product category
            categories = ['All', 'Software', 'Hardware', 'Services']
            selected_category = st.selectbox("Product Category", categories)
            
            # Market conditions
            market_condition = st.selectbox(
                "Market Conditions", 
                ['Normal', 'Competitive', 'Economic Downturn', 'Growth Phase']
            )
            
            if st.button("🚀 Run Simulation"):
                # Generate simulation results
                simulation_results = self._simulate_price_impact(
                    price_adjustment, selected_segment, selected_category, market_condition
                )
                
                # Store results in session state
                st.session_state.simulation_results = simulation_results
        
        with col2:
            if hasattr(st.session_state, 'simulation_results'):
                results = st.session_state.simulation_results
                
                # Display key metrics
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    st.metric(
                        "Win Rate Impact", 
                        f"{results['win_rate_change']:+.1%}",
                        delta=f"{results['win_rate_change']:+.1%}"
                    )
                
                with col2b:
                    st.metric(
                        "Revenue Impact", 
                        f"{results['revenue_change']:+.1%}",
                        delta=f"${results['revenue_dollar_impact']:+,.0f}"
                    )
                
                with col2c:
                    st.metric(
                        "Margin Impact", 
                        f"{results['margin_change']:+.1%}",
                        delta=f"{results['margin_change']:+.1%}"
                    )
                
                # Visualization
                fig = go.Figure()
                
                # Win rate curve
                fig.add_trace(go.Scatter(
                    x=results['price_range'],
                    y=results['win_rates'],
                    mode='lines+markers',
                    name='Win Rate',
                    line=dict(color='blue', width=3)
                ))
                
                # Current point
                current_idx = len(results['price_range']) // 2
                fig.add_trace(go.Scatter(
                    x=[results['price_range'][current_idx]],
                    y=[results['win_rates'][current_idx]],
                    mode='markers',
                    name='Current Position',
                    marker=dict(color='red', size=12, symbol='diamond')
                ))
                
                fig.update_layout(
                    title="Win Rate vs Price Adjustment",
                    xaxis_title="Price Adjustment (%)",
                    yaxis_title="Win Rate",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_optimization(self):
        """Render strategy optimization interface"""
        st.subheader("🎯 Pricing Strategy Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Optimization Objectives**")
            
            # Objective selection
            objective = st.selectbox(
                "Primary Objective",
                ['Revenue Maximization', 'Profit Maximization', 'Win Rate Maximization', 'Market Share Growth']
            )
            
            # Constraints
            st.markdown("**Constraints**")
            min_win_rate = st.slider("Minimum Win Rate (%)", 0, 100, 30)
            max_price_increase = st.slider("Max Price Increase (%)", 0, 100, 20)
            min_margin = st.slider("Minimum Margin (%)", 0, 50, 15)
            
            # Risk tolerance
            risk_tolerance = st.selectbox("Risk Tolerance", ['Conservative', 'Moderate', 'Aggressive'])
            
            if st.button("🔍 Optimize Strategy"):
                optimization_results = self._optimize_pricing_strategy(
                    objective, min_win_rate, max_price_increase, min_margin, risk_tolerance
                )
                st.session_state.optimization_results = optimization_results
        
        with col2:
            if hasattr(st.session_state, 'optimization_results'):
                results = st.session_state.optimization_results
                
                st.success("✅ Optimization Complete!")
                
                # Recommended strategy
                st.markdown("**Recommended Strategy**")
                st.info(f"""
                **Optimal Price Adjustment:** {results['optimal_price_change']:+.1%}
                
                **Expected Outcomes:**
                - Win Rate: {results['expected_win_rate']:.1%}
                - Revenue Change: {results['expected_revenue_change']:+.1%}
                - Margin: {results['expected_margin']:.1%}
                
                **Confidence Level:** {results['confidence_level']}
                """)
                
                # Strategy comparison chart
                strategies = results['strategy_comparison']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=[s['price_change'] for s in strategies],
                    y=[s['expected_return'] for s in strategies],
                    mode='markers',
                    marker=dict(
                        size=[s['risk_score']*20 for s in strategies],
                        color=[s['win_rate'] for s in strategies],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Win Rate")
                    ),
                    text=[f"Strategy {i+1}" for i in range(len(strategies))],
                    name='Strategies'
                ))
                
                # Highlight optimal strategy
                optimal = results['optimal_strategy']
                fig.add_trace(go.Scatter(
                    x=[optimal['price_change']],
                    y=[optimal['expected_return']],
                    mode='markers',
                    marker=dict(color='gold', size=20, symbol='star'),
                    name='Optimal Strategy'
                ))
                
                fig.update_layout(
                    title="Strategy Risk-Return Analysis",
                    xaxis_title="Price Change (%)",
                    yaxis_title="Expected Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_competitive_response_modeling(self):
        """Render competitive response modeling interface"""
        st.subheader("⚔️ Competitive Response Modeling")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Competitive Scenario**")
            
            # Our pricing action
            our_action = st.selectbox(
                "Our Pricing Action",
                ['Price Increase', 'Price Decrease', 'Maintain Price', 'Aggressive Discount']
            )
            
            action_magnitude = st.slider("Action Magnitude (%)", 0, 50, 10)
            
            # Competitor characteristics
            competitor_aggressiveness = st.selectbox(
                "Competitor Aggressiveness",
                ['Low', 'Medium', 'High']
            )
            
            market_concentration = st.selectbox(
                "Market Concentration",
                ['Fragmented', 'Moderate', 'Concentrated']
            )
            
            # Time horizon
            time_horizon = st.selectbox(
                "Analysis Time Horizon",
                ['1 Month', '3 Months', '6 Months', '1 Year']
            )
            
            if st.button("🎮 Model Competitive Response"):
                competitive_results = self._model_competitive_response(
                    our_action, action_magnitude, competitor_aggressiveness, 
                    market_concentration, time_horizon
                )
                st.session_state.competitive_results = competitive_results
        
        with col2:
            if hasattr(st.session_state, 'competitive_results'):
                results = st.session_state.competitive_results
                
                # Market share impact
                st.markdown("**Market Share Impact**")
                
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    st.metric(
                        "Our Market Share",
                        f"{results['our_market_share']:.1%}",
                        delta=f"{results['market_share_change']:+.1%}"
                    )
                
                with col2b:
                    st.metric(
                        "Competitor Response",
                        results['competitor_response'],
                        delta=f"{results['response_magnitude']:+.1%}"
                    )
                
                with col2c:
                    st.metric(
                        "Market Stability",
                        results['market_stability'],
                        delta=results['stability_trend']
                    )
                
                # Competitive dynamics visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Market Share Evolution', 'Price Competition Timeline'),
                    shared_xaxes=True
                )
                
                # Market share over time
                time_periods = results['time_periods']
                fig.add_trace(
                    go.Scatter(
                        x=time_periods,
                        y=results['our_share_evolution'],
                        mode='lines+markers',
                        name='Our Share',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_periods,
                        y=results['competitor_share_evolution'],
                        mode='lines+markers',
                        name='Competitor Share',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                # Price evolution
                fig.add_trace(
                    go.Scatter(
                        x=time_periods,
                        y=results['our_price_evolution'],
                        mode='lines+markers',
                        name='Our Price',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_periods,
                        y=results['competitor_price_evolution'],
                        mode='lines+markers',
                        name='Competitor Price',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title_text="Competitive Dynamics Analysis")
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_multi_scenario_analysis(self):
        """Render multi-scenario analysis interface"""
        st.subheader("📊 Multi-Scenario Analysis")
        
        # Scenario builder
        st.markdown("**Scenario Builder**")
        
        scenarios = []
        
        # Allow users to create multiple scenarios
        num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=5, value=3)
        
        for i in range(num_scenarios):
            with st.expander(f"Scenario {i+1}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_change = st.slider(f"Price Change % (S{i+1})", -30, 30, 0, key=f"price_{i}")
                    market_condition = st.selectbox(f"Market Condition (S{i+1})", 
                                                  ['Normal', 'Recession', 'Growth'], key=f"market_{i}")
                
                with col2:
                    competition = st.selectbox(f"Competition Level (S{i+1})", 
                                             ['Low', 'Medium', 'High'], key=f"comp_{i}")
                    demand_shift = st.slider(f"Demand Shift % (S{i+1})", -20, 20, 0, key=f"demand_{i}")
                
                with col3:
                    probability = st.slider(f"Scenario Probability (S{i+1})", 0.0, 1.0, 1.0/num_scenarios, key=f"prob_{i}")
                
                scenarios.append({
                    'name': f'Scenario {i+1}',
                    'price_change': price_change,
                    'market_condition': market_condition,
                    'competition': competition,
                    'demand_shift': demand_shift,
                    'probability': probability
                })
        
        if st.button("🔄 Run Multi-Scenario Analysis"):
            # Normalize probabilities
            total_prob = sum(s['probability'] for s in scenarios)
            for scenario in scenarios:
                scenario['probability'] = scenario['probability'] / total_prob
            
            multi_scenario_results = self._run_multi_scenario_analysis(scenarios)
            st.session_state.multi_scenario_results = multi_scenario_results
        
        # Display results
        if hasattr(st.session_state, 'multi_scenario_results'):
            results = st.session_state.multi_scenario_results
            
            # Expected value analysis
            st.markdown("**Expected Value Analysis**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Revenue Change", f"{results['expected_revenue_change']:+.1%}")
            
            with col2:
                st.metric("Revenue at Risk (95%)", f"{results['revenue_at_risk']:+.1%}")
            
            with col3:
                st.metric("Best Case Scenario", f"{results['best_case']:+.1%}")
            
            with col4:
                st.metric("Worst Case Scenario", f"{results['worst_case']:+.1%}")
            
            # Scenario comparison
            scenario_df = pd.DataFrame(results['scenario_details'])
            
            fig = px.bar(
                scenario_df,
                x='scenario_name',
                y='expected_outcome',
                color='probability',
                title="Scenario Outcomes Comparison",
                labels={'expected_outcome': 'Expected Revenue Change (%)', 'scenario_name': 'Scenario'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk analysis
            st.markdown("**Risk Analysis**")
            
            risk_fig = go.Figure()
            
            # Add probability distribution
            risk_fig.add_trace(go.Histogram(
                x=results['outcome_distribution'],
                nbinsx=20,
                name='Outcome Distribution',
                opacity=0.7
            ))
            
            # Add VaR line
            risk_fig.add_vline(
                x=results['revenue_at_risk'],
                line_dash="dash",
                line_color="red",
                annotation_text="Value at Risk (95%)"
            )
            
            risk_fig.update_layout(
                title="Revenue Change Distribution",
                xaxis_title="Revenue Change (%)",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(risk_fig, use_container_width=True)
    
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
            elif page == "🔍 Scenario Analysis":
                self.render_scenario_analysis_page()
            elif page == "📈 Performance Analytics":
                self.render_performance_page()
            elif page == "🔍 Training Monitor":
                self.render_training_monitor_page()
            
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.exception(e)


    def _simulate_price_impact(self, price_adjustment: float, segment: str, category: str, market_condition: str) -> Dict[str, Any]:
        """Simulate the impact of price changes on business metrics"""
        try:
            # Generate price range around the adjustment
            price_range = np.linspace(price_adjustment - 20, price_adjustment + 20, 21)
            
            # Simulate win rates based on price elasticity
            base_win_rate = 0.35  # Base win rate
            elasticity = -0.8 if market_condition == 'Competitive' else -0.5  # Price elasticity
            
            win_rates = []
            for price_change in price_range:
                # Apply elasticity formula
                win_rate = base_win_rate * (1 + elasticity * (price_change / 100))
                
                # Apply segment adjustments
                if segment == 'Enterprise':
                    win_rate *= 1.1  # Less price sensitive
                elif segment == 'SMB':
                    win_rate *= 0.9  # More price sensitive
                
                # Apply category adjustments
                if category == 'Software':
                    win_rate *= 1.05  # Higher margins, less sensitivity
                elif category == 'Hardware':
                    win_rate *= 0.95  # More commoditized
                
                win_rates.append(max(0.05, min(0.95, win_rate)))  # Bound between 5% and 95%
            
            # Calculate current position
            current_idx = len(price_range) // 2
            current_win_rate = win_rates[current_idx]
            new_win_rate = win_rates[np.argmin(np.abs(price_range - price_adjustment))]
            
            # Calculate impacts
            win_rate_change = new_win_rate - current_win_rate
            revenue_change = price_adjustment / 100 + win_rate_change  # Price effect + volume effect
            margin_change = price_adjustment / 100 * 0.8  # Assuming 80% flows to margin
            
            # Dollar impact (assuming base revenue)
            base_revenue = 1000000  # $1M base
            revenue_dollar_impact = base_revenue * revenue_change
            
            return {
                'price_range': price_range.tolist(),
                'win_rates': win_rates,
                'win_rate_change': win_rate_change,
                'revenue_change': revenue_change,
                'margin_change': margin_change,
                'revenue_dollar_impact': revenue_dollar_impact,
                'current_win_rate': current_win_rate,
                'new_win_rate': new_win_rate
            }
        
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return {}
    
    def _optimize_pricing_strategy(self, objective: str, min_win_rate: float, max_price_increase: float, 
                                 min_margin: float, risk_tolerance: str) -> Dict[str, Any]:
        """Optimize pricing strategy based on objectives and constraints"""
        try:
            # Generate strategy alternatives
            price_changes = np.linspace(-20, max_price_increase, 20)
            strategies = []
            
            for price_change in price_changes:
                # Calculate expected outcomes
                base_win_rate = 0.35
                elasticity = -0.6
                
                win_rate = base_win_rate * (1 + elasticity * (price_change / 100))
                win_rate = max(0.05, min(0.95, win_rate))
                
                # Skip if below minimum win rate
                if win_rate < min_win_rate / 100:
                    continue
                
                revenue_change = price_change / 100 + (win_rate - base_win_rate)
                margin = min_margin / 100 + price_change / 100 * 0.8
                
                # Skip if below minimum margin
                if margin < min_margin / 100:
                    continue
                
                # Calculate expected return based on objective
                if objective == 'Revenue Maximization':
                    expected_return = revenue_change
                elif objective == 'Profit Maximization':
                    expected_return = revenue_change * margin
                elif objective == 'Win Rate Maximization':
                    expected_return = win_rate - base_win_rate
                else:  # Market Share Growth
                    expected_return = (win_rate - base_win_rate) * 2  # Weight win rate more
                
                # Risk score based on price change magnitude
                risk_score = abs(price_change) / 50  # Normalize to 0-1
                
                # Adjust for risk tolerance
                if risk_tolerance == 'Conservative':
                    expected_return -= risk_score * 0.5
                elif risk_tolerance == 'Aggressive':
                    expected_return += risk_score * 0.2
                
                strategies.append({
                    'price_change': price_change,
                    'win_rate': win_rate,
                    'expected_return': expected_return,
                    'risk_score': risk_score,
                    'revenue_change': revenue_change,
                    'margin': margin
                })
            
            if not strategies:
                return {'error': 'No feasible strategies found with given constraints'}
            
            # Find optimal strategy
            optimal_strategy = max(strategies, key=lambda x: x['expected_return'])
            
            # Confidence level based on number of feasible strategies and risk
            if len(strategies) > 10 and optimal_strategy['risk_score'] < 0.3:
                confidence_level = 'High'
            elif len(strategies) > 5:
                confidence_level = 'Medium'
            else:
                confidence_level = 'Low'
            
            return {
                'optimal_price_change': optimal_strategy['price_change'],
                'expected_win_rate': optimal_strategy['win_rate'],
                'expected_revenue_change': optimal_strategy['revenue_change'],
                'expected_margin': optimal_strategy['margin'],
                'confidence_level': confidence_level,
                'strategy_comparison': strategies,
                'optimal_strategy': optimal_strategy
            }
        
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            return {}
    
    def _model_competitive_response(self, our_action: str, action_magnitude: float, 
                                  competitor_aggressiveness: str, market_concentration: str, 
                                  time_horizon: str) -> Dict[str, Any]:
        """Model competitive response to our pricing actions"""
        try:
            # Time periods based on horizon
            horizon_map = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12}
            periods = horizon_map[time_horizon]
            time_periods = list(range(periods + 1))
            
            # Initial market shares
            our_initial_share = 0.25
            competitor_initial_share = 0.75
            
            # Response parameters
            aggressiveness_map = {'Low': 0.3, 'Medium': 0.6, 'High': 0.9}
            concentration_map = {'Fragmented': 0.4, 'Moderate': 0.6, 'Concentrated': 0.8}
            
            response_factor = aggressiveness_map[competitor_aggressiveness]
            concentration_factor = concentration_map[market_concentration]
            
            # Simulate evolution over time
            our_share_evolution = [our_initial_share]
            competitor_share_evolution = [competitor_initial_share]
            our_price_evolution = [100]  # Base price index
            competitor_price_evolution = [100]
            
            # Our action impact
            if our_action == 'Price Decrease':
                our_price_change = -action_magnitude
                share_gain = action_magnitude * 0.02  # 2% share gain per 1% price cut
            elif our_action == 'Price Increase':
                our_price_change = action_magnitude
                share_gain = -action_magnitude * 0.015  # 1.5% share loss per 1% price increase
            elif our_action == 'Aggressive Discount':
                our_price_change = -action_magnitude * 1.5
                share_gain = action_magnitude * 0.03
            else:  # Maintain Price
                our_price_change = 0
                share_gain = 0
            
            for period in range(1, periods + 1):
                # Competitor response (delayed and scaled)
                response_delay = max(1, period - 1)  # Competitors respond with delay
                competitor_response = our_price_change * response_factor * concentration_factor
                competitor_response *= min(1, response_delay / 2)  # Gradual response
                
                # Update prices
                our_price = our_price_evolution[-1] * (1 + our_price_change / 100)
                competitor_price = competitor_price_evolution[-1] * (1 + competitor_response / 100)
                
                our_price_evolution.append(our_price)
                competitor_price_evolution.append(competitor_price)
                
                # Update market shares
                relative_price_advantage = (competitor_price - our_price) / our_price
                share_change = relative_price_advantage * 0.1  # 10% elasticity
                
                new_our_share = our_share_evolution[-1] + share_change
                new_competitor_share = 1 - new_our_share
                
                # Bound shares
                new_our_share = max(0.05, min(0.95, new_our_share))
                new_competitor_share = 1 - new_our_share
                
                our_share_evolution.append(new_our_share)
                competitor_share_evolution.append(new_competitor_share)
            
            # Determine competitor response type
            if abs(competitor_response) > action_magnitude * 0.8:
                response_type = 'Aggressive Match'
            elif abs(competitor_response) > action_magnitude * 0.4:
                response_type = 'Moderate Response'
            else:
                response_type = 'Limited Response'
            
            # Market stability
            price_volatility = np.std(our_price_evolution + competitor_price_evolution)
            if price_volatility < 5:
                stability = 'Stable'
                stability_trend = '→'
            elif price_volatility < 15:
                stability = 'Moderate'
                stability_trend = '↕'
            else:
                stability = 'Volatile'
                stability_trend = '⚠'
            
            return {
                'our_market_share': our_share_evolution[-1],
                'market_share_change': our_share_evolution[-1] - our_initial_share,
                'competitor_response': response_type,
                'response_magnitude': competitor_response,
                'market_stability': stability,
                'stability_trend': stability_trend,
                'time_periods': time_periods,
                'our_share_evolution': our_share_evolution,
                'competitor_share_evolution': competitor_share_evolution,
                'our_price_evolution': our_price_evolution,
                'competitor_price_evolution': competitor_price_evolution
            }
        
        except Exception as e:
            st.error(f"Competitive modeling failed: {e}")
            return {}
    
    def _run_multi_scenario_analysis(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Run multi-scenario analysis with risk assessment"""
        try:
            scenario_outcomes = []
            outcome_distribution = []
            
            for scenario in scenarios:
                # Calculate outcome for each scenario
                price_change = scenario['price_change']
                demand_shift = scenario['demand_shift']
                
                # Base calculations
                base_revenue_change = price_change / 100
                
                # Market condition adjustments
                if scenario['market_condition'] == 'Recession':
                    base_revenue_change *= 0.7
                elif scenario['market_condition'] == 'Growth':
                    base_revenue_change *= 1.3
                
                # Competition adjustments
                competition_impact = {'Low': 1.1, 'Medium': 1.0, 'High': 0.9}
                base_revenue_change *= competition_impact[scenario['competition']]
                
                # Demand shift
                base_revenue_change += demand_shift / 100
                
                scenario_outcome = {
                    'scenario_name': scenario['name'],
                    'expected_outcome': base_revenue_change,
                    'probability': scenario['probability']
                }
                
                scenario_outcomes.append(scenario_outcome)
                
                # Add to distribution (weighted by probability)
                num_samples = int(scenario['probability'] * 1000)
                # Add some randomness around the expected outcome
                scenario_samples = np.random.normal(base_revenue_change, abs(base_revenue_change) * 0.2, num_samples)
                outcome_distribution.extend(scenario_samples)
            
            # Calculate expected value
            expected_revenue_change = sum(s['expected_outcome'] * s['probability'] for s in scenario_outcomes)
            
            # Risk metrics
            outcome_distribution = np.array(outcome_distribution)
            revenue_at_risk = np.percentile(outcome_distribution, 5)  # 95% VaR
            best_case = np.percentile(outcome_distribution, 95)
            worst_case = np.percentile(outcome_distribution, 5)
            
            return {
                'expected_revenue_change': expected_revenue_change,
                'revenue_at_risk': revenue_at_risk,
                'best_case': best_case,
                'worst_case': worst_case,
                'scenario_details': scenario_outcomes,
                'outcome_distribution': outcome_distribution.tolist()
            }
        
        except Exception as e:
            st.error(f"Multi-scenario analysis failed: {e}")
            return {}


def main():
    """Main function"""
    dashboard = PriceElasticityDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

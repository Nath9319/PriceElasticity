# Price Elasticity Data Science System - User Guide

## 🎯 Overview

The Price Elasticity Data Science System is a comprehensive B2B pricing analytics platform that uses advanced machine learning to predict quote win probabilities and analyze price sensitivity across customer segments and product categories.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Virtual environment (recommended)
- At least 4GB RAM available
- 2GB free disk space

### Installation
```bash
# Clone or navigate to the project directory
cd "Price Elasticity"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 System Architecture

```
Raw Data → EDA → Feature Engineering → Model Training → Inference → Business Insights
     ↓         ↓            ↓               ↓            ↓            ↓
   10K Quotes → Analysis → 201 Features → ML Models → Predictions → CSV Reports
```

## 🎮 Usage Scenarios

### Scenario 1: Complete Pipeline (Recommended for First Time)
**Use Case**: You want to run the entire system from data generation to final results.

```bash
python scripts/run_complete_pipeline.py
```

**What This Does**:
- ✅ Generates realistic sample data (10K quotes, 2K customers, 500 products)
- ✅ Performs comprehensive EDA
- ✅ Engineers 201 advanced features
- ✅ Trains machine learning models
- ✅ Generates business insights
- ✅ Saves all artifacts for reuse

**Time**: ~5-10 minutes  
**Output**: Trained models, processed data, summary reports

### Scenario 2: Inference on New Data
**Use Case**: You have trained models and want to predict on new quote data.

```bash
python run_inference_and_save.py
```

**Requirements**: 
- Trained models must exist in `models/trained/`
- Feature engineering artifacts in `models/feature_engineering/`
- New data in the expected format

**Output**: CSV files with predictions and business analysis

### Scenario 3: Interactive Dashboard
**Use Case**: You want a user-friendly web interface for exploration and what-if analysis.

```bash
streamlit run inference.py
```

**Features**:
- 📊 Interactive data exploration
- 🤖 Real-time model inference
- 🔍 Scenario analysis and simulation
- 📈 Performance monitoring
- 💡 Business insights dashboard

## 📁 Directory Structure

```
Price Elasticity/
├── 📂 datasets/           # Generated sample data
│   ├── quote_history.csv  # Main quotes dataset
│   ├── sales_history.csv  # Historical sales
│   ├── customer_master.csv
│   ├── customer_segmentation.csv
│   └── product_master.csv
├── 📂 models/
│   ├── trained/           # Trained ML models
│   └── feature_engineering/ # Feature processing artifacts
├── 📂 results/
│   ├── inference/         # Prediction results (CSV)
│   ├── eda/              # Exploratory data analysis
│   └── plots/            # Visualizations
├── 📂 src/               # Source code
│   ├── feature_engineering/
│   ├── training/
│   └── utils/
└── 📂 scripts/           # Execution scripts
```

## 🔧 Core Components

### 1. Data Generation & Management
- **Script**: `scripts/run_complete_pipeline.py`
- **Purpose**: Creates realistic B2B pricing datasets
- **Features**: 10K quotes, customer segmentation, product hierarchies
- **Output**: CSV files in `datasets/` folder

### 2. Feature Engineering (201 Features)
- **Location**: `src/feature_engineering/feature_engineering.py`
- **Advanced Features**:
  - 💰 Price dynamics (volatility, trends, ratios)
  - 🏆 Competitive positioning (market position, premiums)
  - 👥 Customer value (RFM, CLV, tenure analysis)
  - 📦 Product hierarchy (lifecycle, performance)
  - ⏰ Temporal patterns (seasonality, trends, lags)
  - 🌐 Graph neural networks (customer-product relationships)
  - 🔗 Advanced interactions (price sensitivity, segment effects)

### 3. Machine Learning Models
- **Hierarchical Bayesian Model**: Multi-level modeling for customer/product effects
- **Graph Neural Network**: Relationship modeling between customers and products
- **Ensemble Methods**: Combined predictions for improved accuracy

### 4. Business Intelligence
- **Segment Analysis**: Win rates and pricing by customer segment
- **Category Analysis**: Product category performance insights
- **Price Sensitivity**: Elasticity analysis across dimensions
- **Scenario Modeling**: What-if analysis for pricing strategies

## 📊 Data Schema

### Core Tables
```sql
-- Quote History (Main Table)
Quote_ID, Customer_ID, Product_ID, Quote_Date, List_Price, Net_Price, 
Offered_Price, Status, Product_Category, Competition_Status, Region

-- Customer Master
Customer_ID, Customer_Name, Industry, Company_Size, Customer_Since_Date,
Credit_Rating, Annual_Revenue

-- Customer Segmentation  
Customer_ID, Customer_Segment, RFM_Score, CLV_Score, Price_Sensitivity,
Negotiation_Style

-- Product Master
Product_ID, Product_Name, Product_Category, Product_Line, Launch_Date,
Standard_Cost, List_Price, Lifecycle_Stage

-- Sales History
Sale_ID, Customer_ID, Product_ID, Sale_Date, Quantity, Unit_Price,
Total_Revenue, COGS
```

## 🤖 Model Usage

### Training New Models
```python
from training.model_training import PriceElasticityModelTraining

# Initialize trainer
trainer = PriceElasticityModelTraining()

# Prepare your data (must have 'Status' column for target)
X, y = trainer.prepare_training_data(your_data, target_col='Status')

# Train all models
results = trainer.train_all_models(X, y)

# Save models
trainer.save_models()
```

### Making Predictions
```python
from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
import joblib

# Load feature engineering pipeline
fe = PriceElasticityFeatureEngineering()
fe.load_feature_engineering_artifacts()

# Load trained model
model = joblib.load('models/trained/hierarchical_bayesian_model.pkl')

# Prepare new data
new_features = fe.create_comprehensive_features(new_data, fit=False)

# Make predictions
predictions = model.predict_proba(new_features)[:, 1]  # Win probability
```

## 📈 Interpreting Results

### Model Performance Metrics
- **AUC Score**: Area Under ROC Curve (>0.7 is good for business use)
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Win prediction quality
- **Feature Importance**: Key drivers of win/loss

### Business Insights
- **Win Rate Analysis**: Success rates by segment/category
- **Price Sensitivity**: How discount depth affects win probability
- **Customer Value**: RFM and CLV scoring for targeting
- **Product Performance**: Category and lifecycle analysis

### CSV Output Files
1. **`full_inference_results_*.csv`**: Complete predictions dataset
   - All original quote data
   - Model predictions and probabilities
   - Business analysis fields

2. **`segment_analysis_*.csv`**: Customer segment performance
   - Win rates by segment
   - Average pricing and discounts
   - Statistical summaries

3. **`category_analysis_*.csv`**: Product category insights
   - Category-specific win rates
   - Pricing patterns
   - Performance metrics

4. **`model_summary_*.csv`**: Model performance comparison
   - Accuracy metrics by model
   - Prediction distributions
   - Model reliability scores

## 🔍 Advanced Features

### Interactive Dashboard Features
When you run `streamlit run inference.py`, you get:

1. **📊 EDA & Insights Page**
   - Data exploration with interactive charts
   - Win rate analysis by segments
   - Price distribution analysis
   - Temporal trends

2. **🤖 Model Inference Page**
   - Single quote prediction
   - Batch prediction processing
   - What-if scenario analysis
   - Confidence scoring

3. **🔍 Scenario Analysis Page**
   - Price impact simulation
   - Strategy optimization
   - Competitive response modeling
   - Multi-scenario risk analysis

4. **📈 Performance Analytics Page**
   - Model comparison charts
   - Feature importance analysis
   - Training metrics
   - Business impact assessment

### API Integration
```python
# Example: Integrate with your CRM/ERP system
import requests
import pandas as pd

def predict_quote_win_probability(quote_data):
    """
    Predict win probability for a new quote
    """
    # Load your trained model and feature engineering pipeline
    # Process the quote_data through feature engineering
    # Return prediction with confidence intervals
    pass

# Usage in your business system
new_quote = {
    'Customer_ID': 'C1234',
    'Product_ID': 'P567',
    'List_Price': 10000,
    'Net_Price': 8500,
    'Competition_Status': 'Medium'
}

win_probability = predict_quote_win_probability(new_quote)
print(f"Win Probability: {win_probability:.2%}")
```

## ⚠️ Troubleshooting

### Common Issues

1. **ImportError: Module not found**
   ```bash
   # Solution: Install requirements
   pip install -r requirements.txt
   ```

2. **Memory Error during training**
   ```bash
   # Solution: Reduce data size or increase RAM
   # Modify the n_samples parameter in data generation
   ```

3. **Feature engineering fails**
   ```bash
   # Solution: Check data format matches expected schema
   # Ensure required columns exist: Quote_ID, Customer_ID, Product_ID, etc.
   ```

4. **Models not loading**
   ```bash
   # Solution: Run complete pipeline first
   python scripts/run_complete_pipeline.py
   ```

### Performance Optimization

1. **Speed up training**:
   - Reduce hyperparameter search space
   - Use smaller sample sizes for development
   - Enable parallel processing

2. **Reduce memory usage**:
   - Process data in chunks
   - Use sparse matrices for categorical features
   - Clear intermediate variables

3. **Improve accuracy**:
   - Collect more historical data
   - Add domain-specific features
   - Tune model hyperparameters

## 🎯 Business Use Cases

### 1. Pricing Strategy Optimization
- **Goal**: Determine optimal pricing for maximum win rate or revenue
- **Process**: Use scenario analysis to test different pricing strategies
- **Output**: Recommended prices with expected outcomes

### 2. Sales Team Support
- **Goal**: Help sales reps identify winnable deals and optimal pricing
- **Process**: Real-time prediction on new quotes
- **Output**: Win probability scores with recommendations

### 3. Customer Segmentation
- **Goal**: Understand price sensitivity across customer types
- **Process**: Analyze win rates and pricing patterns by segment
- **Output**: Targeted pricing strategies by segment

### 4. Product Portfolio Analysis
- **Goal**: Identify high-performing products and categories
- **Process**: Compare win rates and margins across products
- **Output**: Product strategy recommendations

### 5. Competitive Intelligence
- **Goal**: Understand impact of competitive pressure on pricing
- **Process**: Analyze win rates by competition status
- **Output**: Competitive response strategies

## 🔄 Maintenance & Updates

### Regular Maintenance
1. **Monthly**: Update models with new data
2. **Quarterly**: Review feature importance and model performance
3. **Annually**: Full model retraining and feature engineering review

### Data Updates
```python
# Add new data to existing datasets
new_quotes = pd.read_csv('new_quotes.csv')
existing_quotes = pd.read_csv('datasets/quote_history.csv')
updated_quotes = pd.concat([existing_quotes, new_quotes])
updated_quotes.to_csv('datasets/quote_history.csv', index=False)

# Retrain models with updated data
python scripts/run_complete_pipeline.py
```

## 📞 Support & Resources

### Documentation
- `EXECUTION_SUMMARY.md`: Complete execution results
- `README.md`: Technical overview and setup
- `requirements.txt`: Dependencies list
- Source code comments: Detailed implementation notes

### Model Artifacts
- `models/trained/`: Trained model files
- `models/feature_engineering/`: Feature processing components
- `results/`: Analysis outputs and reports

### Contact & Development
For questions, improvements, or custom implementations, refer to the source code documentation and execution logs in the `logs/` directory.

---

## 🎉 Success Metrics

After following this guide, you should achieve:
- ✅ **Model Accuracy**: AUC > 0.7 for business deployment
- ✅ **Feature Engineering**: 201 advanced features consistently applied
- ✅ **Business Insights**: Actionable pricing recommendations
- ✅ **Scalable Pipeline**: Production-ready inference system
- ✅ **Comprehensive Analysis**: Multiple CSV reports for decision-making

Welcome to advanced B2B pricing analytics! 🚀

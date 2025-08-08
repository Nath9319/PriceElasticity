# 🎯 Price Elasticity Analytics Platform

A comprehensive B2B price elasticity modeling platform with advanced machine learning capabilities, interactive analytics dashboard, and real-time inference system.

## 🌟 Features

### 🚀 Core Capabilities
- **Advanced ML Models**: Hierarchical Bayesian, X-Learner (Causal ML), and Ensemble approaches
- **Interactive Dashboard**: Streamlit-powered analytics and visualization platform
- **Real-time Inference**: Single predictions, batch processing, and what-if scenarios
- **Comprehensive EDA**: Interactive visualizations and insights
- **Model Monitoring**: Real-time training progress and performance analytics

### 🔧 Technical Features
- **Feature Engineering**: 100+ automated feature creation with price dynamics, customer value, and competitive positioning
- **Hyperparameter Optimization**: Optuna-powered automated hyperparameter tuning
- **Advanced Logging**: Comprehensive logging with rotation and monitoring
- **Model Persistence**: Automated model and pipeline serialization
- **Scalable Architecture**: Modular design for easy extension

## 🏗️ Project Structure

```
Price Elasticity/
├── src/                           # Source code modules
│   ├── analysis/                  # Advanced analytics and insights
│   │   └── scenario_analysis.py   # What-if scenario modeling
│   ├── data_processing/           # Data ingestion and preprocessing
│   ├── explainability/            # Model interpretation and explanation
│   │   └── model_explainer.py     # SHAP and feature importance analysis
│   ├── feature_engineering/       # Feature engineering pipeline
│   │   └── feature_engineering.py # Comprehensive feature creation
│   ├── inference/                 # Real-time prediction services
│   ├── models/                    # Advanced model architectures
│   │   └── graph_neural_networks.py # GNN for competitive analysis
│   ├── training/                  # Model training components
│   │   └── model_training.py      # Multi-model training pipeline
│   ├── utils/                     # Configuration and utilities
│   │   ├── __init__.py            # Package initialization
│   │   └── config_loader.py       # Configuration management
│   ├── validation/                # Data and model validation
│   │   └── data_validator.py      # Comprehensive data validation
│   └── visualization/             # Advanced plotting and dashboards
├── config/                        # Configuration files
├── data/                          # Data storage and samples
├── datasets/                      # Training and validation datasets
├── logs/                          # Application logs and monitoring
├── models/                        # Trained models and artifacts
│   ├── trained/                   # Saved model files
│   └── feature_engineering/       # Feature transformation artifacts
├── notebooks/                     # Jupyter notebooks for analysis
├── results/                       # Experiment results and reports
├── scripts/                       # Utility and automation scripts
│   └── demo_new_requirements.py   # Requirements demonstration
├── tests/                         # Unit and integration tests
├── model_training.py              # Main training pipeline
├── inference.py                   # Streamlit dashboard application
├── requirements.txt               # Python dependencies
├── comprehensive_validation_report.md # Validation documentation
├── requirements_validation_report.md  # Requirements analysis
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Nath9319/PriceElasticity.git
cd PriceElasticity

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

Run the comprehensive training pipeline:

```bash
python model_training.py
```

**Options:**
- `--data path/to/data.csv`: Use custom data file
- `--config path/to/config.yaml`: Use custom configuration
- `--log-level DEBUG`: Set logging level

**Example with custom data:**
```bash
python model_training.py --data data/my_pricing_data.csv --log-level INFO
```

### 3. Launch Analytics Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run inference.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Features

### 🔍 EDA & Insights
- **Interactive Visualizations**: Price distributions, win rate analysis, customer segmentation
- **Temporal Analysis**: Seasonal patterns, trends, and time-based insights  
- **Customer Analytics**: RFM analysis, segment performance, lifecycle metrics
- **Competitive Analysis**: Market positioning and competitive intelligence

### 🤖 Model Inference
- **Single Predictions**: Individual quote win probability prediction
- **Batch Processing**: Bulk prediction with performance metrics
- **What-If Analysis**: Scenario-based pricing optimization
- **Confidence Intervals**: Prediction uncertainty quantification

### 📈 Performance Analytics
- **Model Comparison**: Side-by-side performance metrics
- **Feature Importance**: Top drivers of price elasticity
- **ROC Analysis**: Model discrimination capabilities
- **Cross-validation Results**: Robust performance evaluation

### 🔍 Training Monitor
- **Real-time Progress**: Live training status and progress
- **Performance Tracking**: Model metrics during training
- **Error Monitoring**: Training issues and diagnostics
- **Log Analysis**: Detailed training logs and debugging

## 🎯 Model Training Pipeline

### Supported Models

1. **Hierarchical Bayesian Model**
   - Captures segment-level variations in price sensitivity
   - Incorporates uncertainty quantification
   - Uses PyMC for MCMC sampling

2. **X-Learner (Causal ML)**
   - Estimates treatment effects of pricing changes
   - Uses EconML for causal inference
   - Addresses selection bias and confounding

3. **Ensemble Model**
   - Stacking of LightGBM, XGBoost, CatBoost, Random Forest
   - Automated hyperparameter optimization
   - Ridge regression meta-learner

### Training Process

1. **Data Loading & Validation**
   - Automatic schema validation
   - Data quality checks
   - Missing value analysis

2. **Feature Engineering**
   - Price dynamics (volatility, trends, ratios)
   - Customer value (RFM, CLV, tenure)
   - Competitive positioning
   - Temporal features (seasonality, trends)
   - Product hierarchy features

3. **Model Training**
   - Cross-validation with time-series splits
   - Hyperparameter optimization with Optuna
   - Model selection and evaluation

4. **Model Persistence**
   - Automatic model serialization
   - Feature pipeline preservation
   - Metadata and configuration saving

## 📋 Data Requirements

### Required Columns
- `Quote_ID`: Unique quote identifier
- `Customer_ID`: Customer identifier
- `Product_ID`: Product identifier
- `Status`: Quote outcome ('Won'/'Lost')

### Optional Columns (Recommended)
- `Quote_Date`: Quote timestamp
- `List_Price`: Original price
- `Net_Price`: Final quoted price
- `Customer_Segment`: Customer classification
- `Product_Category`: Product classification
- `Competition_Status`: Competitive intensity
- `Lifecycle_Stage`: Product lifecycle stage

### Sample Data Format
```csv
Quote_ID,Customer_ID,Product_ID,Quote_Date,List_Price,Net_Price,Status,Customer_Segment,Product_Category
Q000001,C0001,P001,2023-01-15,10000,8500,Won,Enterprise,Software
Q000002,C0002,P002,2023-01-16,15000,14000,Lost,Mid-Market,Hardware
```

## ⚙️ Configuration

### Model Configuration
```yaml
models:
  hierarchical_bayesian:
    chains: 4
    draws: 2000
    tune: 1000
  
  x_learner:
    base_models: ['lightgbm', 'xgboost']
    dml_folds: 3
  
  ensemble:
    base_models: ['lightgbm', 'xgboost', 'catboost', 'random_forest']
    meta_learner: 'ridge'

hyperparameter_optimization:
  optuna:
    n_trials: 100
    timeout: 600
```

### Feature Engineering Configuration
```yaml
feature_engineering:
  price_dynamics:
    windows: [7, 30, 90]
    volatility_window: 90
  
  customer:
    rfm_quantiles: 5
    tenure_bins: [0, 90, 365, 730, 1825]
    clv_method: 'traditional'
```

## 📈 Performance Metrics

### Classification Metrics
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Business Metrics
- **Win Rate**: Proportion of won quotes
- **Revenue Impact**: Expected revenue change from pricing decisions
- **Price Sensitivity**: Elasticity coefficients by segment
- **Optimal Pricing**: Revenue-maximizing price points

## 🔧 Advanced Usage

### Custom Model Development
```python
from src.training.model_training import PriceElasticityModelTraining

# Initialize trainer
trainer = PriceElasticityModelTraining()

# Load your data
df = pd.read_csv('your_data.csv')
X, y = trainer.prepare_training_data(df)

# Train specific model
results = trainer.train_ensemble_model(X, y)
```

### Batch Inference
```python
from src.feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
import joblib

# Load model and feature pipeline
model = joblib.load('models/trained/ensemble_model.pkl')
fe_pipeline = PriceElasticityFeatureEngineering()
fe_pipeline.load_feature_engineering_artifacts()

# Process new data
new_data = pd.read_csv('new_quotes.csv')
features = fe_pipeline.create_comprehensive_features(new_data, fit=False)
predictions = model.predict_proba(features)[:, 1]
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Memory Issues**
   ```python
   # Reduce sample size in configuration
   # Use smaller hyperparameter search space
   # Enable model checkpointing
   ```

3. **Training Failures**
   ```bash
   # Check logs for detailed error messages
   tail -f logs/model_training.log
   
   # Validate data format and required columns
   python scripts/validate_data.py your_data.csv
   ```

### Performance Optimization

1. **Speed up Training**
   - Reduce hyperparameter search trials
   - Use smaller cross-validation folds
   - Disable advanced models (Bayesian/Causal)

2. **Improve Predictions**
   - Add more relevant features
   - Increase training data size
   - Tune model-specific parameters

## 📚 Documentation

### API Documentation
- [Feature Engineering API](docs/feature_engineering.md)
- [Model Training API](docs/model_training.md)
- [Inference API](docs/inference.md)

### Tutorials
- [Getting Started Tutorial](docs/tutorial_getting_started.md)
- [Advanced Modeling](docs/tutorial_advanced_modeling.md)
- [Custom Features](docs/tutorial_custom_features.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMC**: Bayesian modeling framework
- **EconML**: Causal machine learning library
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Interactive visualization library
- **Optuna**: Hyperparameter optimization framework

## 📞 Support

For questions, issues, or feature requests:

1. **GitHub Issues**: [Create an issue](https://github.com/Nath9319/PriceElasticity/issues)
2. **Documentation**: Check the [docs](docs/) folder
3. **Email**: Support available through GitHub

## 🚀 Roadmap

### Version 2.0
- [ ] MLflow integration for experiment tracking
- [ ] API endpoints for production deployment
- [ ] Advanced causal inference methods
- [ ] Multi-objective optimization
- [ ] Real-time streaming predictions

### Version 3.0
- [ ] Deep learning models (Neural ODEs, Transformers)
- [ ] Graph neural networks for competitive analysis
- [ ] Automated feature selection
- [ ] A/B testing framework integration
- [ ] Multi-tenancy support

---

**Happy Modeling! 🎯📊**

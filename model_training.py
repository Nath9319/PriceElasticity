#!/usr/bin/env python3
"""
Price Elasticity Model Training Script
=====================================

This script provides a comprehensive model training pipeline with:
- Advanced logging and progress tracking
- Real-time visualization with Streamlit
- Interactive model training monitoring
- Detailed performance analytics

Usage:
    python model_training.py [--config path/to/config.yaml] [--data path/to/data.csv]
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import yaml
import logging
from logging.handlers import RotatingFileHandler

# Import our modules
from feature_engineering.feature_engineering import PriceElasticityFeatureEngineering
from training.model_training import PriceElasticityModelTraining
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')


class ModelTrainingPipeline:
    """
    Comprehensive Model Training Pipeline with advanced logging and monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the training pipeline
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        self.config = self._load_config(config_path)
        self.setup_logging(log_level)
        
        # Initialize components
        self.feature_engineer = PriceElasticityFeatureEngineering(self.config)
        self.model_trainer = PriceElasticityModelTraining(self.config)
        
        # Training state
        self.training_state = {
            'status': 'initialized',
            'current_step': 'initialization',
            'progress': 0,
            'start_time': None,
            'models_trained': [],
            'performance_metrics': {},
            'errors': [],
            'warnings': []
        }
        
        # Create directories
        self._create_directories()
        
        self.logger.info("Model Training Pipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use default"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return config_loader
    
    def setup_logging(self, log_level: str):
        """Setup comprehensive logging"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("ModelTraining")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / "model_training.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "models/trained",
            "models/feature_engineering", 
            "outputs/plots",
            "outputs/reports",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _update_training_state(self, status: str, step: str, progress: int, **kwargs):
        """Update training state for monitoring"""
        self.training_state.update({
            'status': status,
            'current_step': step,
            'progress': min(progress, 100),
            'last_update': datetime.now().isoformat(),
            **kwargs
        })
        
        # Save state to file for external monitoring
        with open("logs/training_state.json", "w") as f:
            json.dump(self.training_state, f, indent=2, default=str)
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate training data
        
        Args:
            data_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info("🔄 Loading training data...")
        self._update_training_state("loading", "data_loading", 5)
        
        if data_path and Path(data_path).exists():
            # Load from specified path
            self.logger.info(f"Loading data from: {data_path}")
            
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
        else:
            # Generate sample data for demonstration
            self.logger.warning("No data file provided. Generating sample data for demonstration.")
            df = self._generate_sample_data()
        
        # Data validation
        self._validate_data(df)
        
        self.logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        self._update_training_state("loaded", "data_validation", 10)
        
        return df
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration"""
        self.logger.info("Generating sample training data...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Generate realistic B2B pricing data
        data = {
            'Quote_ID': [f'Q{i:06d}' for i in range(n_samples)],
            'Customer_ID': [f'C{i%1000:04d}' for i in range(n_samples)],
            'Product_ID': [f'P{i%200:03d}' for i in range(n_samples)],
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
        
        # Add some logical relationships
        # Higher discounts should increase win probability
        discount_depth = (df['List_Price'] - df['Net_Price']) / df['List_Price']
        win_prob = 0.2 + 0.3 * discount_depth + 0.1 * np.random.random(n_samples)
        df.loc[np.random.random(n_samples) < win_prob, 'Status'] = 'Won'
        
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate data quality and structure"""
        required_columns = ['Quote_ID', 'Customer_ID', 'Product_ID', 'Status']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if df['Status'].nunique() < 2:
            raise ValueError("Target variable must have at least 2 unique values")
        
        self.logger.info("✅ Data validation passed")
    
    def run_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run comprehensive feature engineering pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("🔧 Starting feature engineering pipeline...")
        self._update_training_state("processing", "feature_engineering", 15)
        
        try:
            # Create comprehensive features
            df_features = self.feature_engineer.create_comprehensive_features(df, fit=True)
            
            # Save feature engineering artifacts
            self.feature_engineer.save_feature_engineering_artifacts()
            
            self.logger.info(f"✅ Feature engineering completed: {df_features.shape[1]} features created")
            self._update_training_state("processed", "feature_engineering_complete", 30)
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {str(e)}")
            self.training_state['errors'].append({
                'step': 'feature_engineering',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def train_models(self, df_features: pd.DataFrame, target_col: str = 'Status') -> Dict[str, Any]:
        """
        Train all model types with progress tracking
        
        Args:
            df_features: DataFrame with engineered features
            target_col: Target column name
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("🚀 Starting model training pipeline...")
        self._update_training_state("training", "model_preparation", 35)
        
        # Prepare training data
        X, y = self.model_trainer.prepare_training_data(df_features, target_col)
        
        self.logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Track training progress
        models_to_train = ['hierarchical_bayesian', 'x_learner', 'ensemble']
        progress_step = 20 / len(models_to_train)  # 20% for all model training
        current_progress = 40
        
        all_results = {}
        
        # Train Hierarchical Bayesian Model
        try:
            self.logger.info("📊 Training Hierarchical Bayesian Model...")
            self._update_training_state("training", "hierarchical_bayesian", current_progress)
            
            hb_results = self.model_trainer.train_hierarchical_bayesian_model(X, y)
            all_results['hierarchical_bayesian'] = hb_results
            self.training_state['models_trained'].append('hierarchical_bayesian')
            
            # Store performance metrics
            if 'performance' in hb_results:
                self.training_state['performance_metrics']['hierarchical_bayesian'] = hb_results['performance']
            
            self.logger.info("✅ Hierarchical Bayesian Model training completed")
            current_progress += progress_step
            
        except Exception as e:
            self.logger.error(f"❌ Hierarchical Bayesian Model training failed: {str(e)}")
            self.training_state['errors'].append({
                'model': 'hierarchical_bayesian',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        # Train X-Learner Model
        try:
            self.logger.info("🎯 Training X-Learner (Causal ML) Model...")
            self._update_training_state("training", "x_learner", current_progress)
            
            xl_results = self.model_trainer.train_x_learner_model(X, y)
            all_results['x_learner'] = xl_results
            self.training_state['models_trained'].append('x_learner')
            
            if 'performance' in xl_results:
                self.training_state['performance_metrics']['x_learner'] = xl_results['performance']
            
            self.logger.info("✅ X-Learner Model training completed")
            current_progress += progress_step
            
        except Exception as e:
            self.logger.error(f"❌ X-Learner Model training failed: {str(e)}")
            self.training_state['errors'].append({
                'model': 'x_learner',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        # Train Ensemble Model
        try:
            self.logger.info("🤖 Training Ensemble Model...")
            self._update_training_state("training", "ensemble", current_progress)
            
            ensemble_results = self.model_trainer.train_ensemble_model(X, y)
            all_results['ensemble'] = ensemble_results
            self.training_state['models_trained'].append('ensemble')
            
            if 'performance' in ensemble_results:
                self.training_state['performance_metrics']['ensemble'] = ensemble_results['performance']
            
            self.logger.info("✅ Ensemble Model training completed")
            current_progress += progress_step
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble Model training failed: {str(e)}")
            self.training_state['errors'].append({
                'model': 'ensemble',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        # Compare models
        self._update_training_state("training", "model_comparison", 80)
        model_comparison = self.model_trainer._compare_model_performance(all_results)
        all_results['model_comparison'] = model_comparison
        
        self.logger.info(f"🏆 Best model: {model_comparison.get('best_model', 'Unknown')} "
                        f"(AUC: {model_comparison.get('best_auc', 0):.3f})")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results and models"""
        self.logger.info("💾 Saving training results...")
        self._update_training_state("saving", "save_models", 85)
        
        try:
            # Save models
            self.model_trainer.models = {k: v['model'] for k, v in results.items() if 'model' in v}
            self.model_trainer.training_results = results
            self.model_trainer.save_models()
            
            # Save comprehensive results
            output_file = f"outputs/reports/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert results to serializable format
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Results saved to {output_file}")
            self._update_training_state("completed", "save_complete", 100)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {str(e)}")
            self.training_state['errors'].append({
                'step': 'save_results',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() 
                   if k not in ['model', 'trace'] and not str(k).startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def run_full_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete model training pipeline
        
        Args:
            data_path: Path to training data
            
        Returns:
            Complete training results
        """
        self.logger.info("🚀 Starting Complete Model Training Pipeline")
        self.logger.info("=" * 60)
        
        self.training_state['start_time'] = datetime.now().isoformat()
        
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Feature engineering
            df_features = self.run_feature_engineering(df)
            
            # Train models
            results = self.train_models(df_features)
            
            # Save results
            self.save_results(results)
            
            # Final summary
            self._print_final_summary(results)
            
            self.training_state['status'] = 'completed'
            self.training_state['end_time'] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            self.training_state['status'] = 'failed'
            self.training_state['errors'].append({
                'step': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
        
        finally:
            # Save final state
            self._update_training_state(
                self.training_state['status'], 
                "pipeline_complete", 
                100
            )
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive training summary"""
        self.logger.info("=" * 60)
        self.logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60)
        
        # Models trained
        models_trained = self.training_state.get('models_trained', [])
        self.logger.info(f"📊 Models Trained: {len(models_trained)}")
        for model in models_trained:
            self.logger.info(f"   ✅ {model}")
        
        # Performance summary
        if 'model_comparison' in results:
            comparison = results['model_comparison']
            self.logger.info(f"🏆 Best Model: {comparison.get('best_model', 'Unknown')}")
            self.logger.info(f"🎯 Best AUC: {comparison.get('best_auc', 0):.3f}")
        
        # Performance metrics
        perf_metrics = self.training_state.get('performance_metrics', {})
        if perf_metrics:
            self.logger.info("📈 Performance Summary:")
            for model, metrics in perf_metrics.items():
                auc = metrics.get('auc', 0)
                acc = metrics.get('accuracy', 0)
                self.logger.info(f"   {model}: AUC={auc:.3f}, Accuracy={acc:.3f}")
        
        # Timing
        if self.training_state.get('start_time'):
            start_time = datetime.fromisoformat(self.training_state['start_time'])
            duration = datetime.now() - start_time
            self.logger.info(f"⏱️  Total Duration: {duration}")
        
        # Errors/Warnings
        errors = len(self.training_state.get('errors', []))
        warnings_count = len(self.training_state.get('warnings', []))
        if errors > 0:
            self.logger.warning(f"⚠️  {errors} errors encountered")
        if warnings_count > 0:
            self.logger.warning(f"⚠️  {warnings_count} warnings")
        
        self.logger.info("=" * 60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Price Elasticity Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (generates sample data)
    python model_training.py
    
    # Run with custom data file
    python model_training.py --data path/to/your/data.csv
    
    # Run with custom configuration
    python model_training.py --config config/custom_config.yaml
    
    # Run with debug logging
    python model_training.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        help='Path to training data file (CSV or Parquet)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 80)
    print("🚀 PRICE ELASTICITY MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Python: {sys.version.split()[0]}")
    print(f"📁 Working Directory: {os.getcwd()}")
    if args.data:
        print(f"📊 Data File: {args.data}")
    if args.config:
        print(f"⚙️  Config File: {args.config}")
    print("=" * 80)
    
    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline(
            config_path=args.config,
            log_level=args.log_level
        )
        
        # Run training
        results = pipeline.run_full_pipeline(data_path=args.data)
        
        print("\n🎉 Training completed successfully!")
        print("🌐 To view results in the UI, run:")
        print("   streamlit run inference.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

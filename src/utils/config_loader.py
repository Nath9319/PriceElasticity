"""
Configuration loader and logger utilities for Price Elasticity Modeling
"""

import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader for the price elasticity modeling system
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration parameters
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation like 'data.files.quote_history')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'models.{model_name}', {})
    
    def get_paths(self) -> Dict[str, str]:
        """
        Get all path configurations
        
        Returns:
            Dictionary of path configurations
        """
        return self.get('paths', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration
        
        Returns:
            Data configuration dictionary
        """
        return self.get('data', {})


class Logger:
    """
    Logger setup for the price elasticity modeling system
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize logger with configuration
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader
        self.setup_logging()
    
    def setup_logging(self):
        """
        Setup logging configuration based on config file
        """
        log_config = self.config.get('logging', {})
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=self._get_handlers(log_config)
        )
    
    def _get_handlers(self, log_config: Dict[str, Any]) -> list:
        """
        Get logging handlers based on configuration
        
        Args:
            log_config: Logging configuration
            
        Returns:
            List of logging handlers
        """
        handlers = []
        
        # Console handler
        if log_config.get('handlers', {}).get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_config.get('level', 'INFO')))
            handlers.append(console_handler)
        
        # File handler
        if log_config.get('handlers', {}).get('file', True):
            log_files = log_config.get('log_files', {})
            main_log_file = log_files.get('main', 'logs/price_elasticity.log')
            
            file_handler = logging.FileHandler(main_log_file)
            file_handler.setLevel(getattr(logging, log_config.get('level', 'INFO')))
            handlers.append(file_handler)
        
        return handlers
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


def create_directory_structure(base_path: str = "."):
    """
    Create the required directory structure for the project
    
    Args:
        base_path: Base path for the project
    """
    base_path = Path(base_path)
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "models/trained",
        "models/metadata",
        "results/predictions",
        "results/evaluations", 
        "results/explanations",
        "results/plots",
        "results/reports",
        "logs",
        "notebooks/eda",
        "notebooks/modeling",
        "notebooks/analysis"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print("Directory structure created successfully!")


def validate_data_files(config_loader: ConfigLoader) -> bool:
    """
    Validate that all required data files exist
    
    Args:
        config_loader: Configuration loader instance
        
    Returns:
        True if all files exist, False otherwise
    """
    data_config = config_loader.get_data_config()
    datasets_path = Path(data_config.get('datasets_path', 'datasets'))
    files = data_config.get('files', {})
    
    missing_files = []
    
    for file_key, filename in files.items():
        file_path = datasets_path / filename
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        return False
    
    print("All required data files found!")
    return True


# Global instances
config_loader = ConfigLoader()
logger_setup = Logger(config_loader)
logger = Logger.get_logger(__name__)

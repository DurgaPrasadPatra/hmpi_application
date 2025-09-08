# modules/utils.py
import pandas as pd
import numpy as np
import json
import yaml
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration Manager for HMPI Application
    
    Handles loading, validation, and management of application configurations
    including standards, thresholds, and display settings.
    """
    
    def __init__(self, config_file='config.yaml'):
        """Initialize with configuration file"""
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        default_config = self._get_default_config()
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                    # Merge with defaults
                    return self._deep_merge(default_config, loaded_config)
            else:
                logger.warning(f"Config file {self.config_file} not found. Using defaults.")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return default_config
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'app': {
                'title': 'Heavy Metal Pollution Indices Calculator',
                'version': '1.0.0',
                'author': 'Environmental Data Analytics Team',
                'max_upload_size': 200,
                'supported_formats': ['csv', 'xlsx', 'xls']
            },
            'standards': {
                'default_source': 'WHO',
                'sources': {
                    'WHO': {
                        'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
                        'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
                    },
                    'EPA': {
                        'As': 0.01, 'Cd': 0.005, 'Cr': 0.1, 'Cu': 1.3, 'Fe': 0.3,
                        'Mn': 0.05, 'Ni': 0.1, 'Pb': 0.015, 'Zn': 5.0
                    }
                }
            },
            'quality_categories': {
                'excellent': {'range': [0, 25], 'color': '#2E8B57'},
                'good': {'range': [25, 50], 'color': '#32CD32'},
                'poor': {'range': [50, 100], 'color': '#FFD700'},
                'very_poor': {'range': [100, 200], 'color': '#FF6347'},
                'unsuitable': {'range': [200, 1000], 'color': '#DC143C'}
            }
        }
    
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_standards(self, source=None):
        """Get heavy metal standards for specified source"""
        if source is None:
            source = self.get('standards.default_source', 'WHO')
        return self.get(f'standards.sources.{source}', {})
    
    def get_quality_categories(self):
        """Get quality category definitions"""
        return self.get('quality_categories', {})

class DataValidator:
    """
    Data Validation Utilities for HMPI Application
    
    Provides comprehensive validation for heavy metal concentration data,
    coordinates, and other input parameters.
    """
    
    def __init__(self):
        """Initialize validator with standard parameters"""
        self.heavy_metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
        self.required_columns = ['Sample_ID']
        
        # Realistic concentration ranges (mg/L)
        self.concentration_ranges = {
            'As': {'min': 0, 'max': 1.0, 'typical_max': 0.1},
            'Cd': {'min': 0, 'max': 0.5, 'typical_max': 0.05},
            'Cr': {'min': 0, 'max': 5.0, 'typical_max': 0.5},
            'Cu': {'min': 0, 'max': 10.0, 'typical_max': 5.0},
            'Fe': {'min': 0, 'max': 50.0, 'typical_max': 10.0},
            'Mn': {'min': 0, 'max': 20.0, 'typical_max': 5.0},
            'Ni': {'min': 0, 'max': 2.0, 'typical_max': 0.5},
            'Pb': {'min': 0, 'max': 1.0, 'typical_max': 0.1},
            'Zn': {'min': 0, 'max': 50.0, 'typical_max': 10.0}
        }
    
    def validate_dataset(self, df):
        """Validate entire dataset"""
        validation_report = {
            'is_valid': True,
            'total_samples': len(df),
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check for empty dataset
        if df.empty:
            validation_report['errors'].append("Dataset is empty")
            validation_report['is_valid'] = False
            return validation_report
        
        # Check for required metals (at least 2)
        available_metals = [col for col in df.columns if col in self.heavy_metals]
        if len(available_metals) < 2:
            validation_report['errors'].append("At least 2 heavy metals required for analysis")
            validation_report['is_valid'] = False
        
        # Generate summary statistics
        validation_report['summary'] = {
            'metals_available': available_metals,
            'samples_with_coordinates': len(df.dropna(subset=['Latitude', 'Longitude'])) if 'Latitude' in df.columns else 0,
            'completeness': self._calculate_completeness(df, available_metals)
        }
        
        return validation_report
    
    def _calculate_completeness(self, df, metals):
        """Calculate data completeness percentage"""
        if not metals:
            return 0
        
        total_values = len(df) * len(metals)
        missing_values = df[metals].isnull().sum().sum()
        completeness = ((total_values - missing_values) / total_values) * 100
        return completeness

class DataTransformer:
    """
    Data Transformation Utilities
    
    Handles data cleaning, normalization, and transformation operations
    for heavy metal concentration data.
    """
    
    def __init__(self):
        """Initialize transformer with standard parameters"""
        self.heavy_metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
    
    def standardize_column_names(self, df):
        """Standardize column names to consistent format"""
        column_mapping = {
            # Sample ID variations
            'sample_id': 'Sample_ID', 'sampleid': 'Sample_ID', 'id': 'Sample_ID',
            'sample': 'Sample_ID', 'well_id': 'Sample_ID', 'station_id': 'Sample_ID',
            
            # Metal name variations
            'arsenic': 'As', 'cadmium': 'Cd', 'chromium': 'Cr', 'copper': 'Cu',
            'iron': 'Fe', 'manganese': 'Mn', 'nickel': 'Ni', 'lead': 'Pb', 'zinc': 'Zn'
        }
        
        # Create mapping for current columns
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]
            elif col.upper() in self.heavy_metals:
                rename_dict[col] = col.upper()
        
        return df.rename(columns=rename_dict)
    
    def clean_numeric_data(self, df, columns):
        """Clean and convert numeric data"""
        cleaned_df = df.copy()
        
        for col in columns:
            if col in cleaned_df.columns:
                # Convert to numeric, handling various formats
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Handle negative values (set to zero)
                cleaned_df[col] = cleaned_df[col].clip(lower=0)
        
        return cleaned_df

class StatisticalUtils:
    """
    Statistical Utilities for Heavy Metal Analysis
    
    Provides statistical functions for data analysis, hypothesis testing,
    and descriptive statistics.
    """
    
    def __init__(self):
        """Initialize statistical utilities"""
        pass
    
    def descriptive_stats(self, data, columns):
        """Calculate comprehensive descriptive statistics"""
        stats_dict = {}
        
        for col in columns:
            if col in data.columns:
                col_data = data[col].dropna()
                
                if len(col_data) > 0:
                    stats_dict[col] = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75))
                    }
        
        return stats_dict
    
    def correlation_analysis(self, data, columns):
        """Perform correlation analysis between variables"""
        numeric_data = data[columns].select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return None
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        return {
            'pearson': pearson_corr,
            'variables': list(numeric_data.columns)
        }

# Simple utility functions that were being imported
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load application configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_manager = ConfigManager(config_path)
    return config_manager.config

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration
    
    Args:
        config: Application configuration dictionary
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_file_format(filename: str) -> bool:
    """
    Validate if file format is supported
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_extensions = ['.csv', '.xlsx', '.xls']
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in supported_extensions

def get_sample_data_path() -> str:
    """
    Get path to sample data directory
    
    Returns:
        Path string pointing to sample data directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), "data", "sample")
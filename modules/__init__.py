# modules/__init__.py
"""
Heavy Metal Pollution Indices Calculator - Core Modules

This package contains the core modules for the HMPI application:
- HMPI Calculator: Heavy metal pollution index calculations
- Data Processor: Data loading, validation, and preprocessing
- Visualization: Interactive charts and plotting utilities
- Report Generator: Comprehensive report generation
- Geospatial Analyzer: Spatial analysis and mapping
- Utils: Common utilities and helper functions
"""

from .hmpi_calculator import HMPICalculator
from .data_processor import DataProcessor
from .visualization import Visualizer
from .report_generator import ReportGenerator
from .geospatial_analyzer import GeospatialAnalyzer

# Import only the classes and functions that actually exist in utils.py
from .utils import (
    ConfigManager,
    DataValidator,
    DataTransformer,
    StatisticalUtils,
    load_config,
    setup_logging,
    validate_file_format,
    get_sample_data_path
)

__version__ = "1.0.0"
__author__ = "Environmental Data Analytics Team"
__email__ = "team@environmental-analytics.com"

# Package metadata - only include what actually exists
__all__ = [
    # Core classes
    "HMPICalculator",
    "DataProcessor",
    "Visualizer", 
    "ReportGenerator",
    "GeospatialAnalyzer",
    
    # Utility classes that actually exist
    "ConfigManager",
    "DataValidator",
    "DataTransformer",
    "StatisticalUtils",
    
    # Functions that actually exist
    "load_config",
    "setup_logging",
    "validate_file_format",
    "get_sample_data_path"
]

# Module information
MODULES_INFO = {
    "hmpi_calculator": {
        "description": "Heavy Metal Pollution Index calculations",
        "main_class": "HMPICalculator",
        "key_features": ["HMPI", "PLI", "Degree of Contamination", "Nemerow Index"]
    },
    "data_processor": {
        "description": "Data loading, validation, and preprocessing",
        "main_class": "DataProcessor",
        "key_features": ["Multi-format loading", "Data validation", "Preprocessing", "Sample generation"]
    },
    "visualization": {
        "description": "Interactive charts and plotting utilities", 
        "main_class": "Visualizer",
        "key_features": ["Interactive plots", "Statistical charts", "Correlation analysis", "Export functions"]
    },
    "report_generator": {
        "description": "Comprehensive report generation",
        "main_class": "ReportGenerator", 
        "key_features": ["Multi-section reports", "Multiple formats", "Statistical summaries", "Recommendations"]
    },
    "geospatial_analyzer": {
        "description": "Spatial analysis and mapping",
        "main_class": "GeospatialAnalyzer",
        "key_features": ["Interactive maps", "Spatial clustering", "Hotspot analysis", "Spatial statistics"]
    },
    "utils": {
        "description": "Common utilities and helper functions",
        "main_classes": ["ConfigManager", "DataValidator", "DataTransformer", "StatisticalUtils"],
        "key_features": ["Configuration management", "Data validation", "File utilities", "Statistical analysis"]
    }
}

def get_module_info():
    """Get information about all modules in the package"""
    return MODULES_INFO

def get_version():
    """Get package version"""
    return __version__

def list_available_classes():
    """List all available classes in the package"""
    return __all__
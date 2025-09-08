# data_processor.py
# modules/data_processor.py
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data Processing Module for Heavy Metal Data
    
    Handles data loading, validation, cleaning, and preprocessing
    for heavy metal pollution analysis.
    """
    
    def __init__(self):
        """Initialize with required column mappings and validation rules"""
        self.required_metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
        self.optional_columns = ['Sample_ID', 'Latitude', 'Longitude', 'Location', 'Date_Collected']
        
        # Common column name mappings
        self.column_mappings = {
            'sample_id': 'Sample_ID',
            'sampleid': 'Sample_ID',
            'id': 'Sample_ID',
            'sample': 'Sample_ID',
            'lat': 'Latitude',
            'latitude': 'Latitude',
            'lon': 'Longitude',
            'lng': 'Longitude',
            'longitude': 'Longitude',
            'long': 'Longitude',
            'location': 'Location',
            'site': 'Location',
            'place': 'Location',
            'date': 'Date_Collected',
            'date_collected': 'Date_Collected',
            'sampling_date': 'Date_Collected',
            'collection_date': 'Date_Collected'
        }
        
        # Metal symbol mappings
        self.metal_mappings = {
            'arsenic': 'As',
            'cadmium': 'Cd',
            'chromium': 'Cr',
            'copper': 'Cu',
            'iron': 'Fe',
            'manganese': 'Mn',
            'nickel': 'Ni',
            'lead': 'Pb',
            'zinc': 'Zn'
        }
    
    def load_data(self, file_input):
        """
        Load data from uploaded file (CSV, Excel)
        
        Args:
            file_input: Streamlit uploaded file object or file path
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Determine file type and load accordingly
            if hasattr(file_input, 'name'):
                filename = file_input.name.lower()
            else:
                filename = str(file_input).lower()
            
            if filename.endswith('.csv'):
                # Try different encodings for CSV
                try:
                    # Handle both file-like objects and file paths
                    if hasattr(file_input, 'seek'):
                        file_input.seek(0)  # Reset file pointer if it's a file object
                    df = pd.read_csv(file_input, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        if hasattr(file_input, 'seek'):
                            file_input.seek(0)
                        df = pd.read_csv(file_input, encoding='latin-1')
                    except UnicodeDecodeError:
                        if hasattr(file_input, 'seek'):
                            file_input.seek(0)
                        df = pd.read_csv(file_input, encoding='cp1252')
                        
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_input)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Basic data info
            print(f"Loaded data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self, df):
        """
        Validate the loaded data for heavy metal analysis
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            dict: Validation results with errors and warnings
        """
        errors = []
        validation_warnings = []  # Renamed to avoid conflict with warnings module
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("Dataset is empty")
            return {'is_valid': False, 'errors': errors, 'warnings': validation_warnings}
        
        # Check for minimum required columns
        df_columns_lower = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Check for at least some heavy metals
        found_metals = []
        for metal in self.required_metals:
            if metal in df.columns:
                found_metals.append(metal)
            elif metal.lower() in df_columns_lower:
                # Find the original column name that matches
                idx = df_columns_lower.index(metal.lower())
                original_col = list(df.columns)[idx]
                found_metals.append(metal)
            else:
                # Check for full metal names
                for full_name, symbol in self.metal_mappings.items():
                    if full_name in df_columns_lower and symbol == metal:
                        found_metals.append(metal)
                        break
        
        if len(found_metals) < 2:
            errors.append("At least 2 heavy metals must be present in the data")
        elif len(found_metals) < 5:
            validation_warnings.append(f"Only {len(found_metals)} metals found. More metals provide better assessment")
        
        # Check for coordinate columns if geospatial analysis is needed
        has_coordinates = False
        for lat_col in ['latitude', 'lat']:
            for lon_col in ['longitude', 'lon', 'lng', 'long']:
                if lat_col in df_columns_lower and lon_col in df_columns_lower:
                    has_coordinates = True
                    break
            if has_coordinates:
                break
        
        if not has_coordinates:
            validation_warnings.append("No coordinate columns found. Geospatial analysis will not be available")
        
        # Check for negative values in metal concentrations
        for metal in found_metals:
            metal_col = None
            # Find the actual column name for this metal
            if metal in df.columns:
                metal_col = metal
            else:
                # Look for it in original columns
                for col in df.columns:
                    if col.lower().replace(' ', '_') == metal.lower():
                        metal_col = col
                        break
                    # Check metal name mappings
                    for full_name, symbol in self.metal_mappings.items():
                        if col.lower().replace(' ', '_') == full_name and symbol == metal:
                            metal_col = col
                            break
            
            if metal_col:
                # Check if column is object/string type first
                if pd.api.types.is_object_dtype(df[metal_col]):
                    # Try to convert to numeric to see if it contains non-numeric values
                    try:
                        numeric_values = pd.to_numeric(df[metal_col], errors='coerce')
                        if numeric_values.isnull().any() and not df[metal_col].isnull().all():
                            errors.append(f"{metal} column contains non-numeric values")
                    except:
                        errors.append(f"{metal} column contains non-numeric values")
                elif pd.api.types.is_numeric_dtype(df[metal_col]):
                    if (df[metal_col] < 0).any():
                        errors.append(f"{metal} column contains negative values")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing_cols = missing_data[missing_data > len(df) * 0.5].index.tolist()
            if high_missing_cols:
                validation_warnings.append(f"High missing values (>50%) in columns: {high_missing_cols}")
        
        # Check data types for coordinate columns if they exist
        coord_columns = ['Latitude', 'Longitude']
        for col in coord_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except:
                    errors.append(f"Column {col} cannot be converted to numeric")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': validation_warnings,
            'found_metals': found_metals,
            'has_coordinates': has_coordinates
        }
    
    def preprocess_data(self, df):
        """
        Clean and preprocess the data for analysis
        
        Args:
            df (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed data
        """
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Standardize column names
        processed_df = self._standardize_column_names(processed_df)
        
        # Handle missing Sample_ID
        if 'Sample_ID' not in processed_df.columns:
            processed_df['Sample_ID'] = [f'SAMPLE_{i+1:03d}' for i in range(len(processed_df))]
        
        # Clean and convert metal concentrations
        metal_columns = [col for col in processed_df.columns if col in self.required_metals]
        
        for metal in metal_columns:
            # Convert to numeric, handling any string values
            processed_df[metal] = pd.to_numeric(processed_df[metal], errors='coerce')
            
            # Handle negative values (set to 0 or small positive value)
            processed_df.loc[processed_df[metal] < 0, metal] = 0
            
            # Handle missing values (fill with detection limit)
            detection_limit = self._get_detection_limit(metal)
            processed_df[metal] = processed_df[metal].fillna(detection_limit)
        
        # Clean coordinate columns
        if 'Latitude' in processed_df.columns and 'Longitude' in processed_df.columns:
            processed_df['Latitude'] = pd.to_numeric(processed_df['Latitude'], errors='coerce')
            processed_df['Longitude'] = pd.to_numeric(processed_df['Longitude'], errors='coerce')
            
            # Remove obviously invalid coordinates
            valid_coords_mask = (
                processed_df['Latitude'].between(-90, 90, na=False) & 
                processed_df['Longitude'].between(-180, 180, na=False)
            )
            processed_df = processed_df[valid_coords_mask]
        
        # Handle date columns
        if 'Date_Collected' in processed_df.columns:
            processed_df['Date_Collected'] = pd.to_datetime(
                processed_df['Date_Collected'], 
                errors='coerce'
            )
        
        # Remove rows with all missing metal data
        if metal_columns:  # Only if there are metal columns
            processed_df = processed_df.dropna(subset=metal_columns, how='all')
        
        # Add data quality flags
        if metal_columns:  # Only if there are metal columns
            processed_df = self._add_quality_flags(processed_df, metal_columns)
        
        return processed_df
    
    def _standardize_column_names(self, df):
        """Standardize column names based on common mappings"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            
            # Check direct mappings
            if col_lower in self.column_mappings:
                column_mapping[col] = self.column_mappings[col_lower]
            
            # Check metal mappings
            elif col_lower in self.metal_mappings:
                column_mapping[col] = self.metal_mappings[col_lower]
            
            # Check if it's already a standard metal symbol
            elif col.upper() in self.required_metals:
                column_mapping[col] = col.upper()
        
        return df.rename(columns=column_mapping)
    
    def _get_detection_limit(self, metal):
        """Get typical detection limit for each metal (mg/L)"""
        detection_limits = {
            'As': 0.0001,
            'Cd': 0.0001,
            'Cr': 0.001,
            'Cu': 0.001,
            'Fe': 0.01,
            'Mn': 0.001,
            'Ni': 0.001,
            'Pb': 0.0001,
            'Zn': 0.001
        }
        return detection_limits.get(metal, 0.001)
    
    def _add_quality_flags(self, df, metal_columns):
        """Add data quality flags to identify potential issues"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Flag for samples with high number of missing metals
        missing_metals = df[metal_columns].isnull().sum(axis=1)
        df['Missing_Metals_Count'] = missing_metals
        df['High_Missing_Flag'] = missing_metals > len(metal_columns) * 0.3
        
        # Flag for samples with extremely high concentrations
        high_conc_flags = []
        for metal in metal_columns:
            # Define "extremely high" as 100x typical environmental levels
            typical_levels = {
                'As': 0.05, 'Cd': 0.01, 'Cr': 0.1, 'Cu': 1.0, 'Fe': 2.0,
                'Mn': 0.5, 'Ni': 0.02, 'Pb': 0.05, 'Zn': 5.0
            }
            threshold = typical_levels.get(metal, 1.0) * 100  # 100x typical levels
            high_conc_flags.append(df[metal] > threshold)
        
        # Combine all high concentration flags
        if high_conc_flags:
            df['High_Concentration_Flag'] = pd.concat(high_conc_flags, axis=1).any(axis=1)
        else:
            df['High_Concentration_Flag'] = False
        
        # Overall data quality score (0-100)
        quality_score = 100 - (df['Missing_Metals_Count'] * 5)  # -5 points per missing metal
        quality_score = quality_score - (df['High_Missing_Flag'].astype(int) * 20)  # -20 points for high missing flag
        df['Data_Quality_Score'] = quality_score.clip(0, 100)
        
        return df
    
    def generate_sample_data(self, n_samples=50):
        """
        Generate sample data for testing purposes
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated sample data
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate sample IDs
        sample_ids = [f'SAMPLE_{i+1:03d}' for i in range(n_samples)]
        
        # Generate coordinates (around a hypothetical study area)
        base_lat, base_lon = 13.0827, 80.2707  # Chennai coordinates as example
        latitudes = base_lat + np.random.normal(0, 0.1, n_samples)
        longitudes = base_lon + np.random.normal(0, 0.1, n_samples)
        
        # Generate locations
        locations = [f'Site_{i+1}' for i in range(n_samples)]
        
        # Generate dates (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
        
        # Generate heavy metal concentrations with realistic distributions
        metal_data = {}
        
        # Define realistic concentration ranges and distributions
        metal_params = {
            'As': {'mean': 0.005, 'std': 0.5, 'contaminated_mean': 0.05},
            'Cd': {'mean': 0.001, 'std': 0.5, 'contaminated_mean': 0.01},
            'Cr': {'mean': 0.02, 'std': 0.5, 'contaminated_mean': 0.1},
            'Cu': {'mean': 0.5, 'std': 0.5, 'contaminated_mean': 3.0},
            'Fe': {'mean': 0.1, 'std': 0.5, 'contaminated_mean': 1.0},
            'Mn': {'mean': 0.05, 'std': 0.5, 'contaminated_mean': 0.8},
            'Ni': {'mean': 0.01, 'std': 0.5, 'contaminated_mean': 0.1},
            'Pb': {'mean': 0.005, 'std': 0.5, 'contaminated_mean': 0.05},
            'Zn': {'mean': 1.0, 'std': 0.5, 'contaminated_mean': 5.0}
        }
        
        # Create mix of clean and contaminated samples
        contaminated_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
        
        for metal, params in metal_params.items():
            concentrations = np.zeros(n_samples)
            
            # Generate clean samples
            clean_mask = ~np.isin(range(n_samples), contaminated_indices)
            n_clean = np.sum(clean_mask)
            if n_clean > 0:
                # Lognormal distribution naturally produces positive values only
                concentrations[clean_mask] = np.random.lognormal(
                    np.log(params['mean']), params['std'], n_clean
                )
            
            # Generate contaminated samples
            contaminated_mask = np.isin(range(n_samples), contaminated_indices)
            n_contaminated = np.sum(contaminated_mask)
            if n_contaminated > 0:
                # Lognormal distribution naturally produces positive values only
                concentrations[contaminated_mask] = np.random.lognormal(
                    np.log(params['contaminated_mean']), params['std'], n_contaminated
                )
            
            # Ensure no values below detection limits
            concentrations = np.maximum(concentrations, self._get_detection_limit(metal))
            metal_data[metal] = concentrations
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'Sample_ID': sample_ids,
            'Latitude': latitudes,
            'Longitude': longitudes,
            'Location': locations,
            'Date_Collected': dates,
            **metal_data
        })
        
        return sample_data
    
    def export_template(self):
        """
        Generate a template file for data input
        
        Returns:
            pd.DataFrame: Template with proper column names and sample data
        """
        # Create sample rows
        sample_data = {
            'Sample_ID': ['SAMPLE_001', 'SAMPLE_002'],
            'Latitude': [13.0827, 13.0850],
            'Longitude': [80.2707, 80.2750],
            'Location': ['Site_A', 'Site_B'],
            'Date_Collected': ['2024-01-15', '2024-01-16'],
            'As': [0.005, 0.003],
            'Cd': [0.001, 0.002],
            'Cr': [0.02, 0.015],
            'Cu': [0.5, 0.3],
            'Fe': [0.1, 0.2],
            'Mn': [0.05, 0.08],
            'Ni': [0.01, 0.012],
            'Pb': [0.005, 0.004],
            'Zn': [1.0, 0.8]
        }
        
        template = pd.DataFrame(sample_data)
        return template
    
    def get_data_summary(self, df):
        """Generate comprehensive data summary"""
        if df is None or df.empty:
            return "No data available"
        
        metal_columns = [col for col in df.columns if col in self.required_metals]
        
        summary = {
            'total_samples': len(df),
            'total_metals': len(metal_columns),
            'metals_present': metal_columns,
            'date_range': None,
            'coordinate_coverage': None,
            'data_completeness': {}
        }
        
        # Date range analysis
        if 'Date_Collected' in df.columns:
            valid_dates = pd.to_datetime(df['Date_Collected'], errors='coerce').dropna()
            if not valid_dates.empty:
                summary['date_range'] = {
                    'earliest': valid_dates.min().strftime('%Y-%m-%d'),
                    'latest': valid_dates.max().strftime('%Y-%m-%d'),
                    'span_days': (valid_dates.max() - valid_dates.min()).days
                }
        
        # Coordinate coverage analysis
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            valid_coords = df[['Latitude', 'Longitude']].dropna()
            if not valid_coords.empty:
                summary['coordinate_coverage'] = {
                    'samples_with_coordinates': len(valid_coords),
                    'lat_range': [float(valid_coords['Latitude'].min()), float(valid_coords['Latitude'].max())],
                    'lon_range': [float(valid_coords['Longitude'].min()), float(valid_coords['Longitude'].max())],
                }
        
        # Data completeness analysis
        for metal in metal_columns:
            total_values = len(df)
            missing_values = df[metal].isnull().sum()
            summary['data_completeness'][metal] = {
                'completeness_percentage': ((total_values - missing_values) / total_values) * 100,
                'missing_count': int(missing_values),
                'mean_concentration': float(df[metal].mean()) if not df[metal].isnull().all() else None,
                'max_concentration': float(df[metal].max()) if not df[metal].isnull().all() else None
            }
        
        return summary
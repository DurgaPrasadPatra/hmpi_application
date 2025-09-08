# hmpi_calculator.py
# modules/hmpi_calculator.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class HMPICalculator:
    """
    Heavy Metal Pollution Index Calculator
    
    This class implements various heavy metal pollution indices calculation
    methods based on standard formulas and WHO/EPA guidelines.
    """
    
    def __init__(self):
        """Initialize with WHO/EPA standard values for heavy metals in drinking water (mg/L)"""
        self.standard_values = {
            'As': 0.01,   # Arsenic
            'Cd': 0.003,  # Cadmium
            'Cr': 0.05,   # Chromium
            'Cu': 2.0,    # Copper
            'Fe': 0.3,    # Iron
            'Mn': 0.4,    # Manganese
            'Ni': 0.07,   # Nickel
            'Pb': 0.01,   # Lead
            'Zn': 3.0     # Zinc
        }
        
        # Weight factors for different metals based on toxicity
        self.weight_factors = {
            'As': 1.0,    # Maximum weight for highly toxic metals
            'Cd': 1.0,    # Maximum weight for highly toxic metals
            'Cr': 0.8,    # High weight
            'Cu': 0.6,    # Medium weight
            'Fe': 0.4,    # Lower weight (essential mineral)
            'Mn': 0.4,    # Lower weight (essential mineral)
            'Ni': 0.7,    # Medium-high weight
            'Pb': 1.0,    # Maximum weight for highly toxic metals
            'Zn': 0.5     # Lower weight (essential mineral)
        }
    
    def calculate_hmpi(self, data):
        """
        Calculate Heavy Metal Pollution Index (HMPI) for groundwater samples
        
        HMPI Formula:
        HMPI = Σ(Wi × Qi) / Σ(Wi)
        
        Where:
        Wi = Weight of ith metal
        Qi = Quality rating of ith metal = (Ci / Si) × 100
        Ci = Concentration of ith metal
        Si = Standard value of ith metal
        
        Args:
            data (pd.DataFrame): DataFrame containing heavy metal concentrations
            
        Returns:
            pd.DataFrame: DataFrame with calculated HMPI and quality categories
        """
        results = data.copy()
        
        # Available metals in the dataset
        available_metals = [col for col in data.columns 
                          if col in self.standard_values.keys()]
        
        if not available_metals:
            raise ValueError("No recognized heavy metals found in the dataset")
        
        # Calculate quality ratings (Qi) for each metal
        quality_ratings = {}
        for metal in available_metals:
            if metal in data.columns:
                Ci = data[metal]  # Concentration
                Si = self.standard_values[metal]  # Standard value
                Qi = (Ci / Si) * 100  # Quality rating
                quality_ratings[metal] = Qi
                results[f'{metal}_Qi'] = Qi
        
        # Calculate HMPI
        hmpi_numerator = np.zeros(len(data))
        hmpi_denominator = 0
        
        for metal in available_metals:
            if metal in quality_ratings:
                Wi = self.weight_factors.get(metal, 1.0)  # Weight factor
                Qi = quality_ratings[metal]  # Quality rating
                
                hmpi_numerator += Wi * Qi
                hmpi_denominator += Wi
        
        # Final HMPI calculation
        results['HMPI'] = hmpi_numerator / hmpi_denominator if hmpi_denominator > 0 else 0
        
        # Add quality categories based on HMPI values
        results['Quality_Category'] = results['HMPI'].apply(self._categorize_quality)
        results['Risk_Level'] = results['HMPI'].apply(self._assess_risk_level)
        
        # Calculate individual metal pollution indices
        results = self._calculate_individual_indices(results, available_metals)
        
        return results
    
    def _categorize_quality(self, hmpi_value):
        """
        Categorize water quality based on HMPI value
        
        HMPI Categories:
        - Excellent: ≤ 25
        - Good: 25-50
        - Poor: 50-100
        - Very Poor: 100-200
        - Unsuitable: > 200
        """
        if hmpi_value <= 25:
            return 'Excellent'
        elif hmpi_value <= 50:
            return 'Good'
        elif hmpi_value <= 100:
            return 'Poor'
        elif hmpi_value <= 200:
            return 'Very Poor'
        else:
            return 'Unsuitable'
    
    def _assess_risk_level(self, hmpi_value):
        """Assess health risk level based on HMPI"""
        if hmpi_value <= 25:
            return 'No Risk'
        elif hmpi_value <= 50:
            return 'Low Risk'
        elif hmpi_value <= 100:
            return 'Moderate Risk'
        elif hmpi_value <= 200:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def _calculate_individual_indices(self, data, available_metals):
        """Calculate individual pollution indices for each metal"""
        for metal in available_metals:
            if metal in data.columns:
                # Metal Pollution Index (MPI)
                concentration = data[metal]
                standard = self.standard_values[metal]
                
                # Individual Metal Index (IMI)
                data[f'{metal}_IMI'] = concentration / standard
                
                # Metal Contamination Factor (CF)
                data[f'{metal}_CF'] = concentration / standard
                
                # Add metal-specific quality category
                data[f'{metal}_Category'] = (concentration / standard).apply(
                    lambda x: 'Safe' if x <= 1 else 'Moderate' if x <= 3 else 'High' if x <= 5 else 'Severe'
                )
        
        return data
    
    def calculate_degree_of_contamination(self, data):
        """
        Calculate Degree of Contamination (Cd)
        Cd = Σ(Cf) where Cf is contamination factor for each metal
        """
        available_metals = [col for col in data.columns 
                          if col in self.standard_values.keys()]
        
        contamination_factors = []
        for metal in available_metals:
            if metal in data.columns:
                cf = data[metal] / self.standard_values[metal]
                contamination_factors.append(cf)
        
        if contamination_factors:
            degree_of_contamination = sum(contamination_factors)
            return degree_of_contamination
        else:
            return pd.Series([0] * len(data))
    
    def calculate_pollution_load_index(self, data):
        """
        Calculate Pollution Load Index (PLI)
        PLI = (CF1 × CF2 × ... × CFn)^(1/n)
        where CF is contamination factor and n is number of metals
        """
        available_metals = [col for col in data.columns 
                          if col in self.standard_values.keys()]
        
        contamination_factors = []
        for metal in available_metals:
            if metal in data.columns:
                cf = data[metal] / self.standard_values[metal]
                contamination_factors.append(cf)
        
        if contamination_factors:
            # Calculate geometric mean
            cf_product = np.prod(contamination_factors, axis=0)
            pli = np.power(cf_product, 1.0 / len(contamination_factors))
            return pli
        else:
            return pd.Series([0] * len(data))
    
    def calculate_nemerow_index(self, data):
        """
        Calculate Nemerow Pollution Index (NPI)
        NPI = √[(Piave^2 + Pimax^2) / 2]
        where Piave is average and Pimax is maximum single factor index
        """
        available_metals = [col for col in data.columns 
                          if col in self.standard_values.keys()]
        
        pollution_indices = []
        for metal in available_metals:
            if metal in data.columns:
                pi = data[metal] / self.standard_values[metal]
                pollution_indices.append(pi)
        
        if pollution_indices:
            pollution_df = pd.DataFrame(pollution_indices).T
            pi_ave = pollution_df.mean(axis=1)
            pi_max = pollution_df.max(axis=1)
            
            npi = np.sqrt((pi_ave**2 + pi_max**2) / 2)
            return npi
        else:
            return pd.Series([0] * len(data))
    
    def calculate_comprehensive_indices(self, data):
        """Calculate all pollution indices"""
        results = data.copy()
        
        # Calculate primary HMPI
        results = self.calculate_hmpi(results)
        
        # Calculate additional indices
        results['Degree_of_Contamination'] = self.calculate_degree_of_contamination(data)
        results['Pollution_Load_Index'] = self.calculate_pollution_load_index(data)
        results['Nemerow_Index'] = self.calculate_nemerow_index(data)
        
        # Add comprehensive assessment
        results['Overall_Assessment'] = results.apply(self._comprehensive_assessment, axis=1)
        
        return results
    
    def _comprehensive_assessment(self, row):
        """Provide comprehensive assessment based on multiple indices"""
        hmpi = row.get('HMPI', 0)
        pli = row.get('Pollution_Load_Index', 0)
        npi = row.get('Nemerow_Index', 0)
        
        # Weight-based scoring
        score = (hmpi * 0.4 + pli * 30 * 0.3 + npi * 30 * 0.3)
        
        if score <= 25:
            return 'Pristine'
        elif score <= 50:
            return 'Slightly Polluted'
        elif score <= 100:
            return 'Moderately Polluted'
        elif score <= 200:
            return 'Heavily Polluted'
        else:
            return 'Extremely Polluted'
    
    def get_metal_standards(self):
        """Return current standard values"""
        return self.standard_values.copy()
    
    def update_standards(self, new_standards):
        """Update standard values"""
        self.standard_values.update(new_standards)
    
    def validate_data_for_calculation(self, data):
        """Validate data before HMPI calculation"""
        errors = []
        warnings = []
        
        # Check for required columns
        available_metals = [col for col in data.columns 
                          if col in self.standard_values.keys()]
        
        if not available_metals:
            errors.append("No recognized heavy metals found in dataset")
        
        if len(available_metals) < 3:
            warnings.append("Less than 3 metals available. Results may be less reliable")
        
        # Check for negative values
        for metal in available_metals:
            if (data[metal] < 0).any():
                errors.append(f"Negative values found in {metal} column")
        
        # Check for extremely high values (potential data entry errors)
        for metal in available_metals:
            max_val = data[metal].max()
            threshold = self.standard_values[metal] * 1000  # 1000x standard
            if max_val > threshold:
                warnings.append(f"Extremely high {metal} value detected: {max_val:.6f} mg/L")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'available_metals': available_metals
        }
    
    def generate_calculation_summary(self, results):
        """Generate summary statistics of calculations"""
        available_metals = [col for col in results.columns 
                          if col in self.standard_values.keys()]
        
        summary = {
            'total_samples': len(results),
            'metals_analyzed': available_metals,
            'hmpi_statistics': {
                'mean': float(results['HMPI'].mean()),
                'median': float(results['HMPI'].median()),
                'std': float(results['HMPI'].std()),
                'min': float(results['HMPI'].min()),
                'max': float(results['HMPI'].max())
            },
            'quality_distribution': results['Quality_Category'].value_counts().to_dict(),
            'contaminated_samples': int((results['HMPI'] > 100).sum()),
            'safe_samples': int((results['HMPI'] <= 100).sum()),
            'contamination_percentage': float((results['HMPI'] > 100).mean() * 100)
        }
        
        # Add metal-specific statistics
        metal_stats = {}
        for metal in available_metals:
            if metal in results.columns:
                metal_stats[metal] = {
                    'mean_concentration': float(results[metal].mean()),
                    'max_concentration': float(results[metal].max()),
                    'standard_exceedances': int((results[metal] > self.standard_values[metal]).sum()),
                    'exceedance_percentage': float((results[metal] > self.standard_values[metal]).mean() * 100)
                }
        
        summary['metal_statistics'] = metal_stats
        
        return summary
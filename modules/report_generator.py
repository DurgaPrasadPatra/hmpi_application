# report_generator.py
# modules/report_generator.py
import pandas as pd
import numpy as np
import json
from datetime import datetime
import base64
import io
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

class ReportGenerator:
    """
    Report Generation Module for Heavy Metal Pollution Analysis
    
    Creates comprehensive reports in various formats including
    HTML, Markdown, and JSON summaries.
    """
    
    def __init__(self):
        """Initialize with report templates and configurations"""
        self.report_templates = {
            'executive_summary': self._get_executive_summary_template(),
            'methodology': self._get_methodology_template(),
            'results_analysis': self._get_results_analysis_template(),
            'quality_assessment': self._get_quality_assessment_template(),
            'statistical_analysis': self._get_statistical_analysis_template(),
            'geospatial_analysis': self._get_geospatial_analysis_template(),
            'conclusions': self._get_conclusions_template(),
            'recommendations': self._get_recommendations_template()
        }
    
    def generate_report(self, data, config):
        """
        Generate comprehensive report
        
        Args:
            data (pd.DataFrame): Analysis results data
            config (dict): Report configuration
            
        Returns:
            str: Generated report content
        """
        report_sections = []
        
        # Report header
        report_sections.append(self._generate_header(config))
        
        # Generate each selected section
        for section in config.get('sections', []):
            if section in self.report_templates:
                section_content = self._generate_section(section, data, config)
                report_sections.append(section_content)
        
        # Combine all sections
        full_report = '\n\n'.join(report_sections)
        
        return full_report
    
    def _generate_header(self, config):
        """Generate report header"""
        header = f"""# {config.get('title', 'Heavy Metal Pollution Assessment Report')}

**Organization:** {config.get('organization', 'Environmental Monitoring Agency')}  
**Author:** {config.get('author', 'Data Analyst')}  
**Date:** {config.get('date', datetime.now().strftime('%Y-%m-%d'))}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
        return header
    
    def _generate_section(self, section_name, data, config):
        """Generate individual report section"""
        try:
            if section_name == 'Executive Summary':
                return self._generate_executive_summary(data, config)
            elif section_name == 'Methodology':
                return self._generate_methodology(data, config)
            elif section_name == 'Results Analysis':
                return self._generate_results_analysis(data, config)
            elif section_name == 'Quality Assessment':
                return self._generate_quality_assessment(data, config)
            elif section_name == 'Statistical Analysis':
                return self._generate_statistical_analysis(data, config)
            elif section_name == 'Geospatial Analysis':
                return self._generate_geospatial_analysis(data, config)
            elif section_name == 'Conclusions':
                return self._generate_conclusions(data, config)
            elif section_name == 'Recommendations':
                return self._generate_recommendations(data, config)
            else:
                return f"## {section_name}\n\nSection content not available."
        except Exception as e:
            return f"## {section_name}\n\nError generating section: {str(e)}"
    
    def _generate_executive_summary(self, data, config):
        """Generate executive summary section"""
        total_samples = len(data)
        contaminated = (data['HMPI'] > 100).sum()
        contamination_rate = (contaminated / total_samples) * 100
        avg_hmpi = data['HMPI'].mean()
        max_hmpi = data['HMPI'].max()
        
        # Quality distribution
        quality_dist = data['Quality_Category'].value_counts()
        
        # Most problematic metals
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        metal_issues = []
        
        # Standard values for comparison
        standards = {
            'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
            'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
        }
        
        for metal in metals:
            if metal in data.columns:
                exceeded = (data[metal] > standards.get(metal, 1.0)).sum()
                if exceeded > 0:
                    metal_issues.append(f"{metal}: {exceeded} samples ({exceeded/total_samples*100:.1f}%)")
        
        summary = f"""## Executive Summary

This report presents the analysis of heavy metal pollution in groundwater samples using the Heavy Metal Pollution Index (HMPI) methodology. The assessment covers {total_samples} groundwater samples analyzed for heavy metal contamination.

### Key Findings

- **Total Samples Analyzed:** {total_samples}
- **Average HMPI:** {avg_hmpi:.2f}
- **Maximum HMPI:** {max_hmpi:.2f}
- **Contaminated Samples:** {contaminated} ({contamination_rate:.1f}%)
- **Safe Samples:** {total_samples - contaminated} ({100 - contamination_rate:.1f}%)

### Quality Distribution
"""
        
        for category, count in quality_dist.items():
            percentage = (count / total_samples) * 100
            summary += f"- **{category}:** {count} samples ({percentage:.1f}%)\n"
        
        if metal_issues:
            summary += f"\n### Standards Exceedances\n"
            for issue in metal_issues[:5]:  # Top 5 issues
                summary += f"- {issue}\n"
        
        if contamination_rate > 50:
            summary += f"\n⚠️ **Critical Alert:** {contamination_rate:.1f}% of samples show contamination (HMPI > 100). Immediate action required."
        elif contamination_rate > 25:
            summary += f"\n⚠️ **Warning:** {contamination_rate:.1f}% of samples show contamination. Monitoring and remediation recommended."
        else:
            summary += f"\n✅ **Status:** {contamination_rate:.1f}% contamination rate is within acceptable limits."
        
        return summary
    
    def _generate_methodology(self, data, config):
        """Generate methodology section"""
        metals_analyzed = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        
        methodology = f"""## Methodology

### Heavy Metal Pollution Index (HMPI)

The Heavy Metal Pollution Index (HMPI) is calculated using the following formula:

```
HMPI = Σ(Wi × Qi) / Σ(Wi)
```

Where:
- Wi = Weight of ith metal based on toxicity
- Qi = Quality rating of ith metal = (Ci / Si) × 100
- Ci = Concentration of ith metal in the sample
- Si = Standard permissible value of ith metal

### Standards Used

The analysis uses WHO/EPA drinking water standards:

| Metal | Standard (mg/L) | Full Name |
|-------|----------------|-----------|
| As    | 0.01           | Arsenic   |
| Cd    | 0.003          | Cadmium   |
| Cr    | 0.05           | Chromium  |
| Cu    | 2.0            | Copper    |
| Fe    | 0.3            | Iron      |
| Mn    | 0.4            | Manganese |
| Ni    | 0.07           | Nickel    |
| Pb    | 0.01           | Lead      |
| Zn    | 3.0            | Zinc      |

### Quality Categories

Based on HMPI values, groundwater quality is classified as:

- **Excellent (≤25):** Safe for drinking with no treatment
- **Good (25-50):** Safe for drinking with minimal treatment
- **Poor (50-100):** Requires treatment before consumption
- **Very Poor (100-200):** Extensive treatment required
- **Unsuitable (>200):** Not suitable for drinking purposes

### Metals Analyzed in This Study

{len(metals_analyzed)} heavy metals were analyzed: {', '.join(metals_analyzed)}

### Sample Information

- **Total Samples:** {len(data)}
- **Analysis Period:** {config.get('date', 'Not specified')}
- **Sampling Method:** Random/Systematic sampling from study area
- **Quality Control:** Duplicate analysis and blank samples included
"""
        return methodology
    
    def _generate_results_analysis(self, data, config):
        """Generate results analysis section"""
        total_samples = len(data)
        hmpi_stats = {
            'mean': data['HMPI'].mean(),
            'median': data['HMPI'].median(),
            'std': data['HMPI'].std(),
            'min': data['HMPI'].min(),
            'max': data['HMPI'].max(),
            'q25': data['HMPI'].quantile(0.25),
            'q75': data['HMPI'].quantile(0.75)
        }
        
        # Metal concentration statistics
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        metal_stats = {}
        
        for metal in metals:
            if metal in data.columns:
                metal_stats[metal] = {
                    'mean': data[metal].mean(),
                    'max': data[metal].max(),
                    'min': data[metal].min(),
                    'std': data[metal].std()
                }
        
        # Top contaminated samples
        top_contaminated = data.nlargest(5, 'HMPI')[['Sample_ID', 'HMPI', 'Quality_Category']]
        
        results = f"""## Results Analysis

### HMPI Statistical Summary

| Statistic | Value |
|-----------|-------|
| Mean      | {hmpi_stats['mean']:.2f} |
| Median    | {hmpi_stats['median']:.2f} |
| Minimum   | {hmpi_stats['min']:.2f} |
| Maximum   | {hmpi_stats['max']:.2f} |
| Std Dev   | {hmpi_stats['std']:.2f} |
| 25th Percentile | {hmpi_stats['q25']:.2f} |
| 75th Percentile | {hmpi_stats['q75']:.2f} |

### Heavy Metal Concentration Summary

| Metal | Mean (mg/L) | Max (mg/L) | Min (mg/L) | Std Dev |
|-------|-------------|-----------|-----------|---------|
"""
        
        for metal, stats in metal_stats.items():
            results += f"| {metal} | {stats['mean']:.6f} | {stats['max']:.6f} | {stats['min']:.6f} | {stats['std']:.6f} |\n"
        
        results += f"""
### Most Contaminated Samples

| Sample ID | HMPI | Quality Category |
|-----------|------|------------------|
"""
        
        for _, row in top_contaminated.iterrows():
            results += f"| {row['Sample_ID']} | {row['HMPI']:.2f} | {row['Quality_Category']} |\n"
        
        return results
    
    def _generate_quality_assessment(self, data, config):
        """Generate quality assessment section"""
        quality_dist = data['Quality_Category'].value_counts()
        total_samples = len(data)
        
        # Calculate risk assessment
        risk_levels = data.get('Risk_Level', pd.Series(['Unknown'] * len(data))).value_counts()
        
        assessment = f"""## Quality Assessment

### Overall Quality Distribution

"""
        
        for category, count in quality_dist.items():
            percentage = (count / total_samples) * 100
            assessment += f"**{category}:** {count} samples ({percentage:.1f}%)\n\n"
            
            if category == 'Excellent':
                assessment += "- These samples are safe for direct consumption\n- No treatment required\n- Meet all international standards\n\n"
            elif category == 'Good':
                assessment += "- These samples are generally safe\n- Basic treatment recommended for enhanced safety\n- Minor exceedances may be present\n\n"
            elif category == 'Poor':
                assessment += "- These samples require treatment before consumption\n- Multiple contaminants may be present\n- Regular monitoring recommended\n\n"
            elif category == 'Very Poor':
                assessment += "- These samples need extensive treatment\n- Multiple standards exceeded\n- Not suitable for direct consumption\n\n"
            elif category == 'Unsuitable':
                assessment += "- These samples are not suitable for drinking\n- Severe contamination present\n- Alternative water sources needed\n\n"
        
        if 'Risk_Level' in data.columns:
            assessment += "### Health Risk Assessment\n\n"
            for risk, count in risk_levels.items():
                percentage = (count / total_samples) * 100
                assessment += f"**{risk}:** {count} samples ({percentage:.1f}%)\n"
        
        # Spatial quality assessment if coordinates available
        if 'Latitude' in data.columns and 'Longitude' in data.columns:
            contaminated_areas = data[data['HMPI'] > 100]
            if len(contaminated_areas) > 0:
                assessment += f"""
### Spatial Quality Concerns

- {len(contaminated_areas)} contaminated samples identified
- Geographic clustering analysis recommended
- Potential point sources of contamination to be investigated
"""
        
        return assessment
    
    def _generate_statistical_analysis(self, data, config):
        """Generate statistical analysis section"""
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        
        # Correlation analysis
        if len(metals) >= 2:
            correlation_data = data[metals + ['HMPI']].corr()
            
        # Standards exceedance analysis
        standards = {
            'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
            'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
        }
        
        exceedances = {}
        for metal in metals:
            if metal in data.columns and metal in standards:
                exceeded = (data[metal] > standards[metal]).sum()
                exceedances[metal] = {
                    'count': exceeded,
                    'percentage': (exceeded / len(data)) * 100
                }
        
        analysis = f"""## Statistical Analysis

### Descriptive Statistics

The dataset shows the following statistical characteristics:

- **Sample Size:** {len(data)} groundwater samples
- **Metals Analyzed:** {len(metals)} heavy metals
- **HMPI Range:** {data['HMPI'].min():.2f} - {data['HMPI'].max():.2f}
- **Mean HMPI:** {data['HMPI'].mean():.2f} ± {data['HMPI'].std():.2f}

### Standards Exceedance Analysis

| Metal | Samples Exceeding Standard | Percentage |
|-------|---------------------------|------------|
"""
        
        for metal, exc_data in exceedances.items():
            analysis += f"| {metal} | {exc_data['count']} | {exc_data['percentage']:.1f}% |\n"
        
        # Distribution analysis
        skewness = data['HMPI'].skew()
        kurtosis = data['HMPI'].kurtosis()
        
        analysis += f"""
### Distribution Analysis

- **Skewness:** {skewness:.3f} {"(Right-skewed)" if skewness > 0.5 else "(Left-skewed)" if skewness < -0.5 else "(Nearly symmetric)"}
- **Kurtosis:** {kurtosis:.3f} {"(Heavy-tailed)" if kurtosis > 3 else "(Light-tailed)" if kurtosis < 3 else "(Normal-tailed)"}

### Key Statistical Insights

"""
        
        if skewness > 1:
            analysis += "- Data shows strong positive skew, indicating presence of highly contaminated samples\n"
        if data['HMPI'].std() / data['HMPI'].mean() > 1:
            analysis += "- High coefficient of variation suggests significant spatial heterogeneity\n"
        if (data['HMPI'] > 100).sum() > len(data) * 0.25:
            analysis += "- Over 25% of samples exceed safe limits, indicating widespread contamination\n"
        
        return analysis
    
    def _generate_geospatial_analysis(self, data, config):
        """Generate geospatial analysis section"""
        if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
            return """## Geospatial Analysis

Geospatial analysis is not available as coordinate data was not provided in the dataset.
"""
        
        # Filter valid coordinates
        valid_coords = data.dropna(subset=['Latitude', 'Longitude'])
        
        if len(valid_coords) == 0:
            return """## Geospatial Analysis

No valid coordinate data available for geospatial analysis.
"""
        
        # Calculate spatial extent
        lat_range = valid_coords['Latitude'].max() - valid_coords['Latitude'].min()
        lon_range = valid_coords['Longitude'].max() - valid_coords['Longitude'].min()
        
        # Spatial clustering analysis
        contaminated_samples = valid_coords[valid_coords['HMPI'] > 100]
        
        analysis = f"""## Geospatial Analysis

### Spatial Coverage

- **Samples with Coordinates:** {len(valid_coords)} out of {len(data)}
- **Latitude Range:** {valid_coords['Latitude'].min():.4f}° to {valid_coords['Latitude'].max():.4f}°
- **Longitude Range:** {valid_coords['Longitude'].min():.4f}° to {valid_coords['Longitude'].max():.4f}°
- **Study Area Extent:** {lat_range:.4f}° × {lon_range:.4f}°

### Spatial Distribution of Contamination

- **Contaminated Locations:** {len(contaminated_samples)}
- **Clean Locations:** {len(valid_coords) - len(contaminated_samples)}

### Spatial Patterns

"""
        
        if len(contaminated_samples) > 0:
            # Calculate contamination density
            contamination_density = len(contaminated_samples) / len(valid_coords) * 100
            analysis += f"- **Contamination Density:** {contamination_density:.1f}% of mapped locations\n"
            
            if contamination_density > 50:
                analysis += "- **Pattern:** Widespread contamination across study area\n"
                analysis += "- **Recommendation:** Regional source investigation required\n"
            elif contamination_density > 25:
                analysis += "- **Pattern:** Moderate contamination with potential clustering\n"
                analysis += "- **Recommendation:** Focus on hotspot areas\n"
            else:
                analysis += "- **Pattern:** Localized contamination points\n"
                analysis += "- **Recommendation:** Point source investigation\n"
        
        return analysis
    
    def _generate_conclusions(self, data, config):
        """Generate conclusions section"""
        total_samples = len(data)
        contaminated = (data['HMPI'] > 100).sum()
        contamination_rate = (contaminated / total_samples) * 100
        avg_hmpi = data['HMPI'].mean()
        
        # Identify primary contaminants
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        standards = {
            'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
            'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
        }
        
        primary_contaminants = []
        for metal in metals:
            if metal in data.columns and metal in standards:
                exceedance_rate = (data[metal] > standards[metal]).mean() * 100
                if exceedance_rate > 20:  # More than 20% samples exceed standard
                    primary_contaminants.append(f"{metal} ({exceedance_rate:.1f}%)")
        
        conclusions = f"""## Conclusions

Based on the comprehensive analysis of {total_samples} groundwater samples using the Heavy Metal Pollution Index (HMPI) methodology, the following conclusions can be drawn:

### Overall Assessment

1. **Contamination Status:** {contamination_rate:.1f}% of samples exceed the safe HMPI threshold (>100), indicating {"widespread contamination concerns" if contamination_rate > 25 else "localized contamination issues" if contamination_rate > 10 else "generally acceptable water quality"}.

2. **Average Pollution Level:** The mean HMPI value of {avg_hmpi:.2f} indicates {"severe pollution" if avg_hmpi > 200 else "moderate pollution" if avg_hmpi > 100 else "acceptable water quality" if avg_hmpi > 50 else "excellent water quality"}.

3. **Quality Distribution:** The majority of samples fall into the {"poor to unsuitable" if (data['Quality_Category'].isin(['Poor', 'Very Poor', 'Unsuitable'])).mean() > 0.5 else "good to excellent"} category range.

### Primary Contaminants

"""
        
        if primary_contaminants:
            conclusions += "The following metals show significant exceedance rates:\n"
            for contaminant in primary_contaminants:
                conclusions += f"- {contaminant} of samples exceed standards\n"
        else:
            conclusions += "No metals show widespread exceedance of standards across the study area.\n"
        
        conclusions += f"""
### Key Findings

1. **Public Health Impact:** {"High priority intervention required" if contamination_rate > 50 else "Moderate health risk present" if contamination_rate > 25 else "Low health risk overall"}

2. **Treatment Requirements:** {"Extensive treatment infrastructure needed" if contamination_rate > 50 else "Selective treatment for contaminated sources" if contamination_rate > 10 else "Minimal treatment required"}

3. **Monitoring Needs:** {"Continuous monitoring essential" if avg_hmpi > 100 else "Regular monitoring recommended"}

### Data Quality

- Sample size of {total_samples} provides {"excellent" if total_samples > 50 else "good" if total_samples > 25 else "adequate"} statistical power
- {len(metals)} heavy metals analyzed provide comprehensive pollution assessment
- {"Geospatial data available enables spatial analysis" if 'Latitude' in data.columns else "Limited spatial analysis due to missing coordinates"}
"""
        
        return conclusions
    
    def _generate_recommendations(self, data, config):
        """Generate recommendations section"""
        total_samples = len(data)
        contaminated = (data['HMPI'] > 100).sum()
        contamination_rate = (contaminated / total_samples) * 100
        
        # Identify critical samples
        critical_samples = data[data['HMPI'] > 200]
        
        recommendations = f"""## Recommendations

Based on the analysis results, the following recommendations are proposed for groundwater management and public health protection:

### Immediate Actions

"""
        
        if len(critical_samples) > 0:
            recommendations += f"""1. **Emergency Response:** {len(critical_samples)} samples show HMPI > 200 (unsuitable for drinking)
   - Immediately discontinue use of these water sources
   - Provide alternative safe water supply
   - Conduct health assessment for affected populations

"""
        
        if contamination_rate > 25:
            recommendations += """2. **Water Treatment Implementation:**
   - Install appropriate water treatment systems
   - Consider reverse osmosis, ion exchange, or activated carbon filtration
   - Implement quality control testing protocols

"""
        
        recommendations += f"""### Short-term Actions (1-6 months)

1. **Enhanced Monitoring Program:**
   - Increase sampling frequency for contaminated locations
   - Expand monitoring network if spatial coverage is limited
   - Include additional heavy metals in analysis if resources permit

2. **Source Investigation:**
   - Identify potential pollution sources (industrial, agricultural, natural)
   - Conduct upstream/downgradient sampling
   - Assess groundwater flow patterns

3. **Public Health Measures:**
   - Issue public advisories for contaminated wells
   - Provide bottled water or alternative sources where needed
   - Conduct health screenings in high-exposure areas

### Medium-term Actions (6 months - 2 years)

1. **Remediation Planning:**
   - Develop site-specific remediation strategies
   - Consider pump-and-treat, permeable reactive barriers, or in-situ treatment
   - Prioritize based on contamination levels and population at risk

2. **Regulatory Compliance:**
   - Ensure compliance with local and international water quality standards
   - Develop monitoring and reporting protocols
   - Establish water quality databases

3. **Infrastructure Development:**
   - Upgrade water supply infrastructure in affected areas
   - Consider centralized treatment facilities for high-contamination zones
   - Implement distribution system improvements

### Long-term Actions (2+ years)

1. **Preventive Measures:**
   - Implement wellhead protection programs
   - Establish land use controls around water sources
   - Develop groundwater protection regulations

2. **Sustainable Management:**
   - Create integrated water resource management plans
   - Develop alternative water sources (surface water, rainwater harvesting)
   - Implement water conservation programs

3. **Capacity Building:**
   - Train local personnel in water quality monitoring
   - Establish laboratory capabilities for routine analysis
   - Develop emergency response protocols

### Specific Technical Recommendations

"""
        
        # Metal-specific recommendations
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        standards = {
            'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
            'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
        }
        
        treatment_recommendations = {
            'As': 'Coagulation-fliltration, reverse osmosis, or iron-based adsorption',
            'Cd': 'Reverse osmosis, ion exchange, or lime softening',
            'Cr': 'Reduction followed by precipitation, or ion exchange',
            'Cu': 'pH adjustment, ion exchange, or reverse osmosis',
            'Fe': 'Oxidation and filtration, or ion exchange',
            'Mn': 'Oxidation and filtration, or ion exchange',
            'Ni': 'Ion exchange, reverse osmosis, or lime softening',
            'Pb': 'Corrosion control, ion exchange, or reverse osmosis',
            'Zn': 'pH adjustment, ion exchange, or reverse osmosis'
        }
        
        for metal in metals:
            if metal in data.columns and metal in standards:
                exceedance_rate = (data[metal] > standards[metal]).mean() * 100
                if exceedance_rate > 10:
                    treatment = treatment_recommendations.get(metal, 'Appropriate treatment technology')
                    recommendations += f"- **{metal} Treatment:** {treatment} (affects {exceedance_rate:.1f}% of samples)\n"
        
        recommendations += f"""
### Monitoring Program Recommendations

1. **Sampling Frequency:**
   - Monthly monitoring for contaminated sites (HMPI > 100)
   - Quarterly monitoring for sites with HMPI 50-100
   - Annual monitoring for clean sites (HMPI < 50)

2. **Additional Parameters:**
   - pH, electrical conductivity, total dissolved solids
   - Additional heavy metals if industrial sources are suspected
   - Bacterial indicators if biological contamination is possible

3. **Quality Assurance:**
   - Use certified laboratories with appropriate detection limits
   - Include duplicate samples and blanks (10% of total samples)
   - Participate in inter-laboratory comparison programs

### Budget Considerations

Estimated budget requirements for implementation:
- Emergency response: High priority, immediate funding needed
- Treatment systems: Major investment, consider phased implementation
- Monitoring program: Ongoing operational cost, essential for management
- Infrastructure: Long-term investment, seek multiple funding sources

### Success Metrics

- Reduction in percentage of samples exceeding HMPI threshold
- Decreased average HMPI values over time
- Compliance with drinking water standards
- Reduced health risk indicators in affected populations
"""
        
        return recommendations
    
    def generate_summary_statistics(self, data):
        """Generate summary statistics in JSON format"""
        stats = {}
        
        # Basic statistics
        stats['sample_info'] = {
            'total_samples': int(len(data)),
            'analysis_date': datetime.now().isoformat(),
            'metals_analyzed': [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        }
        
        # HMPI statistics
        stats['hmpi_statistics'] = {
            'mean': float(data['HMPI'].mean()),
            'median': float(data['HMPI'].median()),
            'std': float(data['HMPI'].std()),
            'min': float(data['HMPI'].min()),
            'max': float(data['HMPI'].max()),
            'q25': float(data['HMPI'].quantile(0.25)),
            'q75': float(data['HMPI'].quantile(0.75))
        }
        
        # Quality distribution
        quality_dist = data['Quality_Category'].value_counts()
        stats['quality_distribution'] = {str(k): int(v) for k, v in quality_dist.items()}
        
        # Contamination summary
        contaminated = (data['HMPI'] > 100).sum()
        stats['contamination_summary'] = {
            'contaminated_samples': int(contaminated),
            'contamination_rate': float(contaminated / len(data) * 100),
            'safe_samples': int(len(data) - contaminated)
        }
        
        # Metal-specific statistics
        metals = [col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        stats['metal_statistics'] = {}
        
        standards = {
            'As': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Cu': 2.0, 'Fe': 0.3,
            'Mn': 0.4, 'Ni': 0.07, 'Pb': 0.01, 'Zn': 3.0
        }
        
        for metal in metals:
            if metal in data.columns:
                exceeded = (data[metal] > standards.get(metal, 1.0)).sum()
                stats['metal_statistics'][metal] = {
                    'mean': float(data[metal].mean()),
                    'max': float(data[metal].max()),
                    'min': float(data[metal].min()),
                    'std': float(data[metal].std()),
                    'standard_value': standards.get(metal, 1.0),
                    'samples_exceeded': int(exceeded),
                    'exceedance_rate': float(exceeded / len(data) * 100)
                }
        
        return json.dumps(stats, indent=2)
    
    def _get_executive_summary_template(self):
        """Get executive summary template"""
        return "Executive summary template"
    
    def _get_methodology_template(self):
        """Get methodology template"""
        return "Methodology template"
    
    def _get_results_analysis_template(self):
        """Get results analysis template"""
        return "Results analysis template"
    
    def _get_quality_assessment_template(self):
        """Get quality assessment template"""
        return "Quality assessment template"
    
    def _get_statistical_analysis_template(self):
        """Get statistical analysis template"""
        return "Statistical analysis template"
    
    def _get_geospatial_analysis_template(self):
        """Get geospatial analysis template"""
        return "Geospatial analysis template"
    
    def _get_conclusions_template(self):
        """Get conclusions template"""
        return "Conclusions template"
    
    def _get_recommendations_template(self):
        """Get recommendations template"""
        return "Recommendations template"
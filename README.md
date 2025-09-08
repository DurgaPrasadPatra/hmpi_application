# README.md
# Heavy Metal Pollution Indices Calculator

A comprehensive web application for calculating Heavy Metal Pollution Indices (HMPI) in groundwater using standard methodologies.

## Features

- **Automated HMPI Calculation**: Compute multiple pollution indices using standard formulas
- **Data Processing**: Load, validate, and preprocess heavy metal concentration data
- **Interactive Visualizations**: Create plots, charts, and maps for data exploration
- **Geospatial Analysis**: Analyze spatial distribution of contamination
- **Comprehensive Reports**: Generate detailed assessment reports
- **User-Friendly Interface**: Streamlit-based web interface for easy operation

## Supported Heavy Metals

- Arsenic (As)
- Cadmium (Cd)
- Chromium (Cr)
- Copper (Cu)
- Iron (Fe)
- Manganese (Mn)
- Nickel (Ni)
- Lead (Pb)
- Zinc (Zn)

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/hmpi-calculator.git
cd hmpi-calculator
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

### 1. Data Input

Upload your groundwater data in CSV or Excel format with the following columns:
- Sample_ID (optional)
- Latitude, Longitude (optional, for geospatial analysis)
- Heavy metal concentrations (As, Cd, Cr, Cu, Fe, Mn, Ni, Pb, Zn)

### 2. Calculate HMPI

The application automatically calculates:
- Heavy Metal Pollution Index (HMPI)
- Quality categories (Excellent, Good, Poor, Very Poor, Unsuitable)
- Individual metal pollution indices
- Risk assessments

### 3. Visualizations

Generate interactive plots:
- HMPI distribution histograms
- Quality category pie charts
- Metal concentration comparisons
- Correlation heatmaps
- Geospatial contamination maps

### 4. Reports

Create comprehensive reports including:
- Executive summary
- Methodology
- Results analysis
- Quality assessment
- Statistical analysis
- Geospatial analysis
- Conclusions and recommendations

## Data Format Example

```csv
Sample_ID,Latitude,Longitude,As,Cd,Cr,Cu,Fe,Mn,Ni,Pb,Zn
SAMPLE_001,13.0827,80.2707,0.005,0.001,0.02,0.5,0.1,0.05,0.01,0.005,1.0
SAMPLE_002,13.0850,80.2750,0.003,0.002,0.015,0.3,0.2,0.08,0.012,0.004,0.8
```

## Methodology

The HMPI is calculated using the formula:
```
HMPI = Σ(Wi × Qi) / Σ(Wi)
```

Where:
- Wi = Weight of ith metal based on toxicity
- Qi = Quality rating = (Ci / Si) × 100
- Ci = Concentration of ith metal
- Si = Standard permissible value

## Quality Categories

- **Excellent (≤25)**: Safe for drinking
- **Good (25-50)**: Safe with minimal treatment
- **Poor (50-100)**: Requires treatment
- **Very Poor (100-200)**: Extensive treatment needed
- **Unsuitable (>200)**: Not suitable for drinking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contact

For questions or support, contact: durgatech555@gmail.com  
LinkedIn: www.linkedin.com/in/durgatech555

## Citation

If you use this application in your research, please cite:

```
Heavy Metal Pollution Indices Calculator (2025). 
Available at: https://github.com/DurgaPrasadPatra/hmpi-calculator

```

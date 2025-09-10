# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.hmpi_calculator import HMPICalculator
from modules.data_processor import DataProcessor
from modules.visualization import Visualizer
from modules.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Heavy Metal Pollution Indices Calculator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Header
    st.markdown('<h1 class="main-header">🧪 Heavy Metal Pollution Indices Calculator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox(
            "Select Module",
            ["Data Input", "Calculate HMPI", "Visualizations", "Geospatial Analysis", "Reports"]
        )
        
        st.markdown("---")
        st.markdown("### Application Info")
        st.info("""
        This application calculates Heavy Metal Pollution Indices (HMPI) 
        for groundwater quality assessment using standard methodologies.
        """)
        
        # Quick stats
        if st.session_state.processed_data is not None:
            st.markdown("### Dataset Summary")
            data = st.session_state.processed_data
            st.metric("Total Samples", len(data))
            st.metric("Heavy Metals", len([col for col in data.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]))

    # Main content based on selected page
    if page == "Data Input":
        show_data_input()
    elif page == "Calculate HMPI":
        show_hmpi_calculation()
    elif page == "Visualizations":
        show_visualizations()
    elif page == "Geospatial Analysis":
        show_geospatial_analysis()
    elif page == "Reports":
        show_reports()

def show_data_input():
    st.markdown('<h2 class="section-header">📊 Data Input Module</h2>', unsafe_allow_html=True)
    
    # Data input options
    input_method = st.radio(
        "Select Data Input Method:",
        ["Upload CSV/Excel File", "Manual Entry", "Use Sample Data"]
    )
    
    if input_method == "Upload CSV/Excel File":
        uploaded_file = st.file_uploader(
            "Upload your groundwater heavy metal data",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain heavy metal concentrations and coordinates"
        )
        
        if uploaded_file is not None:
            try:
                data_processor = DataProcessor()
                df = data_processor.load_data(uploaded_file)
                st.session_state.data = df
                
                st.success("✅ Data uploaded successfully!")
                st.dataframe(df.head())
                
                # Data validation
                validation_results = data_processor.validate_data(df)
                if validation_results['is_valid']:
                    st.success("✅ Data validation passed!")
                    st.session_state.processed_data = data_processor.preprocess_data(df)
                else:
                    st.error("❌ Data validation failed!")
                    for error in validation_results['errors']:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    elif input_method == "Manual Entry":
        show_manual_entry()
    
    elif input_method == "Use Sample Data":
        data_processor = DataProcessor()
        sample_data = data_processor.generate_sample_data()
        st.session_state.data = sample_data
        st.session_state.processed_data = sample_data
        
        st.success("✅ Sample data loaded!")
        st.dataframe(sample_data.head())
        
        st.info("""
        **Sample Data Information:**
        - 50 groundwater samples
        - 9 heavy metals: As, Cd, Cr, Cu, Fe, Mn, Ni, Pb, Zn
        - Geographical coordinates included
        - Mix of contaminated and clean samples
        """)

def show_manual_entry():
    st.markdown("#### Manual Data Entry")
    
    col1, col2 = st.columns(2)
    with col1:
        sample_id = st.text_input("Sample ID", value="SAMPLE_001")
        latitude = st.number_input("Latitude", value=0.0, format="%.6f")
        longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        
    with col2:
        location = st.text_input("Location", value="")
        date_collected = st.date_input("Date Collected", value=datetime.now())
    
    st.markdown("#### Heavy Metal Concentrations (mg/L)")
    
    metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
    metal_values = {}
    
    cols = st.columns(3)
    for i, metal in enumerate(metals):
        with cols[i % 3]:
            metal_values[metal] = st.number_input(f"{metal} (mg/L)", min_value=0.0, value=0.001, format="%.6f", key=metal)
    
    if st.button("Add Sample"):
        new_sample = {
            'Sample_ID': sample_id,
            'Latitude': latitude,
            'Longitude': longitude,
            'Location': location,
            'Date_Collected': date_collected,
            **metal_values
        }
        
        if st.session_state.data is None:
            st.session_state.data = pd.DataFrame([new_sample])
        else:
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_sample])], ignore_index=True)
        
        data_processor = DataProcessor()
        st.session_state.processed_data = data_processor.preprocess_data(st.session_state.data)
        st.success(f"✅ Sample {sample_id} added successfully!")

def show_hmpi_calculation():
    st.markdown('<h2 class="section-header">🧮 HMPI Calculation Module</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload or enter data first in the Data Input module.")
        return
    
    # HMPI Configuration
    calculator = HMPICalculator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Calculation Parameters")
        
        # Standard values configuration
        use_custom_standards = st.checkbox("Use Custom Standard Values")
        
        if use_custom_standards:
            st.markdown("##### Custom Standard Values (mg/L)")
            standards = {}
            metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
            
            for metal in metals:
                if metal in st.session_state.processed_data.columns:
                    default_val = calculator.standard_values.get(metal, 0.01)
                    standards[metal] = st.number_input(
                        f"{metal} Standard", 
                        value=default_val, 
                        format="%.6f",
                        key=f"std_{metal}"
                    )
            calculator.standard_values.update(standards)
    
    with col2:
        st.markdown("#### Current Standard Values (mg/L)")
        standards_df = pd.DataFrame(
            list(calculator.standard_values.items()), 
            columns=['Metal', 'Standard Value']
        )
        st.dataframe(standards_df)
    
    # Calculate HMPI
    if st.button("🚀 Calculate HMPI", type="primary"):
        with st.spinner("Calculating Heavy Metal Pollution Indices..."):
            try:
                results = calculator.calculate_hmpi(st.session_state.processed_data)
                st.session_state.results = results
                
                st.success("✅ HMPI calculation completed!")
                
                # Display results summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_hmpi = results['HMPI'].mean()
                    st.metric("Average HMPI", f"{avg_hmpi:.2f}")
                
                with col2:
                    max_hmpi = results['HMPI'].max()
                    st.metric("Max HMPI", f"{max_hmpi:.2f}")
                
                with col3:
                    contaminated = len(results[results['HMPI'] > 100])
                    st.metric("Contaminated Samples", contaminated)
                
                with col4:
                    clean_samples = len(results[results['HMPI'] <= 100])
                    st.metric("Clean Samples", clean_samples)
                
                # Results table
                st.markdown("#### Detailed Results")
                st.dataframe(results.round(4))
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "hmpi_results.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error calculating HMPI: {str(e)}")
    
    # Display existing results if available
    if st.session_state.results is not None:
        st.markdown("#### Quality Classification")
        
        results = st.session_state.results
        quality_dist = results['Quality_Category'].value_counts()
        
        fig = px.pie(
            values=quality_dist.values,
            names=quality_dist.index,
            title="Groundwater Quality Distribution",
            color_discrete_map={
                'Excellent': '#2E8B57',
                'Good': '#32CD32',
                'Poor': '#FFD700',
                'Very Poor': '#FF6347',
                'Unsuitable': '#DC143C'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations():
    st.markdown('<h2 class="section-header">📈 Visualization Module</h2>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("⚠️ Please calculate HMPI first in the Calculate HMPI module.")
        return
    
    visualizer = Visualizer()
    results = st.session_state.results
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["HMPI Distribution", "Metal Concentrations", "Correlation Analysis", "Contamination Hotspots"]
    )
    
    if viz_type == "HMPI Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            # HMPI histogram
            fig_hist = visualizer.plot_hmpi_distribution(results)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Quality categories
            fig_pie = visualizer.plot_quality_categories(results)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # HMPI vs Sample scatter plot
        fig_scatter = visualizer.plot_hmpi_scatter(results)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif viz_type == "Metal Concentrations":
        # Heavy metal concentrations
        metals = [col for col in results.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        selected_metals = st.multiselect("Select Metals to Visualize:", metals, default=metals[:3])
        
        if selected_metals:
            fig_metal = visualizer.plot_metal_concentrations(results, selected_metals)
            st.plotly_chart(fig_metal, use_container_width=True)
            
            # Box plots for selected metals
            fig_box = visualizer.plot_metal_boxplots(results, selected_metals)
            st.plotly_chart(fig_box, use_container_width=True)
    
    elif viz_type == "Correlation Analysis":
        # Correlation heatmap
        fig_corr = visualizer.plot_correlation_heatmap(results)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # PCA analysis if enough features
        metals = [col for col in results.columns if col in ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']]
        if len(metals) >= 2:
            fig_pca = visualizer.plot_pca_analysis(results, metals)
            if fig_pca:
                st.plotly_chart(fig_pca, use_container_width=True)
    
    elif viz_type == "Contamination Hotspots":
        # Top contaminated samples
        top_contaminated = results.nlargest(10, 'HMPI')
        
        fig_top = px.bar(
            top_contaminated,
            x='Sample_ID',
            y='HMPI',
            color='Quality_Category',
            title="Top 10 Most Contaminated Samples",
            color_discrete_map={
                'Excellent': '#2E8B57',
                'Good': '#32CD32',
                'Poor': '#FFD700',
                'Very Poor': '#FF6347',
                'Unsuitable': '#DC143C'
            }
        )
        fig_top.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top, use_container_width=True)

def show_geospatial_analysis():
    st.markdown('<h2 class="section-header">🗺️ Geospatial Analysis Module</h2>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("⚠️ Please calculate HMPI first in the Calculate HMPI module.")
        return
    
    results = st.session_state.results
    
    # Check if coordinates are available
    if 'Latitude' not in results.columns or 'Longitude' not in results.columns:
        st.error("❌ Latitude and Longitude columns are required for geospatial analysis.")
        return
    
    # Filter out rows with invalid coordinates
    valid_coords = results.dropna(subset=['Latitude', 'Longitude'])
    valid_coords = valid_coords[(valid_coords['Latitude'] != 0) | (valid_coords['Longitude'] != 0)]
    
    if len(valid_coords) == 0:
        st.error("❌ No valid coordinates found in the dataset.")
        return
    
    # Map configuration
    st.markdown("#### Interactive Contamination Map")
    
    # Create folium map
    center_lat = valid_coords['Latitude'].mean()
    center_lon = valid_coords['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add markers based on HMPI values
    for _, row in valid_coords.iterrows():
        hmpi = row['HMPI']
        
        # Determine color based on HMPI
        if hmpi <= 25:
            color = 'green'
            category = 'Excellent'
        elif hmpi <= 50:
            color = 'lightgreen'
            category = 'Good'
        elif hmpi <= 100:
            color = 'yellow'
            category = 'Poor'
        elif hmpi <= 200:
            color = 'orange'
            category = 'Very Poor'
        else:
            color = 'red'
            category = 'Unsuitable'
        
        popup_text = f"""
        <b>Sample ID:</b> {row.get('Sample_ID', 'N/A')}<br>
        <b>HMPI:</b> {hmpi:.2f}<br>
        <b>Category:</b> {category}<br>
        <b>Location:</b> {row.get('Location', 'N/A')}<br>
        <b>Coordinates:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>HMPI Categories</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Excellent (≤25)</p>
    <p><i class="fa fa-circle" style="color:lightgreen"></i> Good (25-50)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Poor (50-100)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Very Poor (100-200)</p>
    <p><i class="fa fa-circle" style="color:red"></i> Unsuitable (>200)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Spatial statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spatial Statistics")
        st.metric("Total Mapped Samples", len(valid_coords))
        st.metric("Average HMPI", f"{valid_coords['HMPI'].mean():.2f}")
        st.metric("Spatial Range (km)", f"{calculate_spatial_range(valid_coords):.1f}")
    
    with col2:
        st.markdown("#### Quality Distribution")
        quality_counts = valid_coords['Quality_Category'].value_counts()
        for category, count in quality_counts.items():
            st.metric(category, count)

def calculate_spatial_range(data):
    """Calculate the spatial range of samples in kilometers"""
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    max_dist = 0
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            dist = haversine(
                data.iloc[i]['Longitude'], data.iloc[i]['Latitude'],
                data.iloc[j]['Longitude'], data.iloc[j]['Latitude']
            )
            max_dist = max(max_dist, dist)
    
    return max_dist

def show_reports():
    st.markdown('<h2 class="section-header">📋 Report Generation Module</h2>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("⚠️ Please calculate HMPI first in the Calculate HMPI module.")
        return
    
    report_generator = ReportGenerator()
    results = st.session_state.results
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", "Heavy Metal Pollution Assessment Report")
        organization = st.text_input("Organization", "Environmental Monitoring Agency")
        author = st.text_input("Report Author", "Data Analyst")
    
    with col2:
        report_date = st.date_input("Report Date", datetime.now())
        include_maps = st.checkbox("Include Geospatial Maps", True)
        include_recommendations = st.checkbox("Include Recommendations", True)
    
    # Report sections
    st.markdown("#### Select Report Sections")
    sections = st.multiselect(
        "Report Sections:",
        ["Executive Summary", "Methodology", "Results Analysis", "Quality Assessment", 
         "Statistical Analysis", "Geospatial Analysis", "Conclusions", "Recommendations"],
        default=["Executive Summary", "Results Analysis", "Quality Assessment", "Conclusions"]
    )
    
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            try:
                # Generate report content
                report_config = {
                    'title': report_title,
                    'organization': organization,
                    'author': author,
                    'date': report_date,
                    'sections': sections,
                    'include_maps': include_maps,
                    'include_recommendations': include_recommendations
                }
                
                report_content = report_generator.generate_report(results, report_config)
                
                st.success("✅ Report generated successfully!")
                
                # Display report preview
                st.markdown("#### Report Preview")
                st.markdown(report_content[:2000] + "..." if len(report_content) > 2000 else report_content)
                
                # Download options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Markdown download
                    st.download_button(
                        "📥 Download Report (Markdown)",
                        report_content,
                        f"hmpi_report_{datetime.now().strftime('%Y%m%d')}.md",
                        "text/markdown"
                    )
                
                with col2:
                    # CSV data download
                    csv_data = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Data (CSV)",
                        csv_data,
                        f"hmpi_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
                
                with col3:
                    # Summary statistics
                    summary_stats = report_generator.generate_summary_statistics(results)
                    st.download_button(
                        "📥 Download Statistics (JSON)",
                        summary_stats,
                        f"hmpi_statistics_{datetime.now().strftime('%Y%m%d')}.json",
                        "application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
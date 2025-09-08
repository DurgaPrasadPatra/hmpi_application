# visualization.py
# modules/visualization.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """
    Visualization Module for Heavy Metal Pollution Analysis
    
    Creates interactive plots and charts for data exploration,
    results presentation, and geospatial analysis.
    """
    
    def __init__(self):
        """Initialize with color schemes and plotting configurations"""
        self.color_schemes = {
            'quality_categories': {
                'Excellent': '#2E8B57',      # Sea Green
                'Good': '#32CD32',           # Lime Green
                'Poor': '#FFD700',           # Gold
                'Very Poor': '#FF6347',      # Tomato
                'Unsuitable': '#DC143C'      # Crimson
            },
            'risk_levels': {
                'No Risk': '#228B22',        # Forest Green
                'Low Risk': '#9ACD32',       # Yellow Green
                'Moderate Risk': '#FFA500',   # Orange
                'High Risk': '#FF4500',      # Orange Red
                'Very High Risk': '#8B0000'   # Dark Red
            },
            'metals': px.colors.qualitative.Set3
        }
        
        # Default plot configuration
        self.plot_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ]
        }
    
    def plot_hmpi_distribution(self, data):
        """
        Create HMPI distribution histogram
        
        Args:
            data (pd.DataFrame): Results data with HMPI values
            
        Returns:
            plotly.graph_objects.Figure: HMPI distribution plot
        """
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data['HMPI'],
            nbinsx=30,
            name='HMPI Distribution',
            marker_color='rgba(55, 128, 191, 0.7)',
            marker_line_color='rgba(55, 128, 191, 1.0)',
            marker_line_width=1
        ))
        
        # Add vertical lines for quality thresholds
        thresholds = [25, 50, 100, 200]
        threshold_labels = ['Excellent', 'Good', 'Poor', 'Very Poor']
        colors = ['green', 'lightgreen', 'orange', 'red']
        
        for threshold, label, color in zip(thresholds, threshold_labels, colors):
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{label} ({threshold})",
                annotation_position="top"
            )
        
        # Update layout
        fig.update_layout(
            title="Heavy Metal Pollution Index (HMPI) Distribution",
            xaxis_title="HMPI Value",
            yaxis_title="Frequency",
            showlegend=True,
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_quality_categories(self, data):
        """
        Create pie chart for quality categories distribution
        
        Args:
            data (pd.DataFrame): Results data with quality categories
            
        Returns:
            plotly.graph_objects.Figure: Quality categories pie chart
        """
        quality_counts = data['Quality_Category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=quality_counts.index,
            values=quality_counts.values,
            hole=0.4,
            marker=dict(colors=[self.color_schemes['quality_categories'].get(cat, '#636EFA') 
                              for cat in quality_counts.index])
        )])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )
        
        fig.update_layout(
            title="Groundwater Quality Distribution",
            template="plotly_white",
            height=500,
            annotations=[dict(text='Quality<br>Categories', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        return fig
    
    def plot_hmpi_scatter(self, data):
        """
        Create scatter plot of HMPI values vs sample index
        
        Args:
            data (pd.DataFrame): Results data
            
        Returns:
            plotly.graph_objects.Figure: HMPI scatter plot
        """
        fig = px.scatter(
            data.reset_index(),
            x='index',
            y='HMPI',
            color='Quality_Category',
            color_discrete_map=self.color_schemes['quality_categories'],
            hover_data=['Sample_ID'],
            title="HMPI Values by Sample"
        )
        
        # Add horizontal lines for quality thresholds
        thresholds = [25, 50, 100, 200]
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
        
        fig.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="HMPI Value",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_metal_concentrations(self, data, selected_metals):
        """
        Create bar chart for metal concentrations
        
        Args:
            data (pd.DataFrame): Results data
            selected_metals (list): List of metals to plot
            
        Returns:
            plotly.graph_objects.Figure: Metal concentrations plot
        """
        # Prepare data for plotting
        plot_data = []
        for i, row in data.iterrows():
            for metal in selected_metals:
                if metal in data.columns:
                    plot_data.append({
                        'Sample_ID': row['Sample_ID'],
                        'Metal': metal,
                        'Concentration': row[metal],
                        'Quality_Category': row.get('Quality_Category', 'Unknown')
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        fig = px.bar(
            plot_df,
            x='Sample_ID',
            y='Concentration',
            color='Metal',
            title=f"Heavy Metal Concentrations ({', '.join(selected_metals)})",
            facet_col='Metal',
            facet_col_wrap=2
        )
        
        fig.update_layout(
            height=600,
            template="plotly_white"
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_metal_boxplots(self, data, selected_metals):
        """
        Create box plots for metal concentration distributions
        
        Args:
            data (pd.DataFrame): Results data
            selected_metals (list): List of metals to plot
            
        Returns:
            plotly.graph_objects.Figure: Metal box plots
        """
        fig = go.Figure()
        
        for i, metal in enumerate(selected_metals):
            if metal in data.columns:
                fig.add_trace(go.Box(
                    y=data[metal],
                    name=metal,
                    boxpoints='outliers',
                    marker_color=self.color_schemes['metals'][i % len(self.color_schemes['metals'])]
                ))
        
        fig.update_layout(
            title="Heavy Metal Concentration Distributions",
            yaxis_title="Concentration (mg/L)",
            xaxis_title="Heavy Metals",
            template="plotly_white",
            height=500,
            yaxis_type="log"  # Log scale for better visualization
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data):
        """
        Create correlation heatmap for heavy metals and HMPI
        
        Args:
            data (pd.DataFrame): Results data
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Select numeric columns for correlation
        numeric_cols = [col for col in data.columns if col in 
                       ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn', 'HMPI']]
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Heavy Metals and HMPI Correlation Matrix",
            template="plotly_white",
            height=600,
            width=600
        )
        
        return fig
    
    def plot_pca_analysis(self, data, metal_columns):
        """
        Create PCA analysis plot
        
        Args:
            data (pd.DataFrame): Results data
            metal_columns (list): List of metal columns for PCA
            
        Returns:
            plotly.graph_objects.Figure: PCA plot
        """
        if len(metal_columns) < 2:
            return None
        
        try:
            # Prepare data for PCA
            pca_data = data[metal_columns].fillna(0)
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create DataFrame for plotting
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Quality_Category': data['Quality_Category'],
                'Sample_ID': data['Sample_ID'],
                'HMPI': data['HMPI']
            })
            
            # Create scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Quality_Category',
                color_discrete_map=self.color_schemes['quality_categories'],
                hover_data=['Sample_ID', 'HMPI'],
                title=f"PCA Analysis of Heavy Metals<br>PC1: {pca.explained_variance_ratio_[0]:.1%} variance, PC2: {pca.explained_variance_ratio_[1]:.1%} variance"
            )
            
            # Add loading vectors
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, metal in enumerate(metal_columns):
                fig.add_annotation(
                    ax=0, ay=0,
                    axref="x", ayref="y",
                    x=loadings[i, 0] * 2,
                    y=loadings[i, 1] * 2,
                    xref="x", yref="y",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )
                fig.add_annotation(
                    x=loadings[i, 0] * 2.2,
                    y=loadings[i, 1] * 2.2,
                    text=metal,
                    showarrow=False,
                    font=dict(color="red", size=12)
                )
            
            fig.update_layout(
                template="plotly_white",
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error in PCA analysis: {str(e)}")
            return None
    
    def plot_geospatial_heatmap(self, data):
        """
        Create geospatial heatmap of HMPI values
        
        Args:
            data (pd.DataFrame): Results data with coordinates
            
        Returns:
            plotly.graph_objects.Figure: Geospatial heatmap
        """
        if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
            return None
        
        # Filter valid coordinates
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        fig = px.scatter_mapbox(
            valid_data,
            lat='Latitude',
            lon='Longitude',
            color='HMPI',
            size='HMPI',
            hover_data=['Sample_ID', 'Quality_Category'],
            color_continuous_scale='Viridis',
            size_max=15,
            zoom=10,
            title="Geospatial Distribution of HMPI Values"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        return fig
    
    def plot_metal_standards_comparison(self, data, standards_dict):
        """
        Compare metal concentrations with standards
        
        Args:
            data (pd.DataFrame): Results data
            standards_dict (dict): Standard values for comparison
            
        Returns:
            plotly.graph_objects.Figure: Standards comparison plot
        """
        metals = [col for col in data.columns if col in standards_dict.keys()]
        
        if not metals:
            return None
        
        # Calculate exceedance statistics
        exceedance_data = []
        for metal in metals:
            if metal in data.columns:
                standard = standards_dict[metal]
                exceeded = (data[metal] > standard).sum()
                total = len(data)
                percentage = (exceeded / total) * 100
                
                exceedance_data.append({
                    'Metal': metal,
                    'Samples_Exceeded': exceeded,
                    'Total_Samples': total,
                    'Exceedance_Percentage': percentage,
                    'Standard_Value': standard
                })
        
        exceedance_df = pd.DataFrame(exceedance_data)
        
        # Create bar chart
        fig = px.bar(
            exceedance_df,
            x='Metal',
            y='Exceedance_Percentage',
            title="Percentage of Samples Exceeding Standards",
            hover_data=['Samples_Exceeded', 'Total_Samples', 'Standard_Value']
        )
        
        fig.update_layout(
            yaxis_title="Exceedance Percentage (%)",
            xaxis_title="Heavy Metals",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_dashboard_summary(self, data):
        """
        Create a comprehensive dashboard summary
        
        Args:
            data (pd.DataFrame): Results data
            
        Returns:
            dict: Dictionary containing multiple plot figures
        """
        dashboard = {}
        
        try:
            # HMPI Distribution
            dashboard['hmpi_dist'] = self.plot_hmpi_distribution(data)
            
            # Quality Categories
            dashboard['quality_pie'] = self.plot_quality_categories(data)
            
            # Metal correlation heatmap
            dashboard['correlation'] = self.plot_correlation_heatmap(data)
            
            # Geospatial plot if coordinates available
            if 'Latitude' in data.columns and 'Longitude' in data.columns:
                dashboard['geospatial'] = self.plot_geospatial_heatmap(data)
            
            return dashboard
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            return {}
    
    def export_plots_as_html(self, figures_dict, filename="hmpi_plots.html"):
        """
        Export multiple plots as HTML file
        
        Args:
            figures_dict (dict): Dictionary of plot figures
            filename (str): Output filename
            
        Returns:
            str: HTML content
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HMPI Analysis Plots</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Heavy Metal Pollution Indices Analysis</h1>
        """
        
        for plot_name, fig in figures_dict.items():
            if fig is not None:
                html_content += f"<div id='{plot_name}'></div>"
                html_content += f"<script>Plotly.newPlot('{plot_name}', {fig.to_json()});</script>"
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
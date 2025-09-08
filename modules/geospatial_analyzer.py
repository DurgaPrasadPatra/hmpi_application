# geospatial_analyzer.py
# modules/geospatial_analyzer.py
import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GeospatialAnalyzer:
    """
    Geospatial Analysis Module for Heavy Metal Pollution Data
    
    Provides comprehensive spatial analysis capabilities including
    spatial clustering, interpolation, hotspot detection, and
    spatial statistics for groundwater contamination assessment.
    """
    
    def __init__(self):
        """Initialize with default spatial analysis parameters"""
        self.default_map_style = "OpenStreetMap"
        self.contamination_threshold = 100  # HMPI threshold for contamination
        
        # Color schemes for different contamination levels
        self.contamination_colors = {
            'Excellent': '#2E8B57',      # Sea Green
            'Good': '#32CD32',           # Lime Green  
            'Poor': '#FFD700',           # Gold
            'Very Poor': '#FF6347',      # Tomato
            'Unsuitable': '#DC143C'      # Crimson
        }
        
        # Marker sizes based on contamination level
        self.marker_sizes = {
            'Excellent': 6,
            'Good': 8,
            'Poor': 10,
            'Very Poor': 12,
            'Unsuitable': 15
        }
    
    def validate_spatial_data(self, data):
        """
        Validate spatial data for geospatial analysis
        
        Args:
            data (pd.DataFrame): Input data with coordinates
            
        Returns:
            dict: Validation results
        """
        errors = []
        warnings = []
        
        # Check for coordinate columns
        if 'Latitude' not in data.columns:
            errors.append("Latitude column not found")
        if 'Longitude' not in data.columns:
            errors.append("Longitude column not found")
            
        if errors:
            return {
                'is_valid': False,
                'errors': errors,
                'warnings': warnings,
                'valid_samples': 0
            }
        
        # Check coordinate validity
        valid_coords = data.dropna(subset=['Latitude', 'Longitude'])
        
        if len(valid_coords) == 0:
            errors.append("No valid coordinates found")
        
        # Check coordinate ranges
        invalid_lat = valid_coords[
            ~valid_coords['Latitude'].between(-90, 90)
        ]
        invalid_lon = valid_coords[
            ~valid_coords['Longitude'].between(-180, 180)
        ]
        
        if len(invalid_lat) > 0:
            warnings.append(f"{len(invalid_lat)} samples with invalid latitude values")
        if len(invalid_lon) > 0:
            warnings.append(f"{len(invalid_lon)} samples with invalid longitude values")
        
        # Initialize spatial extent variables
        lat_range = 0
        lon_range = 0
        # Check for spatial extent
        if len(valid_coords) > 1:
            lat_range = valid_coords['Latitude'].max() - valid_coords['Latitude'].min()
            lon_range = valid_coords['Longitude'].max() - valid_coords['Longitude'].min()
            
            if lat_range < 0.001 and lon_range < 0.001:
                warnings.append("Very small spatial extent - points may be clustered")
        
        # Check for duplicate coordinates
        coord_pairs = valid_coords[['Latitude', 'Longitude']].round(6)
        duplicates = coord_pairs.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"{duplicates} samples have duplicate coordinates")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'valid_samples': len(valid_coords),
            'total_samples': len(data),
            'spatial_extent': {
                'lat_range': lat_range if len(valid_coords) > 1 else 0,
                'lon_range': lon_range if len(valid_coords) > 1 else 0
            } if len(valid_coords) > 1 else None
        }
    
    def create_contamination_map(self, data, map_title="Heavy Metal Contamination Map"):
        """
        Create interactive contamination map with folium
        
        Args:
            data (pd.DataFrame): Data with coordinates and HMPI values
            map_title (str): Title for the map
            
        Returns:
            folium.Map: Interactive map object
        """
        # Validate and clean data
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        if len(valid_data) == 0:
            raise ValueError("No valid spatial data available for mapping")
        
        # Calculate map center
        center_lat = valid_data['Latitude'].mean()
        center_lon = valid_data['Longitude'].mean()
        
        # Determine zoom level based on spatial extent
        lat_range = valid_data['Latitude'].max() - valid_data['Latitude'].min()
        lon_range = valid_data['Longitude'].max() - valid_data['Longitude'].min()
        max_range = max(lat_range, lon_range)
        
        if max_range > 1:
            zoom_start = 8
        elif max_range > 0.1:
            zoom_start = 10
        elif max_range > 0.01:
            zoom_start = 12
        else:
            zoom_start = 14
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=self.default_map_style
        )
        
        # Add markers for each sample
        for idx, row in valid_data.iterrows():
            self._add_sample_marker(m, row)
        
        # Add map legend
        self._add_map_legend(m)
        
        # Add map title
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{map_title}</b></h3>
        '''
        m.get_root().add_child(folium.Element(title_html))
        
        return m
    
    def _add_sample_marker(self, map_obj, sample_data):
        """Add individual sample marker to map"""
        lat, lon = sample_data['Latitude'], sample_data['Longitude']
        hmpi = sample_data['HMPI']
        sample_id = sample_data.get('Sample_ID', 'Unknown')
        quality_category = sample_data.get('Quality_Category', 'Unknown')
        
        # Determine marker properties based on contamination level
        color = self.contamination_colors.get(quality_category, '#808080')
        size = self.marker_sizes.get(quality_category, 8)
        
        # Create popup content
        popup_html = self._create_popup_content(sample_data)
        
        # Add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=size,
            popup=folium.Popup(popup_html, max_width=300),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"Sample: {sample_id} | HMPI: {hmpi:.2f}"
        ).add_to(map_obj)
    
    def _create_popup_content(self, sample_data):
        """Create HTML content for map popups"""
        sample_id = sample_data.get('Sample_ID', 'Unknown')
        hmpi = sample_data['HMPI']
        quality = sample_data.get('Quality_Category', 'Unknown')
        location = sample_data.get('Location', 'Not specified')
        
        # Add heavy metal concentrations if available
        metals = ['As', 'Cd', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Pb', 'Zn']
        metal_info = ""
        
        for metal in metals:
            if metal in sample_data and pd.notna(sample_data[metal]):
                metal_info += f"<br><b>{metal}:</b> {sample_data[metal]:.4f} mg/L"
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px;">
            <b>Sample ID:</b> {sample_id}<br>
            <b>HMPI:</b> {hmpi:.2f}<br>
            <b>Quality:</b> {quality}<br>
            <b>Location:</b> {location}<br>
            <b>Coordinates:</b> {sample_data['Latitude']:.4f}, {sample_data['Longitude']:.4f}
            {metal_info}
        </div>
        """
        return popup_html
    
    def _add_map_legend(self, map_obj):
        """Add legend to the map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>HMPI Categories</b></p>
        <p><i class="fa fa-circle" style="color:#2E8B57"></i> Excellent (≤25)</p>
        <p><i class="fa fa-circle" style="color:#32CD32"></i> Good (25-50)</p>
        <p><i class="fa fa-circle" style="color:#FFD700"></i> Poor (50-100)</p>
        <p><i class="fa fa-circle" style="color:#FF6347"></i> Very Poor (100-200)</p>
        <p><i class="fa fa-circle" style="color:#DC143C"></i> Unsuitable (>200)</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def create_heatmap(self, data, radius=20, blur=15):
        """
        Create contamination heatmap
        
        Args:
            data (pd.DataFrame): Data with coordinates and HMPI values
            radius (int): Heat radius for each point
            blur (int): Blur radius for heatmap
            
        Returns:
            folium.Map: Map with heatmap overlay
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        if len(valid_data) == 0:
            raise ValueError("No valid data for heatmap")
        
        # Create base map
        center_lat = valid_data['Latitude'].mean()
        center_lon = valid_data['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=self.default_map_style
        )
        
        # Prepare heatmap data
        heat_data = []
        for idx, row in valid_data.iterrows():
            # Normalize HMPI for heatmap intensity (0-1 scale)
            intensity = min(row['HMPI'] / 200, 1.0)  # Cap at HMPI 200
            heat_data.append([row['Latitude'], row['Longitude'], intensity])
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=1,
            gradient={0.0: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        return m
    
    def spatial_clustering_analysis(self, data, method='dbscan', contaminated_only=True):
        """
        Perform spatial clustering analysis
        
        Args:
            data (pd.DataFrame): Spatial data
            method (str): Clustering method ('dbscan', 'kmeans')
            contaminated_only (bool): Cluster only contaminated samples
            
        Returns:
            dict: Clustering results and analysis
        """
        # Prepare data
        if contaminated_only:
            cluster_data = data[data['HMPI'] > self.contamination_threshold]
        else:
            cluster_data = data.copy()
        
        valid_data = cluster_data.dropna(subset=['Latitude', 'Longitude'])
        
        if len(valid_data) < 3:
            return {
                'error': 'Insufficient data points for clustering analysis',
                'clusters': [],
                'n_clusters': 0
            }
        
        # Extract coordinates
        coords = valid_data[['Latitude', 'Longitude']].values
        
        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        if method.lower() == 'dbscan':
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(coords_scaled)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        elif method.lower() == 'kmeans':
            # Determine optimal number of clusters (max 5)
            max_k = min(5, len(valid_data) // 2)
            if max_k < 2:
                max_k = 2
            
            # Use elbow method to find optimal k
            inertias = []
            k_range = range(2, max_k + 1)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(coords_scaled)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection (use k=3 as default)
            optimal_k = 3 if len(k_range) >= 2 else 2
            
            clustering = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = clustering.fit_predict(coords_scaled)
            n_clusters = optimal_k
        
        # Add cluster labels to data
        valid_data_copy = valid_data.copy()
        valid_data_copy['Cluster'] = labels
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(valid_data_copy, labels)
        
        return {
            'method': method,
            'n_clusters': n_clusters,
            'labels': labels,
            'clustered_data': valid_data_copy,
            'cluster_analysis': cluster_analysis,
            'contaminated_only': contaminated_only
        }
    
    def _analyze_clusters(self, data, labels):
        """Analyze cluster characteristics"""
        cluster_stats = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            cluster_points = data[data['Cluster'] == label]
            
            cluster_stats[f'Cluster_{label}'] = {
                'n_samples': len(cluster_points),
                'mean_hmpi': cluster_points['HMPI'].mean(),
                'max_hmpi': cluster_points['HMPI'].max(),
                'min_hmpi': cluster_points['HMPI'].min(),
                'center_lat': cluster_points['Latitude'].mean(),
                'center_lon': cluster_points['Longitude'].mean(),
                'lat_range': cluster_points['Latitude'].max() - cluster_points['Latitude'].min(),
                'lon_range': cluster_points['Longitude'].max() - cluster_points['Longitude'].min(),
                'quality_distribution': cluster_points['Quality_Category'].value_counts().to_dict()
            }
        
        return cluster_stats
    
    def spatial_autocorrelation_analysis(self, data, distance_threshold=0.01):
        """
        Calculate spatial autocorrelation using Moran's I
        
        Args:
            data (pd.DataFrame): Spatial data
            distance_threshold (float): Distance threshold for spatial weights
            
        Returns:
            dict: Spatial autocorrelation results
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        if len(valid_data) < 5:
            return {
                'error': 'Insufficient data points for spatial autocorrelation analysis',
                'morans_i': None
            }
        
        # Calculate distance matrix
        coords = valid_data[['Latitude', 'Longitude']].values
        distances = pdist(coords, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Create spatial weights matrix (binary: 1 if within threshold, 0 otherwise)
        weights = (distance_matrix <= distance_threshold) & (distance_matrix > 0)
        weights = weights.astype(int)
        
        # Calculate Moran's I
        hmpi_values = valid_data['HMPI'].values
        n = len(hmpi_values)
        
        # Standardize values
        hmpi_std = (hmpi_values - hmpi_values.mean()) / hmpi_values.std()
        
        # Calculate Moran's I statistic
        numerator = 0
        denominator = 0
        w_sum = weights.sum()
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * hmpi_std[i] * hmpi_std[j]
            denominator += hmpi_std[i] ** 2
        
        if w_sum > 0 and denominator > 0:
            morans_i = (n / w_sum) * (numerator / denominator)
            
            # Calculate expected value and variance (simplified)
            expected_i = -1 / (n - 1)
            
            # Z-score calculation (simplified)
            if w_sum > 0:
                variance_i = (n / ((n - 1) * (n - 2) * (n - 3))) * (
                    (n ** 2 - 3 * n + 3) * w_sum - n * (weights ** 2).sum() + 3 * w_sum ** 2
                )
                if variance_i > 0:
                    z_score = (morans_i - expected_i) / np.sqrt(variance_i)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = 0
                    p_value = 1
            else:
                z_score = 0
                p_value = 1
        else:
            morans_i = 0
            expected_i = -1 / (n - 1)
            z_score = 0
            p_value = 1
        
        # Interpret results
        interpretation = self._interpret_morans_i(morans_i, p_value)
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': interpretation,
            'n_samples': n,
            'distance_threshold': distance_threshold
        }
    
    def _interpret_morans_i(self, morans_i, p_value):
        """Interpret Moran's I results"""
        alpha = 0.05
        
        if p_value > alpha:
            return "No significant spatial autocorrelation detected"
        
        if morans_i > 0:
            return f"Significant positive spatial autocorrelation (I = {morans_i:.3f}): Similar values tend to cluster together"
        elif morans_i < 0:
            return f"Significant negative spatial autocorrelation (I = {morans_i:.3f}): Dissimilar values tend to be adjacent"
        else:
            return "Random spatial distribution"
    
    def identify_hotspots(self, data, method='getis_ord', significance_level=0.05):
        """
        Identify contamination hotspots using local spatial statistics
        
        Args:
            data (pd.DataFrame): Spatial data
            method (str): Method for hotspot detection
            significance_level (float): Statistical significance level
            
        Returns:
            dict: Hotspot analysis results
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        if len(valid_data) < 5:
            return {
                'error': 'Insufficient data for hotspot analysis',
                'hotspots': [],
                'coldspots': []
            }
        
        # Simple hotspot detection based on local clustering of high values
        coords = valid_data[['Latitude', 'Longitude']].values
        hmpi_values = valid_data['HMPI'].values
        
        # Calculate local indicators
        hotspots = []
        coldspots = []
        
        for i, (coord, hmpi) in enumerate(zip(coords, hmpi_values)):
            # Find nearby points (within reasonable distance)
            distances = np.sqrt(((coords - coord) ** 2).sum(axis=1))
            nearby_mask = (distances <= 0.01) & (distances > 0)  # Adjust threshold as needed
            
            if nearby_mask.any():
                nearby_values = hmpi_values[nearby_mask]
                local_mean = nearby_values.mean()
                overall_mean = hmpi_values.mean()
                
                # High-High clusters (hotspots)
                if hmpi > overall_mean and local_mean > overall_mean:
                    hotspots.append({
                        'index': i,
                        'sample_id': valid_data.iloc[i].get('Sample_ID', f'Sample_{i}'),
                        'latitude': coord[0],
                        'longitude': coord[1],
                        'hmpi': hmpi,
                        'local_mean': local_mean,
                        'intensity': (hmpi - overall_mean) / hmpi_values.std()
                    })
                
                # Low-Low clusters (coldspots)
                elif hmpi < overall_mean and local_mean < overall_mean:
                    coldspots.append({
                        'index': i,
                        'sample_id': valid_data.iloc[i].get('Sample_ID', f'Sample_{i}'),
                        'latitude': coord[0],
                        'longitude': coord[1],
                        'hmpi': hmpi,
                        'local_mean': local_mean,
                        'intensity': (overall_mean - hmpi) / hmpi_values.std()
                    })
        
        return {
            'method': method,
            'hotspots': sorted(hotspots, key=lambda x: x['intensity'], reverse=True),
            'coldspots': sorted(coldspots, key=lambda x: x['intensity'], reverse=True),
            'n_hotspots': len(hotspots),
            'n_coldspots': len(coldspots),
            'significance_level': significance_level
        }
    
    def spatial_interpolation(self, data, method='idw', grid_resolution=50):
        """
        Perform spatial interpolation of contamination values
        
        Args:
            data (pd.DataFrame): Spatial data
            method (str): Interpolation method ('idw', 'kriging')
            grid_resolution (int): Grid resolution for interpolation
            
        Returns:
            dict: Interpolation results and grid data
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        if len(valid_data) < 4:
            return {
                'error': 'Insufficient data points for spatial interpolation',
                'grid': None
            }
        
        # Create interpolation grid
        lat_min, lat_max = valid_data['Latitude'].min(), valid_data['Latitude'].max()
        lon_min, lon_max = valid_data['Longitude'].min(), valid_data['Longitude'].max()
        
        # Add buffer around data points
        lat_buffer = (lat_max - lat_min) * 0.1
        lon_buffer = (lon_max - lon_min) * 0.1
        
        lat_grid = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, grid_resolution)
        lon_grid = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, grid_resolution)
        
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
        
        # Perform IDW interpolation
        if method.lower() == 'idw':
            interpolated_values = self._idw_interpolation(
                valid_data, lat_mesh.flatten(), lon_mesh.flatten()
            )
            interpolated_grid = interpolated_values.reshape(lat_mesh.shape)
        
        return {
            'method': method,
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'interpolated_grid': interpolated_grid,
            'extent': [lon_min - lon_buffer, lon_max + lon_buffer, 
                      lat_min - lat_buffer, lat_max + lat_buffer]
        }
    
    def _idw_interpolation(self, data, target_lats, target_lons, power=2):
        """
        Inverse Distance Weighting interpolation
        
        Args:
            data (pd.DataFrame): Source data points
            target_lats (np.array): Target latitude points
            target_lons (np.array): Target longitude points
            power (float): IDW power parameter
            
        Returns:
            np.array: Interpolated values
        """
        source_coords = data[['Latitude', 'Longitude']].values
        source_values = data['HMPI'].values
        
        interpolated = np.zeros(len(target_lats))
        
        for i, (target_lat, target_lon) in enumerate(zip(target_lats, target_lons)):
            # Calculate distances to all source points
            distances = np.sqrt(
                (source_coords[:, 0] - target_lat) ** 2 + 
                (source_coords[:, 1] - target_lon) ** 2
            )
            
            # Avoid division by zero
            distances[distances == 0] = 1e-10
            
            # Calculate weights
            weights = 1.0 / (distances ** power)
            weights_sum = weights.sum()
            
            # Interpolate value
            interpolated[i] = (weights * source_values).sum() / weights_sum
        
        return interpolated
    
    def generate_spatial_summary(self, data):
        """
        Generate comprehensive spatial analysis summary
        
        Args:
            data (pd.DataFrame): Spatial data
            
        Returns:
            dict: Spatial analysis summary
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude', 'HMPI'])
        
        summary = {
            'data_coverage': {
                'total_samples': len(data),
                'samples_with_coordinates': len(valid_data),
                'coverage_percentage': (len(valid_data) / len(data)) * 100 if len(data) > 0 else 0
            },
            'spatial_extent': {},
            'contamination_distribution': {},
            'spatial_patterns': {}
        }
        
        if len(valid_data) == 0:
            summary['error'] = "No valid spatial data available"
            return summary
        
        # Spatial extent analysis
        summary['spatial_extent'] = {
            'latitude_range': [float(valid_data['Latitude'].min()), float(valid_data['Latitude'].max())],
            'longitude_range': [float(valid_data['Longitude'].min()), float(valid_data['Longitude'].max())],
            'center_point': [float(valid_data['Latitude'].mean()), float(valid_data['Longitude'].mean())],
            'bounding_box_area_km2': self._calculate_bounding_box_area(valid_data)
        }
        
        # Contamination distribution
        contaminated = valid_data[valid_data['HMPI'] > self.contamination_threshold]
        summary['contamination_distribution'] = {
            'contaminated_locations': len(contaminated),
            'clean_locations': len(valid_data) - len(contaminated),
            'contamination_density': (len(contaminated) / len(valid_data)) * 100,
            'average_hmpi': float(valid_data['HMPI'].mean()),
            'max_contamination_hmpi': float(valid_data['HMPI'].max()),
            'spatial_contamination_pattern': self._assess_contamination_pattern(valid_data)
        }
        
        # Basic spatial patterns
        if len(valid_data) >= 5:
            autocorr_result = self.spatial_autocorrelation_analysis(valid_data)
            summary['spatial_patterns'] = {
                'spatial_autocorrelation': autocorr_result.get('interpretation', 'Unable to calculate'),
                'morans_i': autocorr_result.get('morans_i'),
                'clustering_potential': len(valid_data) >= 3
            }
        
        return summary
    
    def _calculate_bounding_box_area(self, data):
        """Calculate approximate area of bounding box in km²"""
        lat_range = data['Latitude'].max() - data['Latitude'].min()
        lon_range = data['Longitude'].max() - data['Longitude'].min()
        
        # Rough approximation: 1 degree ≈ 111 km
        area_km2 = lat_range * lon_range * (111.32 ** 2)
        return float(area_km2)
    
    def _assess_contamination_pattern(self, data):
        """Assess overall contamination spatial pattern"""
        contaminated = data[data['HMPI'] > self.contamination_threshold]
        
        if len(contaminated) == 0:
            return "No contamination detected"
        elif len(contaminated) == len(data):
            return "Widespread contamination"
        elif len(contaminated) / len(data) > 0.7:
            return "Extensive contamination"
        elif len(contaminated) / len(data) > 0.3:
            return "Moderate contamination"
        else:
            return "Localized contamination"
    
    def export_spatial_data(self, data, filename="spatial_analysis.geojson"):
        """
        Export spatial data to GeoJSON format
        
        Args:
            data (pd.DataFrame): Spatial data
            filename (str): Output filename
            
        Returns:
            dict: GeoJSON data structure
        """
        valid_data = data.dropna(subset=['Latitude', 'Longitude'])
        
        features = []
        for idx, row in valid_data.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['Longitude'], row['Latitude']]
                },
                "properties": row.drop(['Latitude', 'Longitude']).to_dict()
            }
            features.append(feature)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        # Optionally, save to file
        with open(filename, "w", encoding="utf-8") as f:
            import json
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        return geojson
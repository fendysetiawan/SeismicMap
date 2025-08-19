from math import ceil, floor
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from PIL import Image
import base64
from io import BytesIO

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Map boundaries and settings
CONTINENTAL_US_BOUNDS = {
    'lat_min': 24.5, 'lat_max': 49.5,
    'lon_min': -125, 'lon_max': -66.6
}
GRID_RESOLUTION = 0.05
DEFAULT_ZOOM_RADIUS = 1.5
SEARCH_RADIUS = 0.1
DEFAULT_COLORS = 10

# Default location for testing
DEFAULT_LOCATION = {
    'lat': 37.80423914364421,
    'lon': -122.27615639197262,
    'address': "601 12th Street, Oakland 94607"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pil_image_to_base64_str(path):
    """Convert image file to base64 string for embedding in Plotly"""
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def format_parameter_name(parameter):
    """Format parameter name for display with LaTeX subscript"""
    return parameter.replace('_', '_{') + '}'

def validate_location(lat, lon):
    """Check if location is within continental US bounds"""
    return (CONTINENTAL_US_BOUNDS['lat_min'] <= lat <= CONTINENTAL_US_BOUNDS['lat_max'] and
            CONTINENTAL_US_BOUNDS['lon_min'] <= lon <= CONTINENTAL_US_BOUNDS['lon_max'])

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

@st.cache_data
def load_seismic_data():
    """Load and cache seismic data files"""
    asce_16 = pd.read_csv("data/ASCE7-16_summarySA.csv")
    asce_22 = pd.read_csv("data/ASCE7-22_summarySA.csv")
    ratio = pd.read_csv("data/ASCE7_ratio.csv")
    return asce_16, asce_22, ratio

def get_dataset_by_version(version, asce_16, asce_22, ratio):
    """Return appropriate dataset based on selected version"""
    if version == "ASCE 7-16":
        return asce_16
    elif version == "ASCE 7-22":
        return asce_22
    else:
        return ratio

def create_interpolation_grid():
    """Create regular grid for interpolation"""
    lon_grid = np.arange(CONTINENTAL_US_BOUNDS['lon_min'], 
                        CONTINENTAL_US_BOUNDS['lon_max'] + GRID_RESOLUTION, 
                        GRID_RESOLUTION)
    lat_grid = np.arange(CONTINENTAL_US_BOUNDS['lat_min'], 
                        CONTINENTAL_US_BOUNDS['lat_max'] + GRID_RESOLUTION, 
                        GRID_RESOLUTION)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    return lon_grid, lat_grid, lon_mesh, lat_mesh

def interpolate_data(lat, lon, z, lon_mesh, lat_mesh):
    """Interpolate data onto regular grid"""
    return griddata(
        points=(lon, lat),
        values=z,
        xi=(lon_mesh, lat_mesh),
        method='nearest'
    )

def calculate_global_bounds(version, parameter, asce_16, asce_22, ratio):
    """Calculate global min/max values for parameter"""
    if version == "ASCE 7-16":
        global_min = floor(asce_16[parameter].min() * 10) / 10
        global_max = ceil(asce_16[parameter].max() * 10) / 10
    elif version == "ASCE 7-22":
        global_min = floor(asce_22[parameter].min() * 10) / 10
        global_max = ceil(asce_22[parameter].max() * 10) / 10
    else:
        global_min = floor(ratio[parameter].min() * 10) / 10
        global_max = ceil(ratio[parameter].max() * 10) / 10
    return global_min, global_max

# =============================================================================
# GEOCODING FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=False)
def geocode_address(address):
    """Geocode address to latitude/longitude coordinates"""
    if not address:
        return None
    geolocator = Nominatim(user_agent="SeismicMap")
    return geolocator.geocode(address)

def interpolate_location_value(lat_input, lon_input, lat, lon, z):
    """Interpolate parameter value at specific location"""
    if not validate_location(lat_input, lon_input):
        return None, "Location is outside the continental US bounds."
    
    # Find nearby points for efficient interpolation
    mask = ((lon >= lon_input - SEARCH_RADIUS) & (lon <= lon_input + SEARCH_RADIUS) &
            (lat >= lat_input - SEARCH_RADIUS) & (lat <= lat_input + SEARCH_RADIUS))
    
    if not np.any(mask):
        return None, "No data points found near the selected location."
    
    # Interpolate using nearby points only
    nearby_lon, nearby_lat, nearby_z = lon[mask], lat[mask], z[mask]
    point_value = griddata(
        points=(nearby_lon, nearby_lat),
        values=nearby_z,
        xi=(lon_input, lat_input),
        method='nearest'
    )
    
    # Convert to scalar if needed
    if hasattr(point_value, 'item'):
        point_value = point_value.item()
    
    if np.isnan(point_value):
        return None, "Location is outside the data coverage area or interpolation failed."
    
    return point_value, None

# =============================================================================
# MAP AND VISUALIZATION FUNCTIONS
# =============================================================================

@st.cache_data
def get_map_images():
    """Cache map background images for better performance"""
    usa_map_base = pil_image_to_base64_str("data/USA_map.png")
    usa_mask_base = pil_image_to_base64_str("data/USA_map_transparent_v3.png")
    return usa_map_base, usa_mask_base

@st.cache_data
def create_base_map_layout(show_location_analysis=False, lat_input=None, lon_input=None):
    """Create base map layout with background images and location markers"""
    fig = go.Figure()
    
    # Add map background images
    usa_map_base, usa_mask_base = get_map_images()
    
    fig.add_layout_image(dict(
        source=usa_map_base,
        xref="x", yref="y",
        x=-125.5, y=49.8,
        sizex=59, sizey=25.6,
        sizing="stretch", opacity=0.5, layer="below"
    ))
    
    fig.add_layout_image(dict(
        source=usa_mask_base,
        xref="x", yref="y",
        x=-125.5, y=49.8,
        sizex=59, sizey=25.6,
        sizing="stretch", opacity=1.0, layer="above"
    ))
    
    # Add location marker if enabled
    if show_location_analysis and lat_input is not None and lon_input is not None:
        if validate_location(lat_input, lon_input):
            # Add pin marker
            fig.add_trace(go.Scatter(
                x=[lon_input], y=[lat_input],
                mode='markers',
                marker=dict(
                    size=20, color='red', symbol='circle',
                    line=dict(width=2, color='white'), opacity=0.8
                ),
                name='Selected Location',
                showlegend=False, 
                hovertemplate=f'<b>📍 Selected Location:</b><br>' +
                            f'Latitude: {lat_input:.6f}<br>' +
                            f'Longitude: {lon_input:.6f}<br><extra></extra>',
            ))
            
            # Add crosshair lines
            for direction in ['horizontal', 'vertical']:
                if direction == 'horizontal':
                    x_vals = [lon_input - 0.1, lon_input + 0.1]
                    y_vals = [lat_input, lat_input]
                else:
                    x_vals = [lon_input, lon_input]
                    y_vals = [lat_input - 0.1, lat_input + 0.1]
                
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color='red', width=1, dash='solid'),
                    showlegend=False, hoverinfo='skip',
                ))
            
            # Zoom to location
            x_range = [lon_input - DEFAULT_ZOOM_RADIUS, lon_input + DEFAULT_ZOOM_RADIUS]
            y_range = [lat_input - DEFAULT_ZOOM_RADIUS, lat_input + DEFAULT_ZOOM_RADIUS]
            
            # Ensure zoom stays within bounds
            x_range[0] = max(x_range[0], CONTINENTAL_US_BOUNDS['lon_min'])
            x_range[1] = min(x_range[1], CONTINENTAL_US_BOUNDS['lon_max'])
            y_range[0] = max(y_range[0], CONTINENTAL_US_BOUNDS['lat_min'])
            y_range[1] = min(y_range[1], CONTINENTAL_US_BOUNDS['lat_max'])
            
            fig.update_layout(
                xaxis=dict(title="Longitude", range=x_range, scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Latitude", range=y_range),
                height=800, margin={"r": 10, "t": 50, "l": 10, "b": 10}
            )
        else:
            # Default view
            fig.update_layout(
                xaxis=dict(title="Longitude", range=[-125, -65], scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Latitude", range=[24, 50]),
                height=800, margin={"r": 10, "t": 50, "l": 10, "b": 10}
            )
    else:
        # Default view
        fig.update_layout(
            xaxis=dict(title="Longitude", range=[-125, -65], scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Latitude", range=[24, 50]),
            height=800, margin={"r": 10, "t": 50, "l": 10, "b": 10}
        )
    
    return fig

@st.cache_data
def create_contour_for_region(lon_min, lon_max, lat_min, lat_max, grid_z, lon_grid, lat_grid, 
                             custom_min, custom_max, n_colors, parameter):
    """Create contour trace for a specific geographic region"""
    # Find grid indices for the specified region
    lon_mask = (lon_grid >= lon_min) & (lon_grid <= lon_max)
    lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)
    
    # Extract region data
    region_lon = lon_grid[lon_mask]
    region_lat = lat_grid[lat_mask]
    region_z = grid_z[np.ix_(lat_mask, lon_mask)]
    
    return go.Contour(
        z=region_z, x=region_lon, y=region_lat,
        colorscale='Jet', zmin=custom_min, zmax=custom_max,
        colorbar_title="Sa (g)",
        contours=dict(
            showlabels=True, start=custom_min, end=custom_max,
            size=(custom_max - custom_min) / n_colors
        ),
        opacity=0.5,
        hovertemplate='<b>Latitude:</b> %{y:.4f}<br>' +
                    '<b>Longitude:</b> %{x:.4f}<br>' +
                    f'<b>{parameter.replace("_", "<sub>")}</sub>:</b> %{{z:.2f}}<br>' +
                    '<extra></extra>',
        showlegend=False
    )

def update_contour_data(fig, grid_z, lon_grid, lat_grid, custom_min, custom_max, n_colors, 
                       parameter, version, show_location_analysis=False, view_region=None):
    """Update contour data on existing figure"""
    # Clear existing contour data
    fig.data = [trace for trace in fig.data if trace.type != 'contour']
    
    if view_region is None:
        # Default view - show full continental US
        view_region = {
            'lon_min': CONTINENTAL_US_BOUNDS['lon_min'],
            'lon_max': CONTINENTAL_US_BOUNDS['lon_max'],
            'lat_min': CONTINENTAL_US_BOUNDS['lat_min'],
            'lat_max': CONTINENTAL_US_BOUNDS['lat_max']
        }
    
    # Create and add contour trace
    contour_trace = create_contour_for_region(
        view_region['lon_min'], view_region['lon_max'],
        view_region['lat_min'], view_region['lat_max'],
        grid_z, lon_grid, lat_grid, custom_min, custom_max, n_colors, parameter
    )
    fig.add_trace(contour_trace)
    
    # Update title
    title_suffix = " (📍 Zoomed to Location)" if show_location_analysis else ""
    fig.update_layout(title=f"{version} - {parameter.replace("_", "<sub>")}</sub> Contour Map{title_suffix}")
    
    return fig

# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_parameter_controls():
    """Create parameter selection controls"""
    version = st.selectbox("ASCE Code Version", ["ASCE 7-16", "ASCE 7-22", "Ratio 7-22 / 7-16"])
    parameter = st.selectbox("Seismic Parameter", ["S_MS", "S_M1", "S_DS", "S_D1"])
    return version, parameter

def create_range_controls(global_min, global_max, default_colors):
    """Create range and color controls"""
    custom_min = st.number_input("Minimum Value", value=float(global_min), format="%.2f", step=0.01)
    custom_max = st.number_input("Maximum Value", value=float(global_max), format="%.2f", step=0.01)
    n_colors = st.slider("Number of Colors", min_value=2, max_value=20, value=default_colors, step=1)
    return custom_min, custom_max, n_colors

def create_location_analysis_ui():
    """Create location analysis user interface"""
    coord_mode = st.radio("Location Input Method", ["Address", "Manual Lat/Lon"], 
                         index=st.session_state.get('coord_mode_index', 0),
                         key='coord_mode')
    
    if coord_mode == "Manual Lat/Lon":
        # Preserve lat/lon values across parameter changes
        default_lat = st.session_state.get('manual_lat', DEFAULT_LOCATION['lat'])
        default_lon = st.session_state.get('manual_lon', DEFAULT_LOCATION['lon'])
        
        lat_input = st.number_input("Latitude", value=default_lat, 
                                  format="%.8f", min_value=24.0, max_value=50.0,
                                  key='manual_lat')
        lon_input = st.number_input("Longitude", value=default_lon, 
                                  format="%.8f", min_value=-130.0, max_value=-60.0,
                                  key='manual_lon')
    else:
        # Preserve address across parameter changes
        default_address = st.session_state.get('address_input', DEFAULT_LOCATION['address'])
        address = st.text_input("Enter Address", value=default_address,
                               placeholder=DEFAULT_LOCATION['address'],
                               key='address_input')
        location = geocode_address(address.strip())
        
        if location:
            lat_input, lon_input = location.latitude, location.longitude
            st.info(f"🔍 {location.address}\n Latitude: {lat_input:.8f}, Longitude: {lon_input:.8f}")
        else:
            if address:
                st.warning("Unable to geocode that address. Try refining it.")
            lat_input, lon_input = DEFAULT_LOCATION['lat'], DEFAULT_LOCATION['lon']
    
    return lat_input, lon_input

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(layout="wide")
    
    # Title and header
    st.title("Seismic Parameter Contour Map")
    st.subheader("Contour Plot from ASCE 7-16 / ASCE 7-22 Data")
    
    # Load data
    asce_16, asce_22, ratio = load_seismic_data()
    
    # Create interpolation grid (only need to do this once)
    lon_grid, lat_grid, lon_mesh, lat_mesh = create_interpolation_grid()
    
    # Initialize variables
    df = None
    lat, lon, z = None, None, None
    grid_z = None
    global_min, global_max = 0.0, 1.0
    
    # Create 3-column layout for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Selection")
        version, parameter = create_parameter_controls()
        
        # Update data based on user selection
        df = get_dataset_by_version(version, asce_16, asce_22, ratio)
        lat, lon, z = df["Latitude"].values, df["Longitude"].values, df[parameter].values
        
        # Re-interpolate data with new parameters
        grid_z = interpolate_data(lat, lon, z, lon_mesh, lat_mesh)
        
        # Show global bounds
        global_min, global_max = calculate_global_bounds(version, parameter, asce_16, asce_22, ratio)
        st.info(f"Current data range for ${format_parameter_name(parameter)}$: {global_min:.2f} to {global_max:.2f}")
    
    with col2:
        st.subheader("Range Settings")
        custom_min, custom_max, n_colors = create_range_controls(global_min, global_max, DEFAULT_COLORS)

        # Validate min/max
        if custom_min >= custom_max:
            st.error("Minimum value must be less than maximum value!")
            st.stop()
    
    with col3:
        st.subheader("Display Options")
        
        # Performance options
        enable_lazy_loading = st.checkbox(
            "Enable Lazy Loading (Faster Performance)", 
            value=st.session_state.get('enable_lazy_loading', True),
            key='enable_lazy_loading',
            help="Only render contours for the visible region to improve performance"
        )
        
        # Location analysis
        show_location_analysis = st.checkbox(
            f"Get {version} ${format_parameter_name(parameter)}$ at Location", 
            value=st.session_state.get('show_location_analysis', False),
            key='show_location_analysis'
        )
        
        if show_location_analysis:
            lat_input, lon_input = create_location_analysis_ui()
            
            # Interpolate and display location value
            point_value, error_msg = interpolate_location_value(lat_input, lon_input, lat, lon, z)
            
            if point_value is not None:
                st.success(f"**{version} ${format_parameter_name(parameter)}$ at ({lat_input:.8f}, {lon_input:.8f}):** {point_value:.2f}")
            elif error_msg:
                st.warning(error_msg)
    
    # Determine view region for lazy loading
    view_region = None
    if show_location_analysis and 'lat_input' in locals() and 'lon_input' in locals():
        # Calculate expanded view region (2x radius)
        zoom_radius = 2 * DEFAULT_ZOOM_RADIUS
        x_range = [lon_input - zoom_radius, lon_input + zoom_radius]
        y_range = [lat_input - zoom_radius, lat_input + zoom_radius]
        
        # Ensure bounds
        x_range[0] = max(x_range[0], CONTINENTAL_US_BOUNDS['lon_min'])
        x_range[1] = min(x_range[1], CONTINENTAL_US_BOUNDS['lon_max'])
        y_range[0] = max(y_range[0], CONTINENTAL_US_BOUNDS['lat_min'])
        y_range[1] = min(y_range[1], CONTINENTAL_US_BOUNDS['lat_max'])
        
        # Expand to 2x radius
        x_span, y_span = x_range[1] - x_range[0], y_range[1] - y_range[0]
        contour_x_center = (x_range[0] + x_range[1]) / 2
        contour_y_center = (y_range[0] + y_range[1]) / 2
        
        view_region = {
            'lon_min': max(contour_x_center - x_span, CONTINENTAL_US_BOUNDS['lon_min']),
            'lon_max': min(contour_x_center + x_span, CONTINENTAL_US_BOUNDS['lon_max']),
            'lat_min': max(contour_y_center - y_span, CONTINENTAL_US_BOUNDS['lat_min']),
            'lat_max': min(contour_y_center + y_span, CONTINENTAL_US_BOUNDS['lat_max'])
        }
    
    # Create base map
    fig = create_base_map_layout(show_location_analysis, 
                               lat_input if show_location_analysis else None,
                               lon_input if show_location_analysis else None)
    
    # Update contour data
    if enable_lazy_loading and view_region:
        fig = update_contour_data(
            fig, grid_z, lon_grid, lat_grid, custom_min, custom_max, n_colors,
            parameter, version, show_location_analysis, view_region
        )
    else:
        fig = update_contour_data(
            fig, grid_z, lon_grid, lat_grid, custom_min, custom_max, n_colors,
            parameter, version, show_location_analysis, None
        )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

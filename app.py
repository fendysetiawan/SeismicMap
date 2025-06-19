import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

from PIL import Image
import base64
from io import BytesIO

def pil_image_to_base64_str(path):
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# Wide layout
st.set_page_config(layout="wide")

# Load data
asce_16 = pd.read_csv("data/ASCE7-16_summarySA.csv")
asce_22 = pd.read_csv("data/ASCE7-22_summarySA.csv")
ratio = pd.read_csv("data/ASCE7_ratio.csv")

# UI controls
st.title("Seismic Parameter Contour Map")
st.subheader("Contour Plot from ASCE 7-16 / ASCE 7-22 Data")

col1, col2 = st.columns(2)
with col1:
    version = st.selectbox("ASCE Code Version", ["ASCE 7-16", "ASCE 7-22", "Ratio 7-22 / 7-16"])
with col2:
    parameter = st.selectbox("Seismic Parameter", ["S_MS", "S_M1", "S_DS", "S_D1"])

# Select dataset
if version == "ASCE 7-16":
    df = asce_16
elif version == "ASCE 7-22":
    df = asce_22
else:
    df = ratio

# Extract lat/lon/z
lat = df["Latitude"].values
lon = df["Longitude"].values
z = df[parameter].values

# Create regular grid to interpolate onto
lat_min, lat_max = 24.5, 49.5
lon_min, lon_max = -125, -66

lon_grid = np.arange(lon_min, lon_max + 0.05, 0.05)
lat_grid = np.arange(lat_min, lat_max + 0.05, 0.05)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Use robust 'nearest' interpolation
grid_z = griddata(
    points=(lon, lat),
    values=z,
    xi=(lon_mesh, lat_mesh),
    method='nearest'
)

# Global color limits from both datasets
if version in ("ASCE 7-16", "ASCE 7-22"):
    global_min = min(asce_16[parameter].min(), asce_22[parameter].min())
    global_max = max(asce_16[parameter].max(), asce_22[parameter].max())
else:
    global_min = 0.5
    global_max = 2.0

# Plot contour

fig = go.Figure(data=go.Contour(
    z=grid_z,
    x=lon_grid,
    y=lat_grid,
    colorscale='Jet',
    zmin=global_min,
    zmax=global_max,
    colorbar_title="Sa (g)",
    contours=dict(showlabels=True),
    opacity=0.5
))

usa_map_base = pil_image_to_base64_str("data/USA_map.png")
usa_mask_base = pil_image_to_base64_str("data/USA_map_transparent.png")

fig.add_layout_image(
    dict(
        source=usa_map_base,
        xref="x",
        yref="y",
        x=-125.5,
        y=49.8,
        sizex=59,
        sizey=25.6,
        sizing="stretch",
        opacity=0.5,
        layer="below"
    )
)

fig.add_layout_image(
    dict(
        source=usa_mask_base,
        xref="x",
        yref="y",
        x=-125.5,
        y=49.8,
        sizex=59,
        sizey=25.6,
        sizing="stretch",
        opacity=1.0,
        layer="above"
    )
)

fig.update_layout(
    title=f"{version} - {parameter} Contour Map",
    xaxis=dict(title="Longitude", range=[-125, -65], scaleanchor="y", scaleratio=1),
    yaxis=dict(title="Latitude", range=[24, 50]),
    height=800,
    margin={"r": 10, "t": 50, "l": 10, "b": 10}
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

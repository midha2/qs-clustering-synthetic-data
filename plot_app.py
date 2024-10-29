import streamlit as st
from PIL import Image

# Title of the application
st.title("Data Analysis Plots")

st.header("Histograms")
hist = Image.open('vote_distributions.png')
st.image(hist, use_column_width=True)


# Load and display plots
st.header("PCA Plots")

# Display PCA side-by-side plot
pca_image = Image.open('pca.png')
st.image(pca_image, caption='PCA Side by Side', use_column_width=True)

# Display Radar Plots
st.header("Radar Plots")

# Radar plot by category
radar_category_image = Image.open('category_radar.png')
st.image(radar_category_image, caption='Radar Plot by Category', use_column_width=True)

# Radar plot by category intensity
radar_category_intensity_image = Image.open('category_radar_intensity.png')
st.image(radar_category_intensity_image, caption='Radar Plot by Category Intensity', use_column_width=True)

# Radar plot side-by-side
radar_image = Image.open('radar.png')
st.image(radar_image, caption='Radar Plot Side by Side', use_column_width=True)

# Radar plot side-by-side with intensity
radar_intensity_image = Image.open('radar_intensity.png')
st.image(radar_intensity_image, caption='Radar Plot Side by Side with Intensity', use_column_width=True)

# Violin Plot
st.header("Violin Plot")
violin_image = Image.open('violin.png')
st.image(violin_image, caption='Violin Plot of Category Responses', use_column_width=True)

# Optional: Add interactivity with select boxes
plot_options = {
    "PCA Side by Side": pca_image,
    "Radar by Category": radar_category_image,
    "Radar by Category (Intensity)": radar_category_intensity_image,
    "Radar by Group/Cluster Side by Side": radar_image,
    "Radar by Group/Cluster Side by Side (Intensity)": radar_intensity_image,
    "Violin Plot": violin_image
}

selected_plots = st.multiselect("Choose plots to display", list(plot_options.keys()))

# Display selected plots
for plot_name in selected_plots:
    st.image(plot_options[plot_name], caption=plot_name, use_column_width=True)
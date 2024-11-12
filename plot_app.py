import streamlit as st
from PIL import Image
from pca_experiments_and_plots import FilterData
from skewed_synthetic_data_generator import Group, SkewedSyntheticData

# Title of the application
st.title("Data Analysis Plots")

num_responses = int(2**11)
num_categories = 5
raw_columns = ['Category_' + str(i) for i in range(1, num_categories + 1)]

# Initialize baseline data only once
if "data_loaded" not in st.session_state:
    groups = [Group(name='Group 1', occurrence_prob=0.3, preferences=[0, 1, 0, 0, 7]), 
              Group('Group 2', 0.2, [1, 2, 3, 2, -3]), 
              Group('Group 3', 0.3, [2, -1, -5, 2, 0]), 
              Group('Group 4', 0.2, [0, -1, 1, 0, 5])]
    synthetic_data_generator = SkewedSyntheticData(groups, num_categories, num_responses, credit_budget=60)
    st.session_state.data = FilterData(synthetic_data_generator.data, synthetic_data_generator.raw_columns)
    st.session_state.data_loaded = True

# Filter options
st.sidebar.header("Filter Options")
column_to_filter = st.sidebar.selectbox("Select column to filter", raw_columns)
quantity = st.sidebar.number_input("Quantity")
condition = st.sidebar.radio("Condition", ["greater", "less"])

# Apply filter button
if st.sidebar.button("Apply Filter"):
    st.session_state.data.filter_data(column_to_filter, quantity, condition)
    st.session_state.data.make_plots()

# Clear filter button
if st.sidebar.button("Clear Filter"):
    st.session_state.data.unfilter_data()
    st.session_state.data.make_plots()

# Display plots
st.header("Histograms")
hist = Image.open('vote_distributions.png')
st.image(hist, use_column_width=True)

st.header("PCA Plots")
pca_image = Image.open('pca.png')
st.image(pca_image, caption='PCA Side by Side', use_column_width=True)

st.header("PCA Plots")
pca_image = Image.open('pca.png')
st.image(pca_image, caption='PCA Side by Side', use_column_width=True)

st.header("Radar Plots")
radar_category_image = Image.open('category_radar.png')
st.image(radar_category_image, caption='Radar Plot by Category', use_column_width=True)

radar_category_intensity_image = Image.open('category_radar_intensity.png')
st.image(radar_category_intensity_image, caption='Radar Plot by Category Intensity', use_column_width=True)

radar_image = Image.open('radar.png')
st.image(radar_image, caption='Radar Plot Side by Side', use_column_width=True)

radar_intensity_image = Image.open('radar_intensity.png')
st.image(radar_intensity_image, caption='Radar Plot Side by Side with Intensity', use_column_width=True)

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

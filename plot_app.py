import streamlit as st
from PIL import Image
from pca_experiments_and_plots import FilterData
from skewed_synthetic_data_generator import Group, SkewedSyntheticData

# Title of the application
st.title("Data Analysis Plots")

num_responses = int(2**10)
num_categories = 5
# raw_columns = ['Category_' + str(i) for i in range(1, num_categories + 1)]

# Initialize baseline data only once
if "data_loaded" not in st.session_state:
    groups = [Group(name='Group 1', occurrence_prob=0.3, preferences=[2, 2, -2, -2, 7]), 
              Group('Group 2', 0.2, [-3, -3, 2, 2, 7]), 
              Group('Group 3', 0.3, [7, -1, -3, 2, 0]), 
              Group('Group 4', 0.2, [5, -1, 1, 0, -7])]
    synthetic_data_generator = SkewedSyntheticData(groups, num_categories, num_responses, credit_budget=80)
    st.session_state.data = FilterData(synthetic_data_generator.data, synthetic_data_generator.raw_columns)
    st.session_state.data.make_plots()
    st.session_state.data_loaded = True

if "filter_key" not in st.session_state:
    st.session_state.filter_key = 0

# Filter options
st.sidebar.header("Filter Options")
column_to_filter = st.sidebar.selectbox("Select column to filter", st.session_state.data.raw_columns, key=st.session_state.filter_key)
quantity = st.sidebar.number_input("Quantity")
condition = st.sidebar.radio("Condition", ["greater", "less"])

# Apply filter button
if st.sidebar.button("Apply Filter"):
    st.session_state.data.filter_data(column_to_filter, quantity, condition)
    st.session_state.data.reClusterData()
    st.session_state.data.make_plots()

# Clear filter button
if st.sidebar.button("Clear Filter"):
    st.session_state.data.unfilter_data()
    st.session_state.data.make_plots()

# Rename columns
st.sidebar.header("Rename columns")
column_to_replace = st.sidebar.text_input("Select column to replace")
new_column = st.sidebar.text_input("New name for column")

if st.sidebar.button("Replace column") and column_to_replace and new_column and column_to_replace in st.session_state.data.raw_columns:
    st.session_state.data.rename_column(column_to_replace, new_column)
    st.session_state.data.make_plots()

    # Update the widget keys to force re-rendering
    st.session_state.filter_key += 1

# To CSV
st.sidebar.header("Export to CSV")
csv_filename = st.sidebar.text_input("CSV filename (do NOT include .csv)")
if st.sidebar.button("Export") and csv_filename:
    st.session_state.data.to_csv(csv_filename + '.csv')

# Display plots
st.header("Histograms")
total_votes = Image.open('total_votes.png')
st.image(total_votes, use_column_width=True)

hist = Image.open('vote_distributions.png')
st.image(hist, use_column_width=True)

# st.header("PCA Plots")
# pca_image = Image.open('pca.png')
# st.image(pca_image, caption='PCA Side by Side', use_column_width=True)

st.header("Violin Plot")
violin_image = Image.open('violin.png')
st.image(violin_image, caption='Violin Plot of Category Responses', use_column_width=True)

data = st.session_state.data.data
st.title("Clustered Violin Plots")
for cluster in data['Cluster'].unique():
    plot_path = f'violin_cluster_{cluster}_with_stacked_size.png'
    st.subheader(f'Cluster {cluster}')
    image = Image.open(plot_path)
    st.image(image, caption=f'Violin Plot for Cluster {cluster}', use_column_width=True)

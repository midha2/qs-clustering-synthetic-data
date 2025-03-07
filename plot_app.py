import streamlit as st
from PIL import Image
from filter_data_and_plots import FilterData, cluster_wise_mean_comparison
from skewed_synthetic_data_generator import Group, SkewedSyntheticData
import pandas as pd
import csv

# Title of the application
st.title("Data Analysis Plots")

num_responses = int(2**11)
num_categories = 9

column_key_to_strings = {'party': {1: 'strong_rep', 2: 'ns_rep', 3: 'lean_rep', 4: 'undec/ind', 5: 'lean_dem', 6: 'ns_dem', 7: 'strong_dem'},
                         'age': {1: 'gen-z', 2: 'millenial', 3: 'gen-x', 4: 'boomer', 99: '<18'},
                         'education': {1: '<HS', 2: 'HS', 3: 'some_college', 4: '>=college'}, 
                         'race': {1: 'white', 2: 'black', 3: 'other', 4: 'hispanic', 5: 'multi'},
                         'sex': {1: 'male', 2: 'female'}}

# Initialize baseline data only once
if "data_loaded" not in st.session_state:
    st.session_state.filter_ranges = {}
    data = pd.read_csv('dataset_final-subset.tab', sep='\t', skip_blank_lines=True)
    raw_columns = ["QVgay", "QVgun","QVwall","QVpaidL","QVAA","QVgender","QVminW","QVabortion","QVdeficit","QVenviro"]
    data.dropna(how='any', inplace=True)
    
    for col in column_key_to_strings.keys():
        data[col] = data[col].round().astype(int) 

    pparty_dict = column_key_to_strings['party'].copy()
    for k, v in column_key_to_strings['party'].items():
        if 'rep' in v:
            pparty_dict[k] = 'rep'
        if 'dem' in v:
            pparty_dict[k] = 'dem'
    
    data['pparty'] = data['party'].map(pparty_dict)
    data.replace(column_key_to_strings, inplace=True)

    st.session_state.data = FilterData(data, raw_columns, list(column_key_to_strings.keys()) + ['pparty'])
    st.session_state.data_loaded = True

data = st.session_state.data.data
original = st.session_state.data.original_data

# Sidebar: Filter Options
st.sidebar.header("Filter Options")
column_to_filter = st.sidebar.selectbox("Select column to filter", st.session_state.data.raw_columns)

if column_to_filter not in st.session_state.filter_ranges:
    min_val, max_val = float(data[column_to_filter].min()), float(data[column_to_filter].max())
    st.session_state.filter_ranges[column_to_filter] = (min_val, max_val)

range_selected = st.sidebar.slider(
    "Select value range",
    min_value=float(original[column_to_filter].min()),
    max_value=float(original[column_to_filter].max()),
    value=st.session_state.filter_ranges[column_to_filter]
)

st.session_state.filter_ranges[column_to_filter] = range_selected

if st.sidebar.button("Apply Filter"):
    min_val, max_val = st.session_state.filter_ranges[column_to_filter]
    st.session_state.data.filter_data(column_to_filter, min_val, max_val)
    st.session_state.data.reClusterData()
    st.session_state.data.make_plots()

if st.sidebar.button("Clear Filter"):
    st.session_state.data.unfilter_data()
    st.session_state.filter_ranges[column_to_filter] = (original[column_to_filter].min(), original[column_to_filter].max())
    data = st.session_state.data.data
    st.session_state.data.make_plots()

st.sidebar.header("Cumulative Difference Threshold")
cumulative_diff_threshold = st.sidebar.slider("Select cumulative difference threshold", 0.0, 1.0, 0.9, 0.1)

if st.session_state.data.cumulative_diff_threshold != cumulative_diff_threshold:
    st.session_state.data.set_cumulative_diff_threshold(cumulative_diff_threshold)
    st.session_state.data.plots['mean_comp'], st.session_state.data.plots['opt_mean_comp'] = cluster_wise_mean_comparison(
        st.session_state.data.data, st.session_state.data.raw_columns, st.session_state.data.cumulative_diff_threshold
    )

# Export to CSV
st.sidebar.header("Export to CSV")
csv_filename = st.sidebar.text_input("CSV filename (do NOT include .csv)")
if st.sidebar.button("Export") and csv_filename:
    st.session_state.data.to_csv(csv_filename + '.csv')

# Tabs for visualization
tab1, tab2 = st.tabs(["Basic Plots", "All Plots"])

with tab1:
    st.header("Basic Plots")
    st.pyplot(st.session_state.data.plots['total_votes'])
    st.pyplot(st.session_state.data.plots['histogram'])

with tab2:
    st.header("All Plots")
    st.pyplot(st.session_state.data.plots['total_votes'])
    st.pyplot(st.session_state.data.plots['histogram'])
    st.pyplot(st.session_state.data.plots['violin'])
    
    st.title("Clustered Violin Plots")
    for cluster, fig in st.session_state.data.plots['clustered_violins'].items():
        st.subheader(f'Cluster {cluster}')
        st.pyplot(fig)

    st.title("Cluster Differences")
    st.pyplot(st.session_state.data.plots['mean_comp'])

    if st.session_state.data.dem_cols:
        st.header("Demographic Information")
        for fig in st.session_state.data.plots['dem'].values():
            st.pyplot(fig)

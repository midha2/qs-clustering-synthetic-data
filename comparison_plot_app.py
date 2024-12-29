import streamlit as st
from PIL import Image
from pca_experiments_and_plots import FilterData
from skewed_synthetic_data_generator import Group, SkewedSyntheticData
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the application
st.title("Data Analysis Plots")

num_responses = int(2**10)
num_categories = 5

column_key_to_strings = {'party': {1: 'strong_rep', 2: 'ns_rep', 3: 'lean_rep', 4: 'undec/ind', 5: 'lean_dem', 6: 'ns_dem', 7: 'strong_dem'},
                         'age': {1: 'gen-z', 2: 'millenial', 3: 'gen-x', 4: 'boomer', 99: '<18'},
                         'education': {1: '<HS', 2: 'HS', 3: 'some_college', 4: '>=college'}, 
                         'race': {1: 'white', 2: 'black', 3: 'other', 4: 'hispanic', 5: 'multi'},
                         'sex': {1: 'male', 2: 'female'}}

# Initialize baseline data only once
if "data_original" not in st.session_state:
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

    st.session_state.data_original = data
    st.session_state.raw_columns = raw_columns
    st.session_state.data_1 = FilterData(data.copy(), raw_columns, list(column_key_to_strings.keys()) + ['pparty'])
    st.session_state.data_2 = FilterData(data.copy(), raw_columns, list(column_key_to_strings.keys()) + ['pparty'])

# Filter options for Data 1
st.sidebar.header("Filter Options for Dataset 1")
column_to_filter_1 = st.sidebar.selectbox("Select column to filter (Dataset 1)", st.session_state.raw_columns, key="filter_1")
quantity_1 = st.sidebar.number_input("Quantity (Dataset 1)", key="quantity_1")
condition_1 = st.sidebar.radio("Condition (Dataset 1)", ["greater", "less"], key="condition_1")

if st.sidebar.button("Apply Filter (Dataset 1)"):
    st.session_state.data_1.filter_data(column_to_filter_1, quantity_1, condition_1)
    st.session_state.data_1.reClusterData()
    st.session_state.data_1.make_plots()

if st.sidebar.button("Clear Filter (Dataset 1)"):
    st.session_state.data_1.unfilter_data()
    st.session_state.data_1.make_plots()

# Filter options for Data 2
st.sidebar.header("Filter Options for Dataset 2")
column_to_filter_2 = st.sidebar.selectbox("Select column to filter (Dataset 2)", st.session_state.raw_columns, key="filter_2")
quantity_2 = st.sidebar.number_input("Quantity (Dataset 2)", key="quantity_2")
condition_2 = st.sidebar.radio("Condition (Dataset 2)", ["greater", "less"], key="condition_2")

if st.sidebar.button("Apply Filter (Dataset 2)"):
    st.session_state.data_2.filter_data(column_to_filter_2, quantity_2, condition_2)
    st.session_state.data_2.reClusterData()
    st.session_state.data_2.make_plots()

if st.sidebar.button("Clear Filter (Dataset 2)"):
    st.session_state.data_2.unfilter_data()
    st.session_state.data_2.make_plots()

# Display Plots for Dataset 1 and Dataset 2 Side by Side
st.header("Comparison of Plots (Dataset 1 vs. Dataset 2)")

# Side-by-side display for total votes bar charts
st.subheader("Total Votes")
col1, col2 = st.columns(2)
with col1:
    st.write("**Dataset 1**")
    st.pyplot(st.session_state.data_1.plots['total_votes'])
with col2:
    st.write("**Dataset 2**")
    st.pyplot(st.session_state.data_2.plots['total_votes'])

# Side-by-side display for histograms
st.subheader("Histograms")
col1, col2 = st.columns(2)
with col1:
    st.write("**Dataset 1**")
    st.pyplot(st.session_state.data_1.plots['histogram'])
with col2:
    st.write("**Dataset 2**")
    st.pyplot(st.session_state.data_2.plots['histogram'])

# Side-by-side display for violin plots
st.subheader("Violin Plots")
col1, col2 = st.columns(2)
with col1:
    st.write("**Dataset 1**")
    st.pyplot(st.session_state.data_1.plots['violin'])
with col2:
    st.write("**Dataset 2**")
    st.pyplot(st.session_state.data_2.plots['violin'])

# Side-by-side display for clustered violin plots
st.subheader("Clustered Violin Plots")
for cluster in st.session_state.data_1.plots['clustered_violins']:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset 1 - Cluster {cluster}**")
        st.pyplot(st.session_state.data_1.plots['clustered_violins'][cluster])
    with col2:
        st.write(f"**Dataset 2 - Cluster {cluster}**")
        if cluster in st.session_state.data_2.plots['clustered_violins']:
            st.pyplot(st.session_state.data_2.plots['clustered_violins'][cluster])
        else:
            st.write("No data for this cluster in Dataset 2.")

# Side-by-side display for demographic information (if available)
if st.session_state.data_1.dem_cols and st.session_state.data_2.dem_cols:
    st.subheader("Demographic Information")
    for dem_col in st.session_state.data_1.dem_cols:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Dataset 1 - {dem_col}**")
            st.pyplot(st.session_state.data_1.plots['dem'][dem_col])
        with col2:
            st.write(f"**Dataset 2 - {dem_col}**")
            if dem_col in st.session_state.data_2.plots['dem']:
                st.pyplot(st.session_state.data_2.plots['dem'][dem_col])
            else:
                st.write("No data for this demographic in Dataset 2.")

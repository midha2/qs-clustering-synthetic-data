import streamlit as st
from PIL import Image
from pca_experiments_and_plots import FilterData
from skewed_synthetic_data_generator import Group, SkewedSyntheticData
import pandas as pd
import csv

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
if "data_loaded" not in st.session_state:
    # groups = [Group(name='Group_1', occurrence_prob=0.3, preferences=[2, 2, -2, -2, 7]), 
    #           Group('Group_2', 0.2, [0, -3, 2, 2, 7]), 
    #           Group('Group_3', 0.3, [7, -1, -3, 2, 0]), 
    #           Group('Group_4', 0.2, [5, -1, 1, 0, -7])]
    # synthetic_data_generator = SkewedSyntheticData(groups, num_categories, num_responses, credit_budget=80)
    # data = synthetic_data_generator.data
    # raw_columns = synthetic_data_generator.raw_columns

    
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


    print(len(data))
    print(data)
    num_categories = len(raw_columns)
    # st.session_state.data = FilterData(data, raw_columns, ['Group'])
    st.session_state.data = FilterData(data, raw_columns, list(column_key_to_strings.keys()) + ['pparty'])
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

data = st.session_state.data
# Display plots
st.header("Histograms")
st.pyplot(data.plots['total_votes'])
st.pyplot(data.plots['histogram'])

st.header("Violin Plot")
st.pyplot(data.plots['violin'])

st.title("Clustered Violin Plots")
for cluster, fig in st.session_state.data.plots['clustered_violins'].items():
    st.subheader(f'Cluster {cluster}')
    st.pyplot(fig)

if st.session_state.data.dem_cols:
    st.header("Demographic Information")
    for fig in st.session_state.data.plots['dem'].values():
        st.pyplot(fig)

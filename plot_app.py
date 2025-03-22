import streamlit as st
from PIL import Image
from filter_data_and_plots import FilterData, cluster_wise_mean_comparison
from skewed_synthetic_data_generator import Group, SkewedSyntheticData
import pandas as pd
import csv

def show_plot_description(plot_id: str, description: str, label: str = None):
    desc_key = f"{plot_id}_desc_opened"
    log_key = f"{desc_key}_logged"

    if label is None:
        label = f"ℹ️ Show {plot_id.replace('_', ' ').title()} Description"

    show_desc = st.toggle(label, key=desc_key)

    if show_desc and st.session_state.get("clickstream_enabled", False):
        if not st.session_state.get(log_key, False):
            st.session_state.clickstream_log.append({
                "action": "view_plot_description",
                "plot": plot_id,
                "interface": st.session_state.get("active_tab", None),
                "timestamp": pd.Timestamp.now()
            })
            st.session_state[log_key] = True

    if show_desc:
        st.info(description)


# Title of the application
st.title("Data Analysis Plots")

num_responses = int(2**11)
num_categories = 9

column_key_to_strings = {'party': {1: 'strong_republican', 2: 'moderate_republican', 3: 'lean_republican', 4: 'undec/ind', 5: 'lean_democrat', 6: 'moderate_democrat', 7: 'strong_democrat'},
                         'age': {1: 'gen-z', 2: 'millenial', 3: 'gen-x', 4: 'boomer', 99: '<18'},
                         'education': {1: '<HS', 2: 'HS', 3: 'some_college', 4: '>=college'}, 
                         'race': {1: 'white', 2: 'black', 3: 'other', 4: 'hispanic', 5: 'multi'},
                         'sex': {1: 'male', 2: 'female'}}

plot_keys = {
    "histogram": "Histogram: A histogram represents the distribution of a dataset by grouping data into bins. The height of each bar shows how many data points fall within that range.",
    "bar_chart": "Bar Chart: A bar chart represents categorical data with rectangular bars. The length of each bar corresponds to sum of votes of a particular policy.",
    "violin": "Violin Plot: A violin plot combines aspects of a box plot and a density plot, showing the distribution of the data along with key statistical values like the median and interquartile range. You can see the number of votes corresponding to each number by the thickness of each violin.",
    "clustered_violin": "Clustered Violin Plot: This is a violin plot corresponding to one cluster (group) we have found in the data. A cluster may have similar response patterns in certain policies, allowing us to group them together. To the right, you can see the proportion of the population the cluster makes up."
}

# Initialize baseline data only once
if "data_loaded" not in st.session_state:
    st.session_state.filter_ranges = {}
    st.session_state.applied_filters = {}
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

if "clickstream_enabled" not in st.session_state:
    st.session_state.clickstream_enabled = False
if "clickstream_log" not in st.session_state:
    st.session_state.clickstream_log = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Interface 1"

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
    value=st.session_state.filter_ranges[column_to_filter],
    step=1.
)

st.session_state.filter_ranges[column_to_filter] = range_selected

if st.sidebar.button("Apply Filter"):
    min_val, max_val = st.session_state.filter_ranges[column_to_filter]
    st.session_state.applied_filters[column_to_filter] = (min_val, max_val)
    st.session_state.data.filter_data(column_to_filter, min_val, max_val)
    st.session_state.data.reClusterData()
    st.session_state.data.make_plots()
    if st.session_state.clickstream_enabled:
        st.session_state.clickstream_log.append({
            "action": "apply_filter",
            "column": column_to_filter,
            "range": range_selected,
            "interface": st.session_state.active_tab,
            "timestamp": pd.Timestamp.now()
        })

if st.sidebar.button("Clear All Filters"):
    st.session_state.data.unfilter_data()
    st.session_state.applied_filters.clear()
    for col in st.session_state.data.raw_columns:
        st.session_state.filter_ranges[col] = (original[col].min(), original[col].max())
    data = st.session_state.data.data
    st.session_state.data.make_plots()
    if st.session_state.clickstream_enabled:
        st.session_state.clickstream_log.append({
            "action": "clear_filters",
            "interface": st.session_state.active_tab,
            "timestamp": pd.Timestamp.now()
        })

st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Active Filters"):
    active_filters = st.session_state.applied_filters
    if len(active_filters) == 0:
        st.write("No active filters.")
    else:
        for col, (min_val, max_val) in active_filters.items():
            original_range = (
                original[col].min(),
                original[col].max()
            )
            # Only show if user has actually changed the range
            if (min_val, max_val) != original_range:
                st.markdown(f"**{col}**: {min_val:.1f} – {max_val:.1f}")
        if all((min_val, max_val) == (original[col].min(), original[col].max()) for col, (min_val, max_val) in active_filters.items()):
            st.write("No active filters.")

st.sidebar.header("Clickstream Tracking")

# Toggle switch
st.session_state.clickstream_enabled = st.sidebar.checkbox("Enable Clickstream Tracking", value=st.session_state.clickstream_enabled)

# Export
if st.sidebar.button("Export Clickstream") and st.session_state.clickstream_log:
    df_log = pd.DataFrame(st.session_state.clickstream_log)
    df_log.to_csv("clickstream_log.csv", index=False)
    st.sidebar.success("Clickstream data exported as 'clickstream_log.csv'")

# Tabs for visualization
tab1, tab2 = st.tabs(["Interface 1", "Interface 2"])

with tab1:
    st.session_state.active_tab = "Interface 1"
    st.header("Interface 1")
    show_plot_description(
        plot_id="i1bar_chart",
        description="A bar chart represents categorical data with rectangular bars. The length of each bar corresponds to sum of votes of a particular policy.",
        label="Show bar chart description.")
    st.pyplot(st.session_state.data.plots['total_votes'])

    show_plot_description(
        plot_id="i1histogram",
        description="A histogram represents the distribution of a dataset by grouping data into bins. The height of each bar shows how many data points fall within that range.",
        label="Show histogram description.")
    st.pyplot(st.session_state.data.plots['histogram'])

with tab2:
    st.session_state.active_tab = "Interface 2"
    st.header("Interface 2")

    show_plot_description(
        plot_id="i2bar_chart",
        description="A bar chart represents categorical data with rectangular bars. The length of each bar corresponds to sum of votes of a particular policy.",
        label="Show bar chart description.")
    st.pyplot(st.session_state.data.plots['total_votes'])

    show_plot_description(
        plot_id="i2histogram",
        description="A histogram represents the distribution of a dataset by grouping data into bins. The height of each bar shows how many data points fall within that range.",
        label="Show histogram description.")
    st.pyplot(st.session_state.data.plots['histogram'])

    show_plot_description(
        plot_id="violin_plot",
        description="A violin plot combines aspects of a box plot and a density plot, showing the distribution of the data along with key statistical values like the median and interquartile range. You can see the number of votes corresponding to each number by the thickness of each violin.")
    st.pyplot(st.session_state.data.plots['violin'])

    st.title("Clustered Violin Plots")
    show_plot_description(
            plot_id= "clustered_violin_plots",
            description="This is a violin plot corresponding to one cluster (group) we have found in the data. A cluster may have similar response patterns in certain policies, allowing us to group them together. To the right, you can see the proportion of the population the cluster makes up."
        )
    for cluster, fig in st.session_state.data.plots['clustered_violins'].items():
        st.pyplot(fig)

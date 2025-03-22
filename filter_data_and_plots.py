from math import pi
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from skewed_synthetic_data_generator import Group, SkewedSyntheticData
from scipy.stats import f_oneway
from matplotlib.patches import Patch

def total_votes(data, category_columns):
    # Calculate the sum of votes for each column
    vote_totals = data[category_columns].sum()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    vote_totals.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

    # Customize the chart
    ax.set_title('Total Votes by Category', fontsize=16)
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Total Votes', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig

def histogram(data, category_columns):
    num_categories = len(category_columns)
    ncols = 3  # Set number of columns (e.g., 3)
    nrows = (num_categories + ncols - 1) // ncols  # Calculate rows needed based on categories

    # Plot histograms for each category in data
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()

    # Find the overall y-axis and x-axis limits
    max_y = 0
    min_x, max_x = float('inf'), float('-inf')

    for i, col in enumerate(category_columns):
        # Calculate histogram without plotting to get y-axis limits
        counts, bins = np.histogram(data[col], bins=15)
        max_y = max(max_y, counts.max())
        min_x = min(min_x, bins.min())
        max_x = max(max_x, bins.max())

    # Plot histograms with consistent y and x limits
    for i, col in enumerate(category_columns):
        axes[i].hist(data[col], bins=15, color="steelblue", edgecolor="black")
        axes[i].set_title(f"{col} Vote Distribution")
        axes[i].set_xlabel("Votes")
        axes[i].set_ylabel("Frequency")
        axes[i].set_ylim(0, max_y)
        axes[i].set_xlim(min_x, max_x)

    # Hide any unused subplots
    for j in range(num_categories, len(axes)):
        axes[j].axis("off")

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig

def violin(data, categories):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=data[categories], ax=ax)

    # Customize the plot
    ax.set_title('Violin Plot of Category Responses')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Response Values')
    ax.grid(True)

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig

def clustered_violins_with_stacked_sizes(data: pd.DataFrame, categories, original_data_size, cluster_col = 'Cluster'):
    # Melt the data once for global min and max
    melted_data = data.melt(
        id_vars=[cluster_col], 
        value_vars=categories, 
        var_name='Category', 
        value_name='Value'
    )
    
    # Find global min and max for y-axis
    y_min, y_max = melted_data['Value'].min(), melted_data['Value'].max()

    filtered_data_size = len(data)  # Size of the filtered dataset
    filtered_fraction = filtered_data_size / original_data_size  # Fraction of filtered data size

    figures = dict()

    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]  # Filter data for the cluster
        cluster_size = len(cluster_data)  # Size of the current cluster
        cluster_fraction = cluster_size / original_data_size  # Fraction of original data size

        # Calculate cluster fractions for filtered data
        cluster_sizes_filtered = [
            len(data[data[cluster_col] == c]) / filtered_data_size
            for c in data[cluster_col].unique()
        ]
        cumulative_sizes = [sum(cluster_sizes_filtered[:i]) for i in range(len(cluster_sizes_filtered))]
        cluster_colors = ['blue' if c == cluster else 'gray' for c in data[cluster_col].unique()]

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Violin Plot
        melted_cluster_data = cluster_data.melt(
            id_vars=[cluster_col], 
            value_vars=categories, 
            var_name='Category', 
            value_name='Value'
        )
        sns.violinplot(x='Category', y='Value', data=melted_cluster_data, ax=axes[0], hue='Category', legend=False)
        axes[0].set_title(f'Violin Plot for Cluster {cluster}')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Value')
        axes[0].set_ylim(y_min, y_max)
        axes[0].grid(True)

        # Stacked Bar Plot
        for i, (size, color) in enumerate(zip(cluster_sizes_filtered, cluster_colors)):
            axes[1].bar(
                x=[f'Filtered Data Size: {filtered_data_size}'], 
                height=[size * filtered_fraction], 
                bottom=[cumulative_sizes[i] * filtered_fraction], 
                color=color, 
                edgecolor='black'
            )
        
        axes[1].set_title(f'Cluster Sizes (Filtered Data: {filtered_fraction * 100:.1f}%)')
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Fraction of Original Dataset')
        axes[1].set_xlabel('')

        # Annotate the current cluster size
        current_cluster_index = list(data[cluster_col].unique()).index(cluster)
        current_bottom = cumulative_sizes[current_cluster_index] * filtered_fraction
        axes[1].text(
            x=0, 
            y=current_bottom + cluster_sizes_filtered[current_cluster_index] * filtered_fraction / 2,
            s=f'{cluster_fraction * 100:.1f}%\nSize: {cluster_size}', 
            ha='center', va='center', color='white', weight='bold', fontsize=10
        )

        # Add a title with the original dataset size
        fig.suptitle(f'Original Dataset Size: {original_data_size}', fontsize=12, y=0.98)

        # Adjust layout and return the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the title
        figures[cluster] = fig

    return figures

def demographic_grouped_bar(data, demographic_columns, cluster_col = 'Cluster'):
    clusters = sorted(data[cluster_col].unique())  # Sort clusters for consistent order

    figures = dict()

    for col in demographic_columns:
        # Prepare data for grouped bars
        cluster_counts = {cluster: data[data[cluster_col] == cluster][col].value_counts() for cluster in clusters}
        unique_values = sorted(data[col].dropna().unique())  # Ensure consistent ordering of values

        # Parameters for bar spacing
        group_width = 0.8  # Total width for each group
        bar_width = group_width / len(clusters) * 0.8  # Slightly reduce individual bar width for spacing
        group_spacing = 1.5  # Distance between groups

        # Compute positions for groups and clusters
        x = [i * group_spacing for i in range(len(unique_values))]
        offsets = [-bar_width * (len(clusters) - 1) / 2 + i * bar_width for i in range(len(clusters))]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot bars for each cluster
        for offset, cluster in zip(offsets, clusters):
            counts = [cluster_counts[cluster].get(value, 0) for value in unique_values]
            bars = ax.bar(
                [pos + offset for pos in x],
                counts,
                width=bar_width,
                label=f'Cluster {cluster}',
                edgecolor='black'
            )
            
            # Add labels to each bar
            for bar, count in zip(bars, counts):
                if count > 0:  # Only label if count is non-zero
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{count}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

        # Customize the chart
        ax.set_title(f'Grouped Bar Chart for {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_values, rotation=45, fontsize=10)
        ax.legend(title='Clusters')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the chart
        figures[col] = fig
    
    return figures

def cluster_wise_mean_comparison_best(data, categories):
    clusters = sorted(data['Cluster'].unique())
    cluster_means = {cluster: data[data['Cluster'] == cluster][categories].mean() for cluster in clusters}

    # Create a DataFrame for easy manipulation
    mean_df = pd.DataFrame(cluster_means).T  # Transpose for better plotting
    mean_df.index = [f'Cluster {c}' for c in clusters]

    # Determine the most significant categories
        # Perform ANOVA for each category
    anova_p_values = {
        category: f_oneway(*(data[data['Cluster'] == cluster][category].dropna() for cluster in clusters)).pvalue
        for category in categories
    }
    sorted_p_values = sorted(anova_p_values.items(), key=lambda item: item[1])
    top_categories = [cat for cat, p in sorted_p_values if p <= 0.05]

    # Filter for the top categories
    mean_df = mean_df[top_categories]

    # Bar plot parameters
    num_clusters = len(clusters)
    num_categories = len(top_categories)
    bar_width = 0.8 / num_clusters  # Adjust bar width for multiple clusters
    x = np.arange(num_categories)  # X positions for categories

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for each cluster
    for i, cluster in enumerate(clusters):
        ax.bar(
            x + i * bar_width, 
            mean_df.loc[f'Cluster {cluster}'], 
            width=bar_width, 
            label=f'Cluster {cluster}',
            edgecolor='black'
        )

    # Customize the plot
    ax.set_title('Cluster-Wise Mean Comparison by Category', fontsize=16)
    ax.set_xlabel('Categories', fontsize=14)
    ax.set_ylabel('Mean Value', fontsize=14)
    ax.set_xticks(x + (num_clusters - 1) * bar_width / 2)
    ax.set_xticklabels(top_categories, rotation=45, fontsize=12)
    ax.legend(title='Clusters')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig

def cluster_wise_mean_comparison(data, categories, cum_dif = 0.9):
    clusters = sorted(data['Cluster'].unique())
    cluster_means = {cluster: data[data['Cluster'] == cluster][categories].mean() for cluster in clusters}

    # Create a DataFrame for easy manipulation
    mean_df = pd.DataFrame(cluster_means).T  # Transpose for better plotting
    mean_df.index = [f'Cluster {c}' for c in clusters]

    # Calculate pairwise square root mean differences
    pairwise_differences = {}
    for cat in categories:
        differences = []
        for c1 in clusters:
            for c2 in clusters:
                if c1 < c2:
                    diff = (mean_df.loc[f'Cluster {c1}', cat] - mean_df.loc[f'Cluster {c2}', cat]) ** 2
                    differences.append(diff)
        pairwise_differences[cat] = np.sqrt(np.mean(differences))

    # Determine the most significant categories
        # Sort categories by pairwise differences
    sorted_differences = sorted(pairwise_differences.items(), key=lambda item: item[1], reverse=True)
        # Automatically pick top_n based on cumulative contribution to total difference
    total_difference = sum(diff for _, diff in sorted_differences)
    cumulative_difference = 0
    top_categories = []
    for cat, diff in sorted_differences:
        cumulative_difference += diff
        top_categories.append(cat)
        if cumulative_difference / total_difference >= cum_dif:
            break

        # Bar plot parameters
    num_clusters = len(clusters)
    num_categories = len(categories)
    bar_width = 0.8 / num_clusters  # Adjust bar width for multiple clusters
    x = np.arange(num_categories)  # X positions for categories

    # Assign colors to clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    alpha = 0.05
    gray_shades = np.array([(*color[:3], alpha) for color in cluster_colors])

    # Create the first plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for i, cluster in enumerate(clusters):
        ax1.bar(
            x + i * bar_width,
            [mean_df.loc[f'Cluster {cluster}', cat] for cat in categories],
            width=bar_width,
            label=f'Cluster {cluster}',
            edgecolor=['black' if cat in top_categories else 'gray' for cat in categories],  # Gray edges for non-top categories
            color=[cluster_colors[i] if cat in top_categories else gray_shades[i] for cat in categories]
        )

    ax1.legend(title="Clusters")
    
    ax1.set_title('Cluster-Wise Mean Comparison by Category', fontsize=16)
    ax1.set_xlabel('Categories', fontsize=14)
    ax1.set_ylabel('Mean Value', fontsize=14)
    ax1.set_xticks(x + (num_clusters - 1) * bar_width / 2)
    ax1.set_xticklabels(categories, rotation=45, fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Create the second plot (only top categories)
    num_top_categories = len(top_categories)
    x_top = np.arange(num_top_categories)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for i, cluster in enumerate(clusters):
        ax2.bar(
            x_top + i * bar_width, 
            [mean_df.loc[f'Cluster {cluster}', cat] for cat in top_categories], 
            width=bar_width, 
            label=f'Cluster {cluster}',
            edgecolor='black'
        )
    ax2.set_title('Cluster-Wise Mean Comparison for Top Categories', fontsize=16)
    ax2.set_xlabel('Top Categories', fontsize=14)
    ax2.set_ylabel('Mean Value', fontsize=14)
    ax2.set_xticks(x_top + (num_clusters - 1) * bar_width / 2)
    ax2.set_xticklabels(top_categories, rotation=45, fontsize=12)
    ax2.legend(title='Clusters')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and return the figures
    plt.tight_layout()
    return fig1, fig2

class FilterData:
    def __init__(self, data, raw_columns, dem_columns=None, cluster_col='Cluster'):
        self.data = data.copy()
        self.raw_columns = raw_columns.copy()
        self.num_categories = len(raw_columns)
        if cluster_col not in data.columns:
            self.reClusterData()
        self.original_data = data.copy()
        self.dem_cols = dem_columns
        self.plots = dict()
        self.cumulative_diff_threshold = 0.9
        self.make_plots()
    
    def set_cumulative_diff_threshold(self, threshold):
        assert threshold <= 1.0 and threshold >= 0., "it must be the case that 0 < threshold < 1"
        self.cumulative_diff_threshold = threshold

    def rename_column(self, col_to_rename, new_name):
        assert col_to_rename in self.data.columns, "Cannot rename column that does not exist!"
        self.data.rename(columns={col_to_rename: new_name}, inplace=True)
        self.original_data.rename(columns={col_to_rename: new_name}, inplace=True)
        if col_to_rename in self.raw_columns:
            idx_to_rep = self.raw_columns.index(col_to_rename)
            self.raw_columns[idx_to_rep] = new_name
    
    def filter_data(self, col, quantity, condition="greater"):
        if condition == "greater":
            self.data = self.data[self.data[col] > quantity]
        elif condition == "less":
            self.data = self.data[self.data[col] < quantity]
        else:
            raise ValueError("Condition must be 'greater' or 'less'.")

        return self.data
    
    def filter_data(self, col, minVal, maxVal):
        self.data = self.data[self.data[col] <= maxVal]
        self.data = self.data[self.data[col] >= minVal]

        return self.data
    
    def __findOptimalNumberOfClusters(self, min_number_of_clusters, max_number_of_clusters, cluster_col='Cluster'):
        data = self.data[self.raw_columns]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        k_start = min_number_of_clusters
        k_max = max_number_of_clusters
        sil = np.zeros(1 + (k_max - k_start))
        # Try a bunch of clusters to see which is the optimal number
        for i in range(k_start, k_max+1):
            kmeans = KMeans(n_clusters = i, n_init=10).fit(data_scaled)
            labels = kmeans.labels_
            if cluster_col in data.columns:
                data.loc[:, cluster_col] = labels
            else:
                data[cluster_col] = labels
            
            sil[i - k_start] = silhouette_score(data_scaled, labels, metric = 'euclidean')
        return np.argmax(sil) + k_start
    
    def reClusterData(self, cluster_col='Cluster'):
        data = self.data[self.raw_columns]
        optimal_number_of_clusters = self.__findOptimalNumberOfClusters(min_number_of_clusters=2, 
                                                                max_number_of_clusters=10)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        kmeans_model = KMeans(n_clusters=optimal_number_of_clusters, n_init=10)
        clusters = kmeans_model.fit_predict(data_scaled)
        if cluster_col in self.data.columns:
            self.data.loc[:, cluster_col] = clusters
        else:
            self.data[cluster_col] = clusters
        return data

    def unfilter_data(self):
        # Restore the original data and responses
        self.data = self.original_data.copy()
    
    def make_plots(self):
        if 'Cluster' not in self.data.columns:
            self.reClusterData()
        plt.close('all')
        self.plots['histogram'] = histogram(self.data, self.raw_columns)
        self.plots['violin'] = violin(self.data, self.raw_columns)
        self.plots['clustered_violins'] = clustered_violins_with_stacked_sizes(self.data, self.raw_columns, self.original_data.shape[0])
        self.plots['total_votes'] = total_votes(self.data, self.raw_columns)
        # if self.dem_cols:
        #     self.plots['dem'] = demographic_grouped_bar(self.data, self.dem_cols)
        # self.plots['mean_comp'], self.plots['opt_mean_comp'] = cluster_wise_mean_comparison(self.data, self.raw_columns, self.cumulative_diff_threshold)

    def to_csv(self, name='data.csv'):
        self.data.to_csv(name)
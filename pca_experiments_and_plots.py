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

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig("vote_distributions.png")

def run_pca(df):
    print('Running pca...')
    X = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    return pd.DataFrame(data = components, columns = ['pc1', 'pc2'])

def plot_pca_side_by_side(data, color_by1, color_by2, title1, title2):
    print('Generating plots...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x='pc1', y='pc2', hue=color_by1, data=data, palette='Set1', s=40, alpha=0.7, ax=axes[0])
    axes[0].set_title(f'PCA Plot colored by {title1}')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(loc='best', title=title1)
    axes[0].grid(True)

    sns.scatterplot(x='pc1', y='pc2', hue=color_by2, data=data, palette='Set2', s=40, alpha=0.7, ax=axes[1])
    axes[1].set_title(f'PCA Plot colored by {title2}')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(loc='best', title=title2)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('pca.png')

def _create_radar_subplot(ax, preferences, categories, title):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    preferences += preferences[:1]

    # Create radar plot
    ax.plot(angles, preferences, linewidth=2, linestyle='solid')
    ax.fill(angles, preferences, 'b', alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title(title)

def _plot_radar_axis(df, num_categories, col, ax, axes, categories):
    group_preferences = {}
    for group in df[col].unique():
        group_data = df[df[col] == group]
        group_avg_prefs = group_data.loc[:, df.columns[:num_categories]].mean().tolist()
        group_preferences[f'{col} {group}'] = group_avg_prefs
    
    for i, (group, preferences) in enumerate(group_preferences.items()):
        _create_radar_subplot(axes[ax, i], preferences, categories, group)

def plot_side_by_side_radar(df, num_categories, col_1, col_2, title):
    categories = [f'Cat{i+1}' for i in range(num_categories)]
    
    num_groups = df[col_1].nunique()
    num_clusters = df[col_2].nunique()

    # Adjust subplot layout for single group/cluster cases
    if num_groups == 1 and num_clusters == 1:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
        axes = np.array([[axes[0]], [axes[1]]])
    elif num_groups == 1:
        fig, axes = plt.subplots(2, num_clusters, figsize=(16, 8), subplot_kw=dict(polar=True))
        axes[0] = np.array([axes[0]])  # Ensure it’s a 2D array
    elif num_clusters == 1:
        fig, axes = plt.subplots(2, num_groups, figsize=(16, 8), subplot_kw=dict(polar=True))
        axes[1] = np.array([axes[1]])  # Ensure it’s a 2D array
    else:
        fig, axes = plt.subplots(2, max(num_groups, num_clusters), figsize=(16, 8), subplot_kw=dict(polar=True))

    # Plot radars for groups and clusters
    _plot_radar_axis(df, num_categories, col_1, 0, axes, categories)
    _plot_radar_axis(df, num_categories, col_2, 1, axes, categories)

    # Hide unused subplots
    if num_groups > num_clusters:
        for j in range(num_clusters, num_groups):
            fig.delaxes(axes[1, j])
    elif num_clusters > num_groups:
        for j in range(num_groups, num_clusters):
            fig.delaxes(axes[0, j])

    plt.tight_layout()
    plt.savefig(f'{title}.png')

def _plot_radar_axis_for_categories(df, num_categories, categories, col, ax, axes):
    category_preferences = {}
    # Iterate through each category to get group preferences
    for i, category in enumerate(categories):
        group_preferences = {}
        for group in df[col].unique():
            group_data = df[df[col] == group]
            group_avg_prefs = group_data[category].mean()  # Average preference for the group in the current category
            group_preferences[group] = group_avg_prefs
        
        category_preferences[category] = group_preferences
    
    # Plot each category as a separate radar plot
    for i, (category, group_prefs) in enumerate(category_preferences.items()):
        preferences = list(group_prefs.values())
        _create_radar_subplot(axes[i], preferences, list(group_prefs.keys()), category)

def plot_side_by_side_radar_by_category(df, num_categories, col, title):
    categories = [f'Category_{i+1}' for i in range(num_categories)]
    
    fig, axes = plt.subplots(1, num_categories, figsize=(16, 8), subplot_kw=dict(polar=True))

    _plot_radar_axis_for_categories(df, num_categories, categories, col, 0, axes)

    plt.tight_layout()
    plt.savefig(f'{title}.png')

def violin(data, categories):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data[categories])
    plt.title('Violin Plot of Category Responses')
    plt.xlabel('Categories')
    plt.ylabel('Response Values')
    plt.grid(True)
    plt.savefig('violin.png')

def clustered_violins_with_sizes(data, categories, original_data_size):
    # Melt the data once for global min and max
    melted_data = data.melt(
        id_vars=['Cluster'], 
        value_vars=categories, 
        var_name='Category', 
        value_name='Value'
    )
    
    # Find global min and max for y-axis
    y_min, y_max = melted_data['Value'].min(), melted_data['Value'].max()

    filtered_data_size = len(data)  # Size of the filtered dataset

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]  # Filter data for the cluster
        cluster_size = len(cluster_data)  # Size of the cluster
        cluster_fraction = cluster_size / original_data_size  # Fraction of original data size
        filtered_fraction = filtered_data_size / original_data_size  # Fraction of filtered data size

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Violin Plot
        melted_cluster_data = cluster_data.melt(
            id_vars=['Cluster'], 
            value_vars=categories, 
            var_name='Category', 
            value_name='Value'
        )
        sns.violinplot(x='Category', y='Value', data=melted_cluster_data, palette='Set2', ax=axes[0])
        axes[0].set_title(f'Violin Plot for Cluster {cluster}')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Value')
        axes[0].set_ylim(y_min, y_max)
        axes[0].grid(True)

        # Bar Plot for Cluster and Filtered Data Size
        bars = axes[1].bar(['Filtered Data', 'Cluster'], [filtered_fraction, cluster_fraction], color=['gray', 'blue'])
        axes[1].set_title(f'Cluster {cluster} Size')
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Fraction of Original Dataset')
        axes[1].set_xlabel('')

        # Add text annotations for percentages
        axes[1].bar_label(
            bars, 
            labels=[f'{filtered_fraction:.2%}', f''],
                    # f'{cluster_fraction:.2%}'], 
            label_type='edge', 
            fontsize=10
        )

        # Add size annotations below the bars
        axes[1].text(
            x=0, y=-0.05, 
            s=f'Filtered Size: {filtered_data_size}', 
            fontsize=10, ha='center', va='top'
        )
        axes[1].text(
            x=1, y=-0.05, 
            s=f'Cluster Size: {cluster_size}', 
            fontsize=10, ha='center', va='top'
        )

        # Add a title with the original dataset size
        fig.suptitle(f'Original Dataset Size: {original_data_size}', fontsize=12, y=0.98)

        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the title
        plt.savefig(f'violin_cluster_{cluster}_with_size.png')
        plt.close()





class FilterData:
    def __init__(self, data, raw_columns):
        self.data = data.copy()
        self.original_data = data.copy()
        self.raw_columns = raw_columns.copy()
        self.num_categories = len(raw_columns)
    
    def filter_data(self, col, quantity, condition="greater"):
        if condition == "greater":
            self.data = self.data[self.data[col] > quantity]
        elif condition == "less":
            self.data = self.data[self.data[col] < quantity]
        else:
            raise ValueError("Condition must be 'greater' or 'less'.")

        return self.data
    
    def __findOptimalNumberOfClusters(self, min_number_of_clusters, max_number_of_clusters):
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
            data.loc[:, 'Cluster'] = labels
            sil[i - k_start] = silhouette_score(data_scaled, labels, metric = 'euclidean')
        return np.argmax(sil) + k_start
    
    def reClusterData(self):
        data = self.data[self.raw_columns]
        optimal_number_of_clusters = self.__findOptimalNumberOfClusters(min_number_of_clusters=2, 
                                                                max_number_of_clusters=10)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        kmeans_model = KMeans(n_clusters=optimal_number_of_clusters, n_init=10)
        clusters = kmeans_model.fit_predict(data_scaled)
        self.data.loc[:, 'Cluster'] = clusters
        return data

    def unfilter_data(self):
        # Restore the original data and responses
        self.data = self.original_data.copy()
    
    def make_plots(self):
        violin(self.data, self.raw_columns)
        pca_components = run_pca(self.data.loc[:, self.raw_columns].values)
        finalDf = pd.concat([pca_components, self.data[['Group']]], axis = 1)
        finalDf = pd.concat([finalDf, self.data[['Cluster']]], axis = 1)
        plot_side_by_side_radar_by_category(self.data, self.num_categories, 'Cluster', 'category_radar')
        plot_side_by_side_radar(self.data, self.num_categories, 'Group', 'Cluster', 'radar')
        plot_pca_side_by_side(finalDf, 'Group', 'Cluster', 'Colored by Group', 'Colored by Cluster')

        intensity_data = self.data.copy()
        intensity_data[self.raw_columns] = self.data[self.raw_columns].abs()

        plot_side_by_side_radar_by_category(intensity_data, self.num_categories, 'Cluster', 'category_radar_intensity')
        plot_side_by_side_radar(intensity_data, self.num_categories, 'Group', 'Cluster', 'radar_intensity')
        histogram(self.data, self.raw_columns)
        clustered_violins_with_sizes(self.data, self.raw_columns, self.original_data.shape[0])
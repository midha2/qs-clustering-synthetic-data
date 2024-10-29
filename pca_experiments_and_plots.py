from math import pi
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from skewed_synthetic_data_generator import Group, SkewedSyntheticData

def histogram(responses, num_categories):
    ncols = 3  # Set number of columns (e.g., 3)
    nrows = (num_categories + ncols - 1) // ncols  # Calculate rows needed based on categories

    # Plot histograms for each category
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()

    for i in range(num_categories):
        axes[i].hist(responses[i], bins=15, color="steelblue", edgecolor="black")
        axes[i].set_title(f"Category {i+1} Vote Distribution")
        axes[i].set_xlabel("Votes")
        axes[i].set_ylabel("Frequency")

    # Hide the last subplot if not used
    if num_categories < len(axes):
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

    fig, axes = plt.subplots(2, max(num_groups, num_clusters), figsize=(16, 8), subplot_kw=dict(polar=True))

    _plot_radar_axis(df, num_categories, col_1, 0, axes, categories)
    _plot_radar_axis(df, num_categories, col_2, 1, axes, categories)

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

groups = [Group(name='Group 1', occurrence_prob = 0.4, preferences = [0, 1, 0, 0, 0]), 
          Group('Group 2', 0.2, [1, 2, 3, 2, -3]), 
          Group('Group 3', 0.37, [2, -1, -5, 2, 0]), 
          Group('Group 4', 0.03, [-2, -1, 1, 0, 5])]
num_responses = int(2**11)
num_categories = 5
synthetic_data_generator = SkewedSyntheticData(groups, num_categories, num_responses, credit_budget=60, scaling=True)

pca_components = run_pca(synthetic_data_generator.data.loc[:, synthetic_data_generator.raw_columns].values)
finalDf = pd.concat([pca_components, synthetic_data_generator.data[['Group']]], axis = 1)
finalDf = pd.concat([finalDf, synthetic_data_generator.data[['Cluster']]], axis = 1)

plot_side_by_side_radar_by_category(synthetic_data_generator.data, num_categories, 'Group', 'category_radar')
plot_side_by_side_radar(synthetic_data_generator.data, num_categories, 'Group', 'Cluster', 'radar')
plot_pca_side_by_side(finalDf, 'Group', 'Cluster', 'Colored by Group', 'Colored by Cluster')

intensity_data = synthetic_data_generator.data.copy()
intensity_data[synthetic_data_generator.raw_columns] = synthetic_data_generator.data[synthetic_data_generator.raw_columns].abs()

plot_side_by_side_radar_by_category(intensity_data, num_categories, 'Group', 'category_radar_intensity')
plot_side_by_side_radar(intensity_data, num_categories, 'Group', 'Cluster', 'radar_intensity')
violin(synthetic_data_generator.data, synthetic_data_generator.raw_columns)
histogram(synthetic_data_generator.responses, num_categories)
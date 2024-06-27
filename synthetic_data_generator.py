import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Group:
  def __init__(self, name: str, occurrence_prob: float, preferences: list[int]):
    self.name = name
    self.prob = occurrence_prob
    self.preferences = preferences

def verifyGroups(groups, credit_budget, num_categories):
    for g in groups: 
        assert len(g.preferences) == num_categories, f"Groups must have {num_categories} preferences each"
        assert np.sum(np.array(g.preferences)**2) < credit_budget, "Group preferences must have quadratic sum less than number of credit_budget"
    assert abs(sum([g.prob for g in groups]) - 1.) < 0.01, "Group probabilities must sum to 1"

def mapProbabilitiesToGroups(groups):
    prob_to_group = dict()
    p = 0.
    for group in groups:
        prob_to_group[p + group.prob] = group
        p += group.prob
    return prob_to_group

# Generates synthetic data using groups, num_responses, num_categories, credit_budget
# responses: numpy array where responses[j, i] is the response of person i of category j
# people: list where people[i] is the Group of person i
def generateResponses(groups: list[Group], num_responses: int, num_categories: int, credit_budget: int):
    verifyGroups(groups, credit_budget, num_categories)
    prob_to_group = mapProbabilitiesToGroups(groups)

    responses = np.zeros((num_categories, num_responses))
    people = ['' for i in range(num_responses)]
    
    probs = np.array(list(prob_to_group.keys()))
    for i in range(num_responses):
        group_p = rd.random()
        group_of_person_i = prob_to_group[probs[np.argmax(probs > group_p)]]
        for j in range(num_categories):
            responses[j, i] = rd.normalvariate(group_of_person_i.preferences[j])
        people[i] = group_of_person_i.name
        
        credits_used = np.sum(responses[:,i]**2)
        max_preturb = 10
        while credits_used > credit_budget:
            # Reduce intensity of preferences to ensure user is within budget
            signs = np.sign(responses[:,i])
            intensities_indices = np.argsort(responses[:,i]**2)
            for j in range(1, num_categories + 1):
                for _ in range(max_preturb):
                    responses[intensities_indices[-j], i] -= signs[intensities_indices[-j]] * 0.1
                    credits_used = np.sum(responses[:,i]**2)
                    if credits_used <= credit_budget:
                        break
                if credits_used <= credit_budget:
                        break
    return responses, people

def findOptimalNumberOfClusters(data, min_number_of_clusters, max_number_of_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    k_start = min_number_of_clusters
    k_max = max_number_of_clusters
    sil = np.zeros(1 + (k_max - k_start))
    # Try a bunch of clusters to see which is the optimal number
    for i in range(k_start, k_max+1):
        kmeans = KMeans(n_clusters = i).fit(data_scaled)
        labels = kmeans.labels_
        data['Cluster'] = labels
        sil[i - k_start] = silhouette_score(data_scaled, labels, metric = 'euclidean')
    return np.argmax(sil) + k_start

def optimallyClusterData(data):
    optimal_number_of_clusters = findOptimalNumberOfClusters(data, min_number_of_clusters=2, 
                                                             max_number_of_clusters=10)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans_model = KMeans(n_clusters=optimal_number_of_clusters)
    clusters = kmeans_model.fit_predict(data_scaled)
    data['Cluster'] = clusters
    cluster_list = list([data[data['Cluster'] == i] for i in range(optimal_number_of_clusters)])
    return data, cluster_list


groups = [Group(name='White', occurrence_prob = 0.59, preferences = [-4, 1, 0, -5]), 
          Group('Black', 0.14, [3, 2, 1, 5]), 
          Group('Hispanic', 0.2, [3, -5, -2, 0]), 
          Group('Asian', 0.07, [-2, -1, 6, 0])]
credit_budget = 50
num_responses = int(1e4)
num_categories = 4

# responses: numpy array where responses[j, i] is the response of person i of category j
# people: list where people[i] is the Group of person i
# data: pandas dataframe of responses, columns are 'Category_X' (one-indexed) and 'Group'

responses, people = generateResponses(groups=groups, num_categories=num_categories, 
                                      num_responses=num_responses, credit_budget=credit_budget)
data = pd.DataFrame(responses.T, columns=['Category_' + str(i) for i in range(1, num_categories + 1)])
data['Group'] = people
data.to_csv('exp1_data/synthetic_data.csv') # export generated data to csv

# CLUSTERING STUFF BELOW
data, cluster_list = optimallyClusterData(data.iloc[:, 0:num_categories])
data['Group'] = people

# ARI and NMI measure similarity between clusters and original groups of people
ari = adjusted_rand_score(data['Group'], data['Cluster'])
nmi = normalized_mutual_info_score(data['Group'], data['Cluster'])

# Prints which clusters different groups are placed in
for group in groups:
    g = data[data['Group'] == group.name]
    print(g['Cluster'].value_counts(), group.name)
print(f'Normalized Mutual Information: {nmi}')
print(f'Adjusted Rand Index: {ari}')

sums = [cluster.iloc[:, 0:num_categories].sum(axis=0) for cluster in cluster_list]
# Prints sum of votes in clusters
print(sums)
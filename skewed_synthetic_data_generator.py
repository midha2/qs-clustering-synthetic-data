import numpy as np
import pandas as pd
import random as rd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Group:
    def __init__(self, name: str, occurrence_prob: float, preferences: list[int], sigma=0.5):
        self.name = name
        self.prob = occurrence_prob
        self.preferences = preferences
        self.sigma = sigma


# Generates synthetic data using groups, num_responses, num_categories, credit_budget
#    (if credit budget not specified, defaults to 4 * num_categories^1.5)
# responses: numpy array where responses[j, i] is the response of person i of category j
# original_groups: list where people[i] is the Group of person i
# data: Dataframe indexed by 'Category_i' for i in range(1, num_categories) inclusive, 
#   'Cluster' gives kmeans cluster, 'Group' gives original group
# data_columns : List of columns in data
# cluster_list: List where cluster_list[k] corresponds to the entries in data in the kth cluster
class SkewedSyntheticData:
    def __verifyGroups(self):
        for g in self.groups: 
            assert len(g.preferences) == self.num_categories, f"Groups must have {self.num_categories} preferences each"
            assert np.sum(np.array(g.preferences)**2) < self.credit_budget, "Group preferences must have quadratic sum less than number of credit_budget"
        assert abs(sum([g.prob for g in self.groups]) - 1.) < 0.01, "Group probabilities must sum to 1"
    
    def __mapProbabilitiesToGroups(self):
        prob_to_group = dict()
        p = 0.
        for group in self.scaled_groups:
            prob_to_group[p + group.prob] = group
            p += group.prob
        return prob_to_group
    
    def __generateResponses(self):
        print('Generating Data...')
        credit_budget = self.credit_budget
        num_categories = self.num_categories
        num_responses = self.num_responses
        self.__verifyGroups()
        prob_to_group = self.__mapProbabilitiesToGroups()

        responses = np.zeros((num_categories, num_responses))
        people = ['' for i in range(num_responses)]
        
        probs = np.array(list(prob_to_group.keys()))
        for i in range(num_responses):
            group_p = rd.random()
            group_of_person_i = prob_to_group[probs[np.argmax(probs > group_p)]]
            for j in range(num_categories):
                mean = np.log(abs(group_of_person_i.preferences[j]) + 1)  # Shift to avoid log(0) issues
            
                if group_of_person_i.preferences[j] >= 0:
                    responses[j, i] = np.random.lognormal(mean, sigma=group_of_person_i.sigma)
                else:
                    responses[j, i] = -np.random.lognormal(mean, sigma=group_of_person_i.sigma)
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

    def __findOptimalNumberOfClusters(self, min_number_of_clusters, max_number_of_clusters):
        data = self.data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        k_start = min_number_of_clusters
        k_max = max_number_of_clusters
        sil = np.zeros(1 + (k_max - k_start))
        # Try a bunch of clusters to see which is the optimal number
        for i in range(k_start, k_max+1):
            kmeans = KMeans(n_clusters = i, n_init=10).fit(data_scaled)
            labels = kmeans.labels_
            data['Cluster'] = labels
            sil[i - k_start] = silhouette_score(data_scaled, labels, metric = 'euclidean')
        return np.argmax(sil) + k_start
    
    def __optimallyClusterData(self):
        print('Running kMeans...')
        data = self.data
        optimal_number_of_clusters = self.__findOptimalNumberOfClusters(min_number_of_clusters=2, 
                                                                max_number_of_clusters=10)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        kmeans_model = KMeans(n_clusters=optimal_number_of_clusters, n_init=10)
        clusters = kmeans_model.fit_predict(data_scaled)
        data['Cluster'] = clusters
        cluster_list = list([data[data['Cluster'] == i] for i in range(optimal_number_of_clusters)])
        return data, cluster_list
    
    def RegenerateData(self):
        self.responses, self.original_groups = self.__generateResponses()
        self.data = pd.DataFrame(self.responses.T, columns=['Category_' + str(i) for i in range(1, self.num_categories + 1)])
        self.data_columns = ['Category_' + str(i) for i in range(1, self.num_categories + 1)] + ['Cluster', 'Group']
        self.data, self.cluster_list = self.__optimallyClusterData()
        self.data['Group'] = self.original_groups

    def _scaleGroups(self):
        print('Scaling Groups...')
        self.scaled_groups = []
        for group in groups:
            np_preferences = np.array(group.preferences)
            assert not np.all(np_preferences == 0), f"Can't scale a group with all 0 preferences!"
            target_budget = self.credit_budget - group.sigma
            current_spend = np.sum(np_preferences**2)
            scale_factor = target_budget / current_spend
            new_preferences = list(np_preferences * np.sqrt(scale_factor))
            self.scaled_groups.append(Group(group.name, group.prob, 
                                            new_preferences, 
                                            group.sigma))
    
    def __init__(self, groups, num_categories, num_responses, credit_budget=-1, scaling=False, seed=None):
        self.groups = groups
        self.scaled_groups = groups
        self.credit_budget = 4 * (num_categories**1.5) if credit_budget == -1 else credit_budget
        if scaling: self._scaleGroups()
        if seed: 
            np.random.seed(seed)
            rd.seed(seed)
        self.num_responses = num_responses
        self.num_categories = num_categories
        self.RegenerateData()

    def GenerateUsingNewDataset(self, new_groups, new_num_categories, new_num_responses, new_credit_budget=-1, scaling=False, seed=None):
        self.__init__(new_groups, new_num_categories, new_num_responses, new_credit_budget, scaling, seed)

# Initialize synthetic data generator using groups, responses, categories
groups = [Group(name='White', occurrence_prob = 0.59, preferences = [-2, 1, 0, 0]), 
          Group('Black', 0.14, [1, 2, 1, 2]), 
          Group('Hispanic', 0.2, [2, -2, -7, 0]), 
          Group('Asian', 0.07, [-2, -1, 1, 0])]
num_responses = int(1e3)
num_categories = 4
synthetic_data_generator = SkewedSyntheticData(groups, num_categories, num_responses, credit_budget=1000, scaling=False)

synthetic_data_generator.data.to_csv('skewed_synthetic_data.csv') # export generated data to csv

# Regenerate data using same group configuration
synthetic_data_generator.RegenerateData()

# Regenerate data using new group configuration
new_groups = [Group(name='White', occurrence_prob = 0.59, preferences = [-5, 1, 0, 0], sigma=0.2),
          Group('Black', 0.14, [1, 6, 1, 2], sigma=0.2),
          Group('Hispanic', 0.2, [2, -2, -7, 0], sigma=0.2), 
          Group('Asian', 0.07, [-2, -1, 1, 4], sigma=0.2)]
num_credits = 10000
synthetic_data_generator.GenerateUsingNewDataset(new_groups, num_categories, num_responses, num_credits, scaling=True, seed=2)

# ARI and NMI measure similarity between clusters and original groups of people
ari = adjusted_rand_score(synthetic_data_generator.data['Group'], synthetic_data_generator.data['Cluster'])
nmi = normalized_mutual_info_score(synthetic_data_generator.data['Group'], synthetic_data_generator.data['Cluster'])

# Prints which clusters different groups are placed in
for group in groups:
    g = synthetic_data_generator.data[synthetic_data_generator.data['Group'] == group.name]
    print(g['Cluster'].value_counts(), group.name)
print(f'Normalized Mutual Information: {nmi}')
print(f'Adjusted Rand Index: {ari}')

sums = [cluster.iloc[:, 0:num_categories].mean(axis=0) for cluster in synthetic_data_generator.cluster_list]
# Prints sum of votes in clusters
print(sums)
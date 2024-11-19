# qs-clustering-synthetic-data: USE SKEWED DATA GENERATOR
'skewed_synthetic_data_generator' generates synthetic QS data partitioned into diffferent groups, exports data into CSV (might have to change file path), and clusters using kmeans

##### Groups have three attributes: 

- name: String name of group
- occurrence probability: percentage of population this group makes up (e.g. would be 0.504 for males in world population)
- preferences: list of ints representing Group's credit allocations in QS data. For example [0, 5, -2] would mean this group allocates 0 credits to Category_1, 5 credits to Category_2, -2 credits to Category_3 (on average)

The program will then generate QS data using this group information. 

**Clustering**

Uses silhouette score and kmeans to cluster people, and checks how well those clusters match the initially defined groups. 

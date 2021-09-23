import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Setting figure resolution
plt.rcParams['figure.dpi'] = 300

# Read example files
samples = np.load('samples_2.npy')
labels = np.load('labels_2.npy')

# Setting uninitialized models
scaler = StandardScaler()
f_selection = SelectKBest()
pca = PCA()
cluster = KMeans()

# Setting scaler parameters
scaler.set_params(
    with_mean=True,
    with_std=True,
)

# Scaling features
samples = scaler.fit_transform(samples)

# Running feature selection and PCA iteratively
pc1_pc2_variance = []

for i in range(5, samples.shape[1], 5):
    # Setting up feature selection model
    f_selection.set_params(
        score_func=f_classif,
        k=i,
    )

    # Selecting number of features based on i
    selec_features = f_selection.fit_transform(samples, labels)

    # Setting up pca. Caution: Number of components cannot exceed number of samples!
    pca.set_params(
        n_components=selec_features.shape[1]
    )
    
    # Fitting PCA model
    pca.fit(selec_features)

    # Saving explained variance by PC1 and PC2 on every iteration
    pc1_pc2_variance.append(
        pca.explained_variance_ratio_[0:2].sum() * 100
    )

# Visualizing PC1 and PC2 variance
fig, ax = plt.subplots()

ax.bar(
    np.arange(5, samples.shape[1], 5),
    pc1_pc2_variance,
    width=2,
)

ax.set_xlabel("Number of selected compounds", fontsize=18)
ax.set_ylabel(r"Explained variance % by $\rm PC_1 + \rm PC_2$", fontsize=18)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlim([0, 95])
fig.set_size_inches(9, 6)

plt.savefig('explained_variance_example_2.png')

# Visualizing PCA and PCA + kmeans results for PC1 and PC2 for 5, 10, 15 and 20 selected features.
# In this dataset the number of significant features is 20 by design out of 90. By examining 
# the PCA plots for the PC1 and PC2 pair we can see how feature selection helps with removing
# noisy features and improves the cluster separation. The dataset has four clusters by design.

# Initializing matrix with zeros with shape 10 x 4. 4 cases of selected features (5-20 with increments of 5)
# and 10 cases of clustering for each case with number of clusters ranging from 1 to 10.
cluster_errors = np.zeros((10,4))  
p = 0 # Iterator

for j in range(5,25,5):
    f_selection.set_params(
        score_func=f_classif,
        k=j,
    )

    selec_features = f_selection.fit_transform(samples, labels)

    # Setting up pca. Caution: Number of components cannot exceed number of samples!
    pca.set_params(
        n_components=selec_features.shape[1]
    )
    
    scores = pca.fit_transform(selec_features) # Fitting PCA model and transforming at the same time

    
    fig, ax = plt.subplots()
        
    ax.scatter(
        scores[:,0],
        scores[:,1],
        s=25,
        c=labels,
        edgecolor='k',
    )

    ax.set_xlabel(r"$\rm PC_1$ ({}%)".format(round(pca.explained_variance_ratio_[0] * 100, 2)), fontsize=18)
    ax.set_ylabel(r"$\rm PC_2$ ({}%)".format(round(pca.explained_variance_ratio_[1] * 100, 2)), fontsize=18)
    ax.set_title(r"{} selected features".format(j), fontsize=18)
    ax.tick_params(axis='both', labelsize=18)

    custom_legend = (
        [Line2D([0], [0], marker='o', color='w', label='Cluster 1', mfc='#1f77b4', mec='k'),
         Line2D([0], [0], marker='o', color='w', label='Cluster 2', mfc='#511785', mec='k'),
         Line2D([0], [0], marker='o', color='w', label='Cluster 3', mfc='#2ca02c', mec='k'),
         Line2D([0], [0], marker='o', color='w', label='Cluster 4', mfc='#e6e612', mec='k'),
        ]
    )

    ax.legend(handles=custom_legend, loc='best', fontsize=12)

    fig.set_size_inches(9, 6)

    plt.savefig("{}_selected_compounds_example_2.png".format(j))


    # k-means clustering
    # Here for every set of selected features we will run k-means
    # with different number of clusters (1-10) and we will evaluate
    # the elbow plots for each case. We know that we have 4 clusters 
    for ii in range(1,11):
        cluster.set_params(
            n_clusters=ii,
            n_init=10,
        )
        
        # Fitting the k-means model
        cluster.fit(scores[:,[0,1]])

        # Storing sum of squared distances of samples to their closest cluster center.
        cluster_errors[ii - 1, p] = cluster.inertia_

    p = p + 1 # Incrementing iterator


# creating plots for all cases of selected features
fig, ax = plt.subplots()
pos = np.arange(1,11)
for jj in range(0, cluster_errors.shape[1]):
    ax.plot(
        pos,
        cluster_errors[:,jj],
        'o-',
        markersize=12,
        linewidth=2,
    )

legend_handles = (
    [
        Line2D([0], [0], marker='o', color='w', label='5 compounds', ms=12, mfc='#1f77b4'),
        Line2D([0], [0], marker='o', color='w', label='10 compounds', ms=12, mfc='#ff7f0e'),
        Line2D([0], [0], marker='o', color='w', label='15 compounds', ms=12, mfc='#2ca02c'),
        Line2D([0], [0], marker='o', color='w', label='20 compounds', ms=12, mfc='#d62728'),

    ]
)    

ax.set_ylabel("Total euclidean distance", fontsize=18)
ax.set_xlabel("Number of clusters", fontsize=18)
ax.set_title("Elbow plot", fontsize=18)
ax.tick_params(axis='both', labelsize=18)
ax.legend(handles=legend_handles, loc='best', fontsize=12)

fig.set_size_inches(10, 6)
    
plt.savefig('kmeans_elbow_plot_example_2.png')


# Running feature selection, PCA + kmeans for the optimal number features k=20
f_selection.set_params(
    score_func=f_classif,
    k=20,
)

selec_features = f_selection.fit_transform(samples, labels)

# Setting up pca. Caution: Number of components cannot exceed number of samples!
pca.set_params(
    n_components=selec_features.shape[1]
)
    
scores = pca.fit_transform(selec_features) # Fitting PCA model and transforming at the same time


cluster.set_params(
    n_clusters=4,
    n_init=10,
)
        
# Fitting the k-means model
cluster.fit(scores[:,[0,1]])
y = cluster.labels_

fig, ax = plt.subplots()

ax.scatter(
    scores[:,0],
    scores[:,1],
    s=25,
    c=y,
    edgecolor='k'
)

ax.set_xlabel(r"$\rm PC_1$ ({}%)".format(round(pca.explained_variance_ratio_[0] * 100, 2)), fontsize=18)
ax.set_ylabel(r"$\rm PC_2$ ({}%)".format(round(pca.explained_variance_ratio_[1] * 100, 2)), fontsize=18)
ax.set_title(r"{} selected features".format(j), fontsize=18)
ax.tick_params(axis='both', labelsize=18)

fig.set_size_inches(9, 6)
plt.savefig("cluster_results_example_2.png")

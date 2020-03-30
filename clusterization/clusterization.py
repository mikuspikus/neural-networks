from sklearn import datasets

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

random_state = 100

blobs = datasets.make_blobs(
    n_samples = 500, 
    n_features = 10,
    centers = 4,
    cluster_std = 1,
    center_box = (-15.0, 15.0),
    shuffle = True,
    random_state = random_state
)

features, target = blobs

# iris = datasets.load_iris()

# features = iris.data
# target = iris.target

# within cluster summ of squares
wcss = []
start, stop, step = 2, 6, 1
clusters_range = range(start, stop + step, step)

for n_clusters in clusters_range:
    kms = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=random_state
    )

    kms.fit(features)
    wcss.append(kms.inertia_)

plt.plot(clusters_range, wcss)

plt.title('The elbow method')
plt.xlabel('number of clusters')
plt.ylabel('Within cluster summ of squares')
plt.yticks(wcss)
plt.xticks(clusters_range)

plt.show()

feature_dim = (1, 2)

for n_clusters in clusters_range:
    fig, (fst_plt, snd_plt) = plt.subplots(1, 2)
    # fst_plt is silhouette plot
    # silhouette coefficient ranges from [-1, 1]
    fst_plt.set_xlim([-1, 1])
    # (n_clusters + 1) * 10] for spaces
    fst_plt.set_ylim([0, len(features) + (n_clusters + 1) * 10])
    kms = KMeans(n_clusters = n_clusters, random_state = 10)
    cluster_labels = kms.fit_predict(features)

    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f'For n_clusters = {n_clusters} average silhouette is {silhouette_avg}')

    sample_silhouette_values = silhouette_samples(features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)

        fst_plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor = color,
            edgecolor = color,
            alpha = 0.7
        )

        fst_plt.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i, str(i)
        )

        y_lower = y_upper + 10

    fst_plt.set_title('The silhouette for various clusters')
    fst_plt.set_xlabel('The silhouette coefficient values')
    fst_plt.set_ylabel('Cluster label')

    fst_plt.set_yticks([])
    fst_plt.set_xticks(np.arange(-0.1, 1, 0.1))

    fst_plt.axvline(x = silhouette_avg, color = 'red', linestyle = '--')

    colors = cm.nipy_spectral(
        cluster_labels.astype(float) / n_clusters
    )

    snd_plt.scatter(
        features[:, feature_dim[0]], 
        features[:, feature_dim[1]], 
        marker = '.',
        s = 30, lw = 0, alpha = 0.7, c = colors, edgecolor = 'k'
    )

    centers = kms.cluster_centers_
    snd_plt.scatter(
        centers[:, feature_dim[0]],
        centers[:, feature_dim[1]],
        marker = 'o', c = 'white', alpha = 1, s = 200, edgecolor = 'k'
    )

    for i, center in enumerate(centers):
        snd_plt.scatter(
            center[feature_dim[0]], center[feature_dim[1]], 
            marker = f'${i}$', alpha = 1, s = 50, edgecolor = 'k'
        )

    snd_plt.set_title('The visualisation of the clustered data')
    snd_plt.set_xlabel(f'Feature space for the {feature_dim[0]} feature')
    snd_plt.set_ylabel(f'Feature space for the {feature_dim[1]} feature')

    plt.suptitle(
        ("Silhouette analysis for KMeans clustering on sample data "
            "with n_clusters = %d" % n_clusters),
        fontsize=14, fontweight='bold'
    )

plt.show()
#!/bin/python3
import sys
import os
import getopt
import csv
import json
import itertools
import collections
import tlsh
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import *
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

hash_list = []
for i in range(10001):
    hash_list.append(tlsh.hash(os.urandom(256)))

adj = np.zeros((len(hash_list), len(hash_list)), int)

for i in range(len(hash_list)):
    for j in range(len(hash_list)):
        d = tlsh.diff(hash_list[i], hash_list[j]);
        adj[i][j] = d
        adj[j][i] = d

adj = StandardScaler().fit_transform(adj)
#adj, labels_true = make_blobs(n_samples=1001)

#labels_true = make_blobs(n_samples=1001)

# Compute DBSCAN
#db = DBSCAN(eps=0.4, min_samples=10, metric='precomputed').fit(adj)
#db = DBSCAN(eps=0.4, min_samples=10).fit(adj)
#ms = MeanShift(n_jobs=-1).fit(adj)
ms = MiniBatchKMeans(n_clusters=2).fit(adj)
#db = AgglomerativeClustering(n_clusters=3, affinity='precomputed').fit(adj)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
#labels = db.labels_

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
      #% metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
      #% metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
      #% metrics.silhouette_score(adj, labels))

plt.figure(1)
plt.clf()

colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(adj[my_members, 0], adj[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = [plt.cm.Spectral(each)
          #for each in np.linspace(0, 1, len(unique_labels))]
#for k, col in zip(unique_labels, colors):
    #if k == -1:
        ## Black used for noise.
        #col = [0, 0, 0, 1]
#
    #class_member_mask = (labels == k)
#
    #xy = adj[class_member_mask & core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             #markeredgecolor='k', markersize=14)
#
    #xy = adj[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             #markeredgecolor='k', markersize=6)
#
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()

# Hierarchical_Clustering
Code to perform hierarchical spectral or Kmeans clustering using the sklearn library.

This code was developed as part of the DRUID project but this contains the generic code for any hierarchical clustering problem. 

Lukach, M., Dufton, D., Crosier, J., Hampton, J. M., Bennett, L., and Neely III, R. R.: Hydrometeor classification of quasi-vertical profiles of polarimetric radar measurements using a top-down iterative hierarchical clustering method, Atmos. Meas. Tech., 14, 1075–1098, https://doi.org/10.5194/amt-14-1075-2021, 2021.

hierarchical_clustering.py - this is the code that does the clustering. There are 2 loops: the outer loop decides how many levels of the hierarchy to go down, the inner loop decides the optimal number of clusters to split the current cluster data into. If there are a small number of data points in the samples that might be a long way from the main clusters, these can be treated as outliers (spectral clustering is able to pull these out whereas Kmeans clustering in our experience generally does not). Outliers are all gathered into one cluster under root allowing the user to decide what to do with them at a later date.

scores.py - defines a number of different scores that can be used to test whether the clustering is good. The user can choose which score to use for the inner loop and outer loop. Available scores are Wemmert-Gancarski, Calinski-Harabaz, Silhouette, and Davies-Bouldin.

cluster_tree.py - defines a node in the cluster tree. This inherits from anytree.Node allowing printing and plotting of the hierarchy. This file also contains the code to read and write the cluster tree from/to an hdf file.

""" 
 Code to perform hierarchical spectral or Kmeans clustering on given data
 The data is transformed to PCA before the clustering.
 The code consists of an inner loop that does the clustering for a single leaf
 of the current tree, and an outer node that decides whether to accept that
 clustering and whether to go to the next level in the hierarchy.
 The user can chose which scores to use for both the inner and outer loop from
 Wemmert-Gancarski, CalinskiHarabasz, Silhouette or DaviesBouldin
 If a cluster is too sparsely populated it will be added to an outlier node.

 Authors:
 * Written by Julia Crook, Maryna Lukach
   October 2024

""" 
import warnings
from sklearn.preprocessing import QuantileTransformer
from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA

from cluster_tree import *
from scores import *


#---------------------------------------------------------------------------------
# get_PCA()
# Based on the transformed data get the number of PCA components and the representation
# of X_std in the selected variables
# inputs:
#     X_std[nsamples,nvars] - standard deviation of the variables
#     evr_threshold - explained variance ratio threshold - we use this to determine how many variables to use
#     verbose - if True, print what is going on
# returns:
#     selected_pca - the number of variables to use
#     X_new[nsamples, nvars]
#---------------------------------------------------------------------------------
def get_PCA(X_std,evr_threshold,verbose=False):
    pca =  PCA()
    if(verbose):
        print ('get_PCA: input data shape is {}, evr_threshold is {}'.format(X_std.shape, evr_threshold))
    X_new = pca.fit_transform(X_std)

    if(verbose):
        print ('pca.explained_variance_ratio_', pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))
        print ('pca.components_',pca.components_)
    if(np.sum(pca.explained_variance_ratio_[0:(np.max(np.where(pca.explained_variance_ratio_ >= 0.10))+1)]) >= evr_threshold):
        selected_pca = np.max(np.where(pca.explained_variance_ratio_ >= 0.10)) + 1
    else:
        selected_pca = np.max(np.where(pca.explained_variance_ratio_ >= 0.10)) + 2
    if(verbose):
        print ('selected_pca',selected_pca)

    return selected_pca, X_new
    
#---------------------------------------------------------------------------------
# is_cluster_too_small
# Determines clusters that have fewer than a specified minimum number of points
# and returns the indices of the data points belonging to these small clusters.
# inputs:
#     labels (array-like): Cluster labels for the dataset.
#     min_cluster_size (int): The minimum number of points that a cluster should have.
#     verbose - if True, print what is going on
# returns:
#     small_clusters_indices : array of all indices belonging to these small clusters.
#---------------------------------------------------------------------------------
def is_cluster_too_small(labels, min_cluster_size, verbose=False):

    unique_labels,label_counts = np.unique(labels, return_counts=True)
    small_clusters_indices = []

    for i in range(len(unique_labels)):
        if verbose:
            print('cluster_label = {} label_counts[cluster_label] = {}'.format(unique_labels[i],label_counts[i] ))
        if label_counts[i] < min_cluster_size:
            # Get indices of data points belonging to the small cluster
            indices = np.where(labels == unique_labels[i])
            small_clusters_indices.append(indices[0])
    if len(small_clusters_indices)>0:
        small_clusters_indices=np.concatenate(small_clusters_indices)
    return small_clusters_indices

#---------------------------------------------------------------------------------
# remove_label_gaps
# Replaces the values of labels in the input list so that there are no more gaps in the list of unique labels,
# while keeping the order of label values.
# inputs:
#        labels (list): List of labels, which may have gaps in the unique values.
#
# Returns:
#        new_labels: A new numpy array of labels with the gaps removed.
#---------------------------------------------------------------------------------
def remove_label_gaps(labels):
    # Get the sorted unique labels from the input list
    unique_labels = np.unique(labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    # Apply the mapping to the input list of labels
    new_labels = np.asarray([label_mapping[label] for label in labels])
    return new_labels

#---------------------------------------------------------------------------------
# get_replacement_current_cluster_labels
# replaces all values of 0 in the model_labels with the old_label (that which this cluster had) and set all other non zero labels
# as next_cluster_label plus value in model_labels. Leave the OUTLIER_LABELS as they are.
# inputs:
#    old_label - the label of the cluster that has been subclustered
#    model_labels - the labels that the clustering algorithm came up with. Usually 0-n, but may contain OUTLIER_LABEL
#    next_cluster_label - this is the next available number to use for new labels
# Returns:
#    new_labels: A new numpy array of labels (same size as model_labels) with the 0s replaced by old_label
#                and the other positive labels replaced by next_cluster_label upwards.
#---------------------------------------------------------------------------------
def get_replacement_current_cluster_labels(old_label, model_labels, next_cluster_label):
    # where the model_labels is 0 set to the old label
    # for all the other positive model_labels set to next_cluster_label upwards
    new_labels=np.copy(model_labels)
    ix_zero=np.where(model_labels==0)
    ix_non_zero=np.where(model_labels>0)
    new_labels[ix_zero[0]]=old_label
    new_labels[ix_non_zero[0]]=model_labels[ix_non_zero]+next_cluster_label-1
    return new_labels

#---------------------------------------------------------------------------------
# get_subclusters
# The inner loop of the clustering that finds the optimum number of splits for this_cluster_data
# Starts with nclusters=2, uses sklearn spectral_clustering or kmeans to do the split, checks if there are outliers
# calculates a new score (not including the outliers) to determine if this split is better than previous split
# If better we go round the loop again with nclusters incremented
#
# inputs:
#     this_cluster_data - the data[nsamples,nvars] for the current cluster
#     evr_threshold - explained variance ratio threshold - we use this to determine how many variables to use in the PCA calculation
#     min_npoints_to_cluster - if a cluster holds less than this number of points we wont try to subcluster
#     max_npoints_for_outlier - the number of points that defines a subcluster as being an outlier
#     scorer_type - the type of scorer to use for the inner loop to determine when to stop
#     cluster_name - full label of the cluster we are working on (just for printing purposes)
#     use_kmeans - if True, use kmeans instead of spectral clustering
#     verbose - if True, print what is going on
# returns:
#     prev_labels - the labels defining the new sub clusters of this cluster
#     X_new - the pca data for this cluster
#---------------------------------------------------------------------------------
def get_subclusters(this_cluster_data, evr_threshold, min_npoints_to_cluster, max_npoints_for_outlier, scorer_type, cluster_name, use_kmeans=False, verbose=False):

    # Initialize variables
    number_of_clusters = 2
    scorer=create_scorer(scorer_type)
    split_is_better=True
    
    # convert data to pca
    X_std = QuantileTransformer(output_distribution='normal').fit_transform(this_cluster_data)
    selected_pca, X_new = get_PCA(X_std, evr_threshold, verbose=verbose)
    
    active_indices=np.arange(this_cluster_data.shape[0])  # initially all indices for this data
    nactive=this_cluster_data.shape[0]
    new_labels=np.zeros(this_cluster_data.shape[0], int)
    prev_labels=[]
    clust_type='spectral'
    if use_kmeans:
        clust_type='kmeans'
    
    while split_is_better:
        if verbose:
            print('get_subclusters: number_of_clusters {}'.format(number_of_clusters))

        X_new_filtered = X_new[active_indices, 0:selected_pca] # used to do a copy?
        if verbose:
            print('get_subclusters: doing {} clustering of {} with X_new_filtered.shape = {}'.format(clust_type, cluster_name, X_new_filtered.shape))

        # Apply spectral clustering
        if use_kmeans:
            model = cluster.KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=0, n_init=10) 
            kmeans = model.fit(X_new_filtered)
        else:
            model = cluster.SpectralClustering(n_clusters=number_of_clusters, assign_labels='discretize')
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                    message='the number of connected components of the connectivity matrix is [0-9]{1,2}' +
                ' > 1. Completing it to avoid stopping the tree early.',
                    category=UserWarning)
                warnings.filterwarnings('ignore',
                    message='Graph is not fully connected, spectral embedding may not work as expected.',
                    category=UserWarning)
            model.fit(X_new_filtered)
        labels = model.labels_
        unique_lab=np.unique(labels)

        if verbose:
            print('get_subclusters: finding small clusters')
        # Check and remove small clusters
        small_clusters_indices = is_cluster_too_small(labels, max_npoints_for_outlier,verbose=verbose)
        noutliers=len(small_clusters_indices)
        if (noutliers>0):
            # update the labels to indicate outlier
            labels[small_clusters_indices]=OUTLIER_LABEL
            # recalculate active_indices and make sure the non outlier labels are contiguous from 0
            this_active_indices=np.where(labels!=OUTLIER_LABEL)
            this_labels = remove_label_gaps(labels[this_active_indices])
            labels[this_active_indices]=this_labels
            noutliers=1
            
        new_labels[active_indices]=labels

        # recalculate active_indices to exclude outliers for score calculation
        active_indices=np.where(new_labels!=OUTLIER_LABEL)[0]
        nactive=len(active_indices)
        if nactive<min_npoints_to_cluster:
            # we cannot continue to split
            split_is_better=False
            if verbose:
                print('get_subclusters: remaining active points to few to cluster - abandoning')
            continue
        
        if number_of_clusters - noutliers==1:            
            # only 1 cluster left if we remove the outliers
            if verbose:
                print("get_subclusters: only 1 cluster left after removing outliers")
            continue # we go to the next iteration of while loop with the active_indices to try again without the outliers
            
        X_new_filtered=X_new[active_indices, 0:selected_pca]
        this_score=scorer.compute_score(X_new_filtered, new_labels[active_indices])
        if verbose:
            print('get_subclusters: calculated {} score as'.format(scorer.name), this_score)
        if(number_of_clusters > 2):
            split_is_better=scorer.split_is_better(this_score)
        if split_is_better:
            # we are going round loop again so save these labels and score
            # if we found outliers then don't change the number of clusters but do the clustering again without the outliers
            # otherwise increment the number of clusters
            if noutliers==0:
                number_of_clusters += 1
            scorer.add_score(this_score)# save the scores
            prev_labels=np.copy(new_labels) # save labels from this loop
            if verbose:
                print('get_subclusters: accepting split and continuing') 
        elif verbose:
            print('get_subclusters: rejecting split and stopping') 

    # we stopped because we found a worse split so use prev_labels
    # recalculate active_indices to exclude outliers
    active_indices=np.where(prev_labels!=OUTLIER_LABEL)[0]
    nactive=len(active_indices)
    if nactive==0:
        print('get_subclusters: no active cluster any more!!!!!')
        pdb.set_trace()
    print('get_subclusters: loop complete')
    return prev_labels, X_new

#---------------------------------------------------------------------------------
# perform_hierarchical_clustering
# The outer loop of the clustering
# inputs:
#    cluster_data[namples,nvars] - all the data that will be clustered
#    field_list - list of the names of the radar variables in the cluster_data (this is stored in root of the tree)
#    evr_threshold - explained variance ratio threshold - we use this to determine how many variables to use in the PCA calculation
#    min_pc_points_to_cluster - if a cluster holds less than this % x number of samples we wont try to subcluster
#    max_pc_points_for_outlier - defines the number of points (this % x number of samples) that defines an outlier
#    scorer_outer - the type of scorer for the outer loop
#    scorer_inner - the type of scorer for the inner loop
#    output_dir - where to store the output
#    plot_fname_prefix - the prefix for the output filename
#    use_kmeans - if True use kmeans otherwise use spectral clustering
#    verbose - if True, print what is going on
#---------------------------------------------------------------------------------
def perform_hierarchical_clustering(cluster_data, field_list, evr_threshold, min_pc_points_to_cluster, max_pc_points_for_outlier, scorer_outer, scorer_inner, output_dir, plot_fname_prefix, use_kmeans=False, verbose=False):
        
    nsamples=cluster_data.shape[0]
    min_npoints_to_cluster=int(min_pc_points_to_cluster*nsamples/100)
    max_npoints_for_outlier=int(max_pc_points_for_outlier*nsamples/100)
    if verbose:
        print(nsamples, 'samples, min points to cluster=', min_npoints_to_cluster, 'max points for outlier=',max_npoints_for_outlier)
    all_labels = np.zeros(nsamples, int)
    possible_labels=np.copy(all_labels)
    # whole dataset is the root cluster
    # create root node of the cluster-tree
    root = ClusterNode('root',None,np.arange(nsamples),data=cluster_data,variables=field_list)
    outlier_node=None
    
    n_level = 0
    best_split = False
    scorer=create_scorer(scorer_outer)
 
    # best_split is the result of scorer calculation,
    while (best_split is False):

        n_splits = 0

        if(verbose):
            print('perform_hierarchical_clustering: we are starting the new loop for level',n_level)

        # select all current active clusters
        active_clusters=[]
        root.leaves(active_clusters, ignore_outlier=True)
        nactive_clusters=len(active_clusters)
        next_cluster_label=nactive_clusters
        cluster_sizes=np.asarray([len(cluster.indxs) for cluster in active_clusters])
        sorted_ix = np.flip(np.argsort(cluster_sizes)) # sort in descending order
        if verbose:
            print(nactive_clusters, 'active clusters: sorted clusters is', sorted_ix, 'and the sizes are', cluster_sizes[sorted_ix])

        for k in sorted_ix:
            cluster = active_clusters[k]
            if hasattr(cluster, 'complete'):
                # we tried this before so skip
                continue
            current_cluster_data = cluster_data[cluster.indxs]
            clust_name = cluster.get_full_name() # get the full name so we know where we are in the hierarchy
            if verbose:
                print('perform_hierarchical_clustering: we work with cluster {} name {} of size {}, next cluster_label= {}'.format(k,clust_name,cluster_sizes[k],next_cluster_label))

            # threshold on the minimum number of points to cluster
            if (current_cluster_data.shape[0]>min_npoints_to_cluster):

                best_labels, current_cluster_pca_data = get_subclusters(current_cluster_data, evr_threshold, min_npoints_to_cluster, max_npoints_for_outlier, scorer_inner, clust_name, use_kmeans=use_kmeans, verbose=verbose)
                # find outliers
                ix_outlier=np.where(best_labels==OUTLIER_LABEL)[0]
                noutliers=len(ix_outlier)
                cluster_outlier_indxs = cluster.indxs[ix_outlier]
                ix_non_outlier=np.where(best_labels!=OUTLIER_LABEL)[0]
                nnon_outlier=len(ix_non_outlier)
                if verbose:
                    print('perform_hierarchical_clustering: ', noutliers, 'outlier points', nnon_outlier, 'non outlier points in cluster', clust_name)
                # Use ix_non_outlier to filter current_cluster_data, current_cluster_indxs for further processing
                current_cluster_data_filtered = current_cluster_data[ix_non_outlier]
                current_cluster_pca_data_filtered=current_cluster_pca_data[ix_non_outlier]
                current_cluster_indxs_filtered = cluster.indxs[ix_non_outlier]
                nsub_clusters_filtered = len(np.unique(best_labels[ix_non_outlier]))

                current_label=np.unique(possible_labels[current_cluster_indxs_filtered])[0]
                replacement_labels=get_replacement_current_cluster_labels(current_label, best_labels, next_cluster_label)
                if(verbose):
                    print('perform_hierarchical_clustering: replace current cluster {}, label {}, by its subcluster labels'.format(clust_name, current_label), np.unique(replacement_labels))
                possible_labels[cluster.indxs]=replacement_labels
                if verbose:
                    print('perform_hierarchical_clustering: poss labels after inner loop for cluster', clust_name, np.unique(possible_labels))
                # get all the labels and data that are not outliers
                ix_good=np.where(possible_labels!=OUTLIER_LABEL)[0]
                possible_labels_filtered=possible_labels[ix_good]
                all_data_filtered=cluster_data[ix_good]
                # compute score for this split on the original dataset minus the outliers
                # should this be calculated on actual data or pca data?
                new_score=scorer.compute_score(all_data_filtered, possible_labels_filtered,verbose=verbose)
                if(verbose):
                    print('perform_hierarchical_clustering: got the new {} score'.format(scorer.name), new_score)
                split_is_better = scorer.split_is_better(new_score)

                if split_is_better==False:
                    # go back to previous accepted labels and mark this cluster as tested so we don't try doing it again in the next loop
                    possible_labels=np.copy(all_labels)
                    cluster.complete=True
                    if(verbose):
                        print('perform_hierarchical_clustering: level=', n_level," we don't accept this split")
                        print('perform_hierarchical_clustering: possible labels set back to all_labels', np.unique(possible_labels))
                        
                else:
                    if(verbose):
                        print('perform_hierarchical_clustering: level=',n_level," we accept this split and write all subclusters to the tree")
                    # get the medoids of the clusters
                    # calculate the centroids for the new clusters
                    clf = NearestCentroid(metric='euclidean')
                    print('perform_hierarchical_clustering: calculating centroids for current cluster with data:', current_cluster_pca_data_filtered.shape, best_labels[ix_non_outlier].shape, type(best_labels[0]))
                    clf.fit(current_cluster_data_filtered, best_labels[ix_non_outlier])
                    medoids_indx_filtered, distances_filtered =  pairwise_distances_argmin_min(clf.centroids_, current_cluster_data_filtered)
                    medoids = current_cluster_data[medoids_indx_filtered,:]

                    # now create nodes to add to the tree for the new sub clusters
                    if(verbose):
                        print('perform_hierarchical_clustering: creating {} subclusters'.format(nsub_clusters_filtered))
                    for c in range(nsub_clusters_filtered):
                        ix=np.where(best_labels[ix_non_outlier]==c)[0]
                        current_subcluster_indxs = current_cluster_indxs_filtered[ix]
                        subcluster = ClusterNode("cl{}".format(c+1), cluster,current_subcluster_indxs, centroid=clf.centroids_[c],medoid=medoids[c])

                    # now create any outlier node
                    if noutliers>0:
                        outlier_data = current_cluster_data[ix_outlier]
                        if outlier_node==None:
                            if(verbose):
                                print('perform_hierarchical_clustering: creating outlier node')
                            outlier_node = ClusterNode(OUTLIER_NAME, root, cluster_outlier_indxs)
                        else:
                            # add these outlier indices to the already existing outlier node
                            outlier_node.indxs=np.concatenate([outlier_node.indxs, cluster_outlier_indxs])
                    # copy the labels we worked out this time to all_labels and save the score
                    all_labels=np.copy(possible_labels)
                    next_cluster_label=np.amax(all_labels)+1
                    scorer.add_score(new_score, next_cluster_label)
                    n_splits += 1


        if (n_splits == 0):
            best_split = True
            if verbose:
                print('perform_hierarchical_clustering: level {}. No new splits were done and best_split is reached'.format(n_level))
                print('perform_hierarchical_clustering: The list of scores looks like', scorer.scores)
        elif verbose:
            print('perform_hierarchical_clustering: level {}. We go to the next round'.format(n_level))
        this_plot_fname="{}{}_tree_{}_loop.png".format(output_dir,plot_fname_prefix, n_level)
        if verbose:   
            print('perform_hierarchical_clustering: plotting tree for loop', this_plot_fname)
        root.plot_tree(this_plot_fname)
        
        out_fname='{}_{}_data_{}_level_{}_active_clusters.hdf'.format(plot_fname_prefix, root.name, str(n_level), str(next_cluster_label))
        if verbose:
            print('perform_hierarchical_clustering: writing tree data for level', n_level, out_fname)
        clust_type='spectral'
        if use_kmeans:
            clust_type='Kmeans'
        write_cluster_tree_hdf(root, clust_type, scorer_inner, scorer_outer, max_npoints_for_outlier, min_npoints_to_cluster, output_dir+out_fname)

        n_level += 1
    scorer.plot_scores(output_dir+plot_fname_prefix+'_')

    print('perform_hierarchical_clustering: loop complete clustering is finished')
    return root


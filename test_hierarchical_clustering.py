import pdb
from hierarchical_clustering import *

def test_remove_label_gaps():
    print('remove_label_gaps test')
    labels=np.array([0,0,0,0,1,1,3,3,3,3,4,4,4,6,6,7,7,7,7])
    exp_labels=np.copy(labels)
    ix=np.where(labels==3)
    exp_labels[ix]=2
    ix=np.where(labels==4)
    exp_labels[ix]=3
    ix=np.where(labels==6)
    exp_labels[ix]=4
    ix=np.where(labels==7)
    exp_labels[ix]=5
    labels=remove_label_gaps(labels)
    assert(np.all(labels==exp_labels))

def test_is_cluster_too_small():
    print('is_cluster_too_small test')
    labels=np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2])
    small_clusters_indices=is_cluster_too_small(labels, 10, verbose=True)
    exp_indices=np.where(labels>=1)[0]
    assert(np.all(small_clusters_indices==exp_indices))
    
def test_replace_current_cluster_labels():
    print('replace_current_cluster_labels test')
    # case where we have only 1 node and we have split it into 2
    old_label=0
    model_labels=np.asarray([0,0,1,1,1,1,0,1,0,0])
    next_cluster_label=1
    new_labels=get_replacement_current_cluster_labels(old_label, model_labels, next_cluster_label)
    exp_new_labels=model_labels
    assert(np.all(new_labels==exp_new_labels))
    # case where we have 2 existing nodes so next_cluster_label=2 and we have split label 0 into 3
    old_label=0
    model_labels=np.asarray([0,0,0,1,2,1,0,2])
    next_cluster_label=2
    new_labels=get_replacement_current_cluster_labels(old_label, model_labels, next_cluster_label)
    exp_new_labels=np.asarray([0,0,0,2,3,2,0,3])
    assert(np.all(new_labels==exp_new_labels))
    
def test_hierarchical_clustering():
    print('hierarchical_clustering test')
    nsamples=5000
    nsamples_set1=1550
    ix_set1=np.arange(nsamples_set1)
    nsamples_set2=3400
    ix_set2=np.arange(nsamples_set2)+nsamples_set1
    nsamples_out=nsamples-nsamples_set1-nsamples_set2
    ix_out=np.arange(nsamples_out)+nsamples_set1+nsamples_set2
    labels=np.zeros(nsamples,int)
    nvars=4
    X=np.zeros((nsamples,nvars))
    labels[ix_set1]=np.asarray([0 for i in range(nsamples_set1)])
    labels[ix_set2]=np.asarray([1 for i in range(nsamples_set2)])
    labels[ix_out]=np.asarray([-9 for i in range(nsamples_out)])
    means=np.zeros((3,nvars))
    for v in range(nvars):
        X[ix_set1,v]=np.random.uniform(size=nsamples_set1)
        means[0,v]=np.mean(X[ix_set1,v])
        X[ix_set2,v]=np.random.uniform(size=nsamples_set2)+5
        means[1,v]=np.mean(X[ix_set2,v])
        X[ix_out,v]=np.random.uniform(size=nsamples_out)-100
        means[2,v]=np.mean(X[ix_out,v])

    pc_points_to_cluster=6
    pc_outier_points=100*(nsamples_out+20)/nsamples
    root=perform_hierarchical_clustering(X, ['A','B','C','D'], 0.8, pc_points_to_cluster, pc_outier_points, 'WG', 'SIL', './testing/', 'test_spectral', verbose=True)
    leaves=[]
    root.leaves(leaves)
    leaves=np.asarray(leaves)
    print('spectral')
    root.print()
    npoints=np.asarray([len(leaf.indxs) for leaf in leaves])
    assert(len(leaves)==3)
    ixsort=np.argsort(npoints)
    npoints=npoints[ixsort]
    print(npoints)
    sorted_leaves=leaves[ixsort]
    print(nsamples, 'samples', nsamples_set1, 'set1', nsamples_set2, 'set2', nsamples_out,'outliers')
    assert(npoints[0]>nsamples_out-5 and npoints[0]<nsamples_out+5)
    assert(npoints[1]>nsamples_set1-15 and npoints[1]<nsamples_set1+15)
    assert(npoints[2]>nsamples_set2-15 and npoints[2]<nsamples_set2+15)
    assert(sorted_leaves[0].name==OUTLIER_NAME)
    med_dist_11=np.sum((sorted_leaves[1].medoid-means[0,:])**2)
    med_dist_12=np.sum((sorted_leaves[1].medoid-means[1,:])**2)
    med_dist_1o=np.sum((sorted_leaves[1].medoid-means[2,:])**2)
    print('med dist set1 to set 1', med_dist_11, 'to set 2', med_dist_12, 'to outlier', med_dist_1o)
    assert(med_dist_11<med_dist_12)
    assert(med_dist_11<med_dist_1o)
    med_dist_21=np.sum((sorted_leaves[2].medoid-means[0,:])**2)
    med_dist_22=np.sum((sorted_leaves[2].medoid-means[1,:])**2)
    med_dist_2o=np.sum((sorted_leaves[2].medoid-means[2,:])**2)
    print('med diff set2 to set 1', med_dist_21, 'to set 2', med_dist_22, 'to outlier', med_dist_2o)
    assert(med_dist_22<med_dist_21)
    assert(med_dist_22<med_dist_2o)
    
    print('now doing KMeans')
    root=perform_hierarchical_clustering(X, ['A','B','C','D'], 0.8, pc_points_to_cluster, pc_outier_points, 'SIL', 'SIL', './testing/', 'test_kmeans', use_kmeans=True, verbose=True)
    leaves=[]
    root.leaves(leaves)
    leaves=np.asarray(leaves)
    print('kmeans')
    root.print()
    npoints=np.asarray([len(leaf.indxs) for leaf in leaves])
    ixsort=np.argsort(npoints)
    npoints=npoints[ixsort]
    print(npoints)
    # Kmeans doesn't handle outliers well so groups them in with set 1
    print(nsamples, 'samples', nsamples_set1, 'set1', nsamples_set2, 'set2', nsamples_out,'outliers')
    sorted_leaves=leaves[ixsort]
    print('set2 indxs', np.amin(sorted_leaves[1].indxs), np.amax(sorted_leaves[1].indxs))
    print('set1 indxs', np.amin(sorted_leaves[0].indxs), np.amax(sorted_leaves[0].indxs))
    assert(npoints[0]>nsamples_set1+nsamples_out-10 and npoints[0]<nsamples_set1+nsamples_out+10)
    assert(npoints[1]>nsamples_set2-15 and npoints[1]<nsamples_set2+15)
    med_dist_22=np.sum((sorted_leaves[1].medoid-means[1,:])**2)
    med_dist_21=np.sum((sorted_leaves[1].medoid-means[0,:])**2)
    print('med diff set2 to set 1', med_dist_21, 'to set 2', med_dist_22)
    assert(med_dist_22<med_dist_21)

test_remove_label_gaps()
test_is_cluster_too_small()
test_replace_current_cluster_labels()
test_hierarchical_clustering()
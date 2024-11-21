from cluster_tree import *
import numpy as np

def check_clusters(clusters, leaves, root):
    # check that clusters makes sense
    passed=True
    for n in range(len(leaves)):
        if leaves[n].name=='outlr':
            label=OUTLIER_LABEL
        else:
            label=n
        ix=np.where(clusters==label)
        # check that the leaves[n].indxs matches these
        if len(ix[0])!=len(leaves[n].indxs):
            print('node=',n,'mismatching number of indices set for node compared to number in node itself',len(ix[0]), len(end_nodes[n].indxs))
            passed=False
            pdb.set_trace()
        else:
            ixbad=np.where(abs(root.indxs[ix[0]]-np.sort(leaves[n].indxs))>0)
            if len(ixbad[0])>0:
                print('node=',n,'mismatching root.indxs and leaves[n].indxs',n)
                passed=False
                pdb.set_trace()
    return passed

#check a dummy case of a tree
nsamples=100
data=np.zeros((nsamples,2))
data[:,0]=np.arange(nsamples)+10
data[:,1]=np.arange(nsamples)*2
variables=["var1","var2"]
indxs=np.arange(nsamples)
root=ClusterNode('root',None,indxs,data=data,variables=variables)
child1=ClusterNode('cl1',root,np.arange(20),medoid=[20,20])
child2=ClusterNode('cl2',root,np.arange(20)+20,medoid=[30,30])
print('my test tree')
root.print()
root.plot_tree('./testing/test_cluster_tree.png')
all_var2=root.get_data_by_key('var2')
diffs=all_var2-data[:,1]
assert(np.sum(abs(diffs))==0)
var1=child1.get_data_by_key('var1')
diffs=var1-data[:20,0]
assert(np.sum(abs(diffs))==0)

# write it then read back in to check it matches
fname='./testing/test_cluster_tree.hdf'
clust_type='spectral'
inner_score_type='SIL'
outer_score_type='WG'
outlier_threshold=100
split_threshold=2000
my_attrs={'clustering_type':clust_type,'inner_score':inner_score_type,'outer_score':outer_score_type,
'outlier_threshold':outlier_threshold,'split_threshold':split_threshold}
write_cluster_tree_hdf(root,clust_type, inner_score_type, outer_score_type, outlier_threshold, split_threshold, fname, verbose=True)

root2, leaves, attrs=read_cluster_tree_hdf(fname,verbose=True)
assert(len(leaves)==2)
assert(leaves[0].name=='cl1')
assert(leaves[1].name=='cl2')
root2.print()
for key in my_attrs.keys():
    assert(attrs[key]==my_attrs[key])
# check that we can get the clusters for all data
clusters=get_leaf_index_for_input_indx(leaves, root2)
assert(check_clusters(clusters, leaves, root2))

# add an outlier child and check you can get leaves with and without outlier
outlier=ClusterNode(OUTLIER_NAME,root,np.arange(5))
print('tree with outlier')
root.print()
print('tree ignoring outlier')
root.print(ignore_outlier=True)
leaves=[]
root.leaves(leaves)
assert(len(leaves)==3)
leaves=[]
root.leaves(leaves,ignore_outlier=True)
assert(len(leaves)==2)



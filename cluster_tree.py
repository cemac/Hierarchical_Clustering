""" 
 Code to define a node within a tree and the data associated with clustering for each node

 Authors:
 * Written by Julia Crook, Maryna Lukach
   October 2024

""" 
import numpy as np
import h5py
import pdb
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter

OUTLIER_LABEL=-9 # used to define an OUTLIER node
OUTLIER_NAME='outlr' # name that the outlier node is given

#---------------------------------------------------------------------------------
# class to hold data for a node in a tree - inherits fron anytree.Node (used for peinting and plotting)
# we expect the tree to hold data and variables only in the root node and the indxs in each sub node in the
# tree will define which of the data in root is included in this node.
#---------------------------------------------------------------------------------
class ClusterNode(Node):

    def __init__(self,name,parent,indxs,data=[],variables=[],medoid=[]):
        super().__init__(name, parent=parent)
        self.indxs = np.asarray(indxs)
        if parent==None:
            if len(data)==0:
                raise ValueError('data not given for root')
            if len(variables)==0:
                raise ValueError('variables (names of variables) not given for root')
        self.data = data # either [] or an array of (nsamples, nvars)
        self.medoid = medoid
        self.variables=variables # holds the meaning of the nvars in data

    def get_root(self):
        if self.parent!=None:
            root=self.parent.get_root()
        else:
            root=self
        return root
            
    # extract requested information from data (self.variables holds the keys)
    def get_data_by_key(self, key):
        if len(self.data)==0 or len(self.variables)==0:
            print('ClusterNode:get_data_by_key() no data or variables in this node, finding data from root data')
            # this node does not hold data but root does so get the data from root and use the indxs to find this node's data
            root=self.get_root()
            variables=root.variables
            data=np.asarray(root.data)
            ixs=np.concatenate([np.where(root.indxs==i)[0] for i in self.indxs])
            data=data[ixs,:]
        else:
            variables=self.variables
            data=np.asarray(self.data)
        vix=np.where(np.asarray(variables)==key)
        if len(vix[0])>0:
            data=data[:,vix[0][0]]
        else:
            raise ValueError(key+' Not in data')

        return data
        
    def print(self, ignore_outlier=False):
        if len(self.variables)>0:
            root_variables=self.variables
        for pre, _, node in RenderTree(self):
            if (ignore_outlier==False or node.name!=OUTLIER_NAME):
                data_shape=0
                if len(node.data)>0:
                    data_shape=np.asarray(node.data).shape
                is_leaf_str=''
                if node.is_leaf:
                    is_leaf_str='is leaf,'
                if len(self.variables)>0:
                    variables=self.variables
                else:
                    variables=root_variables
                nmed=len(node.medoid)
                medoid_str=''
                if nmed>0:
                    med_strs=['{v}={m:.2f}'.format(v=variables[i],m=node.medoid[i]) for i in range(nmed)]
                    medoid_str='medoids: '+', '.join(med_strs)+','
                print('%s%s %s %s' % (pre, node.name, is_leaf_str, medoid_str), len(node.indxs), 'indxs, data.shape=', data_shape)

    # use anytree dotexporter library for plotting
    def plot_tree(self, fname):
        UniqueDotExporter(self).to_picture(fname)
        
    # get the label as a string from the top of the hierarchy by working back up through the parent
    # eg root/cl1/cl1/cl2
    def get_label(self):
        label_str=self.name
        parent=self.parent
        while parent!=None:
            label_str=parent.name+'/'+label_str
            parent=parent.parent
            
        return label_str

    # find the end nodes (leaves) of the tree
    def leaves(self, leaves, ignore_outlier=False):
        if self.is_leaf and (ignore_outlier==False or self.name!=OUTLIER_NAME):
            leaves.append(self)
        else:
            for node in self.children:
                node.leaves(leaves, ignore_outlier=ignore_outlier)


#-------------------------------------------------------------------
# functions to write the tree to an hdf file
#-------------------------------------------------------------------
# write_cluster_group()
# writes the group related to cluster to parent_group and calls itself for all children of this cluster
# inputs:
#    cluster - the current node (ClusterNode)
#    parent_group - the hdf group under which we should write this data
#-------------------------------------------------------------------
def write_cluster_group(cluster, parent_group):
    cluster_group = parent_group.create_group(cluster.name)
    if len(cluster.medoid)>0:
        medoid=np.asarray(cluster.medoid)
        medoid_id = cluster_group.create_dataset('medoid', medoid.shape, data=medoid)
    indxs_id = cluster_group.create_dataset('indxs', cluster.indxs.shape, data=cluster.indxs)
    if len(cluster.data)>0:
        data_id = cluster_group.create_dataset('data', cluster.data.shape, data=cluster.data)

    if len(cluster.variables)>0:
        variables=np.asarray([var.encode('utf-8') for var in cluster.variables])
        variables_id = cluster_group.create_dataset('variables', variables.shape, data=variables)

    for subcluster in cluster.children:
        group_id = write_cluster_group(subcluster, cluster_group)
        
    return cluster_group

#-------------------------------------------------------------------
# write_cluster_tree_hdf()
#    save the tree to the HDF5-file
# inputs:
#    root - the top node of the tree (ClusterNode)
#    out_filename - where to save the hdf file
#-------------------------------------------------------------------
def write_cluster_tree_hdf(root, out_filename,verbose=False):
    
    f = h5py.File(out_filename, 'w')
    parent_group = write_cluster_group(root, f)
    if verbose:
        print ('writing the {} file is finished'.format(out_filename))
    f.close()


#-------------------------------------------------------------------
# functions to read the tree in an hdf file
#-------------------------------------------------------------------
# read_keys()
# inputs:
#    group - the next part of the data belonging to the tree
#    current_node - the current node (ClusterNode) - None if we haven't created root yet
#    leaves - a list of the current leaves to which we can add this node if it is a leaf
#    ignore_outlr - if true don't include  the outlr cluster in leaves
#    verbose - if true do some printing
# returns:
#    root
#-------------------------------------------------------------------
def read_keys(group, current_node, leaves, ignore_outlr, verbose):
    root=None
    keys=group.keys()
    for key in keys:
        this_group=group[key]
        if isinstance(this_group, h5py._hl.dataset.Dataset):
            continue
        elif isinstance(this_group,h5py._hl.group.Group):
            # get the data out of the group - all levels should have indxs
            this_keys=this_group.keys()
            if 'indxs' not in this_keys:
                raise ValueError('indxs not in group '+key)
            indxs=this_group['indxs'][:]
            medoid=[]
            if 'medoid' in this_keys:
                medoid=this_group['medoid'][:]
            pca=[]
            if 'pca' in this_keys:
                pca=this_group['pca'][:] # we wont bother storing this in tree
                #print('file has pca for', key)
            data=[]
            if 'data' in this_keys:
                data=this_group['data'][:]
            variables=[]
            if 'variables' in this_keys:
                variables=this_group['variables'][:]
                if isinstance(variables[0], np.bytes_):
                    variables2=[var.decode('utf-8') for var in variables]
                    variables=variables2
            if verbose:
                parent_label=''
                if current_node!=None:
                    parent_label=current_node.get_label()
            new_cluster=ClusterNode(key, current_node, indxs, data=data, variables=variables, medoid=medoid)
            if current_node==None:
                root=new_cluster
            read_keys(this_group, new_cluster, leaves, ignore_outlr, verbose)
        else:
            raise Exception('unexpected group type')
            
    # now we have gone through all elements in this group we know if there are child elements
    # if this is a leaf add it to the leaves list
    if current_node!=None:
        if current_node.is_leaf:
            if ignore_outlr==False or current_node.name!=OUTLIER_NAME:
                leaves.append(current_node)
    return root

#-------------------------------------------------------------------
# read_cluster_tree_hdf()
# opens the file and recursively calls read_keys to traverse the tree
# inputs:
#    filename - full path and filename of file to read
#    verbose - print what we are opening
# returns:
#    root - the top element of the tree that points to all elements in the tree
#    leaves - a list of the elements that are end nodes, i.e. active clusters
#-------------------------------------------------------------------
def read_cluster_tree_hdf(filename, ignore_outlr=False, verbose=False):
    try:
        if verbose:
            print('opening ',filename)
        group = h5py.File(filename, 'r')
    except OSError as err:
        print(filename, err)
        raise err

    leaves=[] # a list of the leaves (these are the active clusters)
    root=read_keys(group, None, leaves, ignore_outlr, verbose)
    return root, leaves

#-------------------------------------------------------------------
# The tree holds an indx of the data that are part of this node for each leaf in the tree.
# This function switches this information round to give a leaf index (ie cluster index) 
# for each sample (indx) in root.
# inputs:
#     leaves - the leaves of the tree (may include or not include outliers)
#     root - the root of the tree
# returns:
#     array of leaf indices for each data point in root
# Note that any root.indxs that have not been put in an end_node must be in outliers
#-------------------------------------------------------------------
def get_leaf_index_for_input_indx(leaves, root):

    leaf_indxs=np.zeros_like(root.indxs, int)+OUTLIER_LABEL
    for n in range(len(leaves)):
        if leaves[n].name==OUTLIER_NAME:
            continue
        for ix in leaves[n].indxs:
            root_ix=np.where(root.indxs==ix)
            if len(root_ix[0])!=1:
                raise Exception('node={}'.format(n)+' zero or more than 1 root indxs found for cluster index {}'.format(ix))
            if leaf_indxs[root_ix[0][0]]!=OUTLIER_LABEL:
                raise Exception('node={}'.format(n)+' leaf_indxs already set for root_ix {}'.format(root_ix[0][0]))
            leaf_indxs[root_ix[0][0]]=n
                
    return leaf_indxs

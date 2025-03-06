""" 
 Code to handle scores

 Authors:
 * Written by Julia Crook, Maryna Lukach
   October 2024

""" 
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid

#---------------------------------------------------------------------------------
#    base class for handling scores - this must be inherited to use it but provides
#    the definition for split_is_better and compute_score and does the plotting
#    name just identifies the score
#    min_max = 0 means a minimum score is best 1 means a maximum score is best
#---------------------------------------------------------------------------------
class Score(object):
    def __init__(self, name, min_max):
        self.scores=[]
        self.nclusters=[]
        self.name=name
        self.min_max=min_max

    def add_score(self, new_score, this_nclusters=None):
        self.scores.append(new_score)
        if this_nclusters!=None:
            self.nclusters.append(this_nclusters)
        
    #---------------------------------------------------------------------------------
    #  split_is_better
    #    input:  new_score
    #    return: True if new score is better, False otherwise. This is determined by whether 
    #            we are looking for a minimum, maximum
    #---------------------------------------------------------------------------------
    def split_is_better(self, new_score):
        better=False
        if len(self.scores)==0:
            # no previous scores so accept
            better=True
        else:
            if self.min_max==1:
                # looking for maximum score
                better = new_score > self.scores[-1]
            else:
                # looking for minimum score
                better = new_score < self.scores[-1]
        return better

    #---------------------------------------------------------------------------------
    #    Computes score for a given clusters
    #
    #    inputs:
    #       X     :  multidimension np array of data points [nsamples,nvars]
    #       labels:   Labels of the clusters
    #    Returns:
    #       score
    #---------------------------------------------------------------------------------
    def compute_score(self, X, labels, verbose=False):
        raise RuntimeError("Not implemented")

    # plot scores vs number of clusters or tries if we don't have nclusters
    def plot_scores(self, outdir):
        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        y=np.asarray(self.scores)
        if len(self.nclusters)>0:
            x=np.asarray(self.nclusters)
            xtitle='number of active clusters'
        else:
            x=np.arange(len(self.scores))+1
            xtitle='number of times'
        plt.scatter(x,y)
        if len(self.scores)>1:
            plt.plot(x,y)
        plt.title(self.name+' scores through the loops')
        plt.xlabel(xtitle)
        if len(x)>1:
            plt.xticks(x)
        plt.ylabel('score')
        plt.savefig(outdir+self.name+'scores_curve.png' , format='png',dpi=200)
        plt.close()
    
#---------------------------------------------------------------------------------
# class for handling Wemmert-Gancarski index
# looking for a max score
#---------------------------------------------------------------------------------
class WG_Score(Score):
    def __init__(self):
        super().__init__('WG',1)

    #    Computes the Wemmert-Gancarski index for a given clusters
    def compute_score(self, X, labels, verbose=False):

        clf = NearestCentroid(metric='euclidean')
        clf.fit(X, labels)

        unique_lab,n=np.unique(labels, return_counts=True)
        m = len(unique_lab)
        # check if the number of centers corresponds to the number of cluster labels
        if(m != len(clf.centroids_)):
            raise Exception('WG_Score.compute_score:Cannot calculate WG: The number of centers {} is not the same as the number of clusters {}'.format(len(centers),m))
        if verbose:
            print('WG_Score.compute_score: m=', m, 'clusters, counts n=', n, 'unique_labels',unique_lab)
        #size of data set
        N, d = X.shape
        if(N != len(labels)):
            if verbose:
                print ('N is {} and d is {}'.format(N,d))
                print ('len(labels) is {}'.format(len(labels)))
            raise Exception('Cannot calculate WG:The number of lines in the labels array is not the same as the number of lines in the data array!')

        J=[]
        for i in range(m):
            this_cluster_distances = distance.cdist(X[np.where(labels == i)], clf.centroids_, 'sqeuclidean')
            dist_to_other_centroids = np.delete(this_cluster_distances,[i],axis=1)
            min_dist_to_other_centroids = np.amin(dist_to_other_centroids,axis=1)
            ix=np.where(min_dist_to_other_centroids==0)
            if len(ix[0])>0:
                raise ValueError('min distance to other centroids should not be zero')
            R = np.divide(this_cluster_distances[:,i],min_dist_to_other_centroids)
            R_sum = sum(R)
            J.append(np.max([0,(1 - (1./n[i]) * R_sum)]))
        
        WG = (1./N) * sum(np.multiply(np.array(J),n))
        return WG
    #---------------------------------------------------------------------------------
    #  split_is_better
    #    input:  new_score
    #    return: As for Score:split_is_better() but we also check here if the score is ==0 
    #            because that should not be treated as better
    #---------------------------------------------------------------------------------
    def split_is_better(self, new_score):
        if new_score==0:
            return False
        else:
            return super().split_is_better(new_score)
    
#---------------------------------------------------------------------------------
# class for handling BIC score
# looking for a max score
#---------------------------------------------------------------------------------
class BIC_Score(Score):
    def __init__(self):
        super().__init__('BIC',1)
        
    def compute_score(self, X, labels, verbose=False):
        # get the centroids
        clf = NearestCentroid(metric='euclidean')
        clf.fit(X, labels)
        #number of clusters
        unique_lab,n=np.unique(labels, return_counts=True)
        m = len(unique_lab)

        #size of data set
        N, d = X.shape

        if(N != len(labels)):
            if verbose:
                print ("N is {} and d is {}".format(N,d))
                print ("len(labels) is {}".format(len(labels)))
            raise Exception('Cannot calculate BIC:The number of lines in the labels array is not the same as the number of lines in the data array!')
            return (None)

        #compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [clf.centroids_[i][:]], 'sqeuclidean')) for i in range(m)])

        const_term = 0.5 * m * np.log(N) * (d+1)

        BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

        return(BIC)

#---------------------------------------------------------------------------------
# class for handling calinski_harabaz score
# we should look for the maximum score
#---------------------------------------------------------------------------------
class CalinskiHarabaz_Score(Score):
    def __init__(self):
        super().__init__('Calinski Harabaz', 1)

    #    Computes the calinski harabaz score for given clusters
    def compute_score(self, X, labels, verbose=False):
        return calinski_harabasz_score(X, labels)

#---------------------------------------------------------------------------------
# class for handling silhouette score
# we should look for the maximum score but not allow <=0
#---------------------------------------------------------------------------------
class Silhouette_Score(Score):
    def __init__(self):
        super().__init__('Silhouette', 1)

    #    Computes the silhouette score for given clusters
    def compute_score(self, X, labels, verbose=False):
        return silhouette_score(X, labels)
    #---------------------------------------------------------------------------------
    #  split_is_better
    #    input:  new_score
    #    return: As for Score:split_is_better() but we also check here if the score is <=0 
    #            because that should not be treated as better
    #---------------------------------------------------------------------------------
    def split_is_better(self, new_score):
        if new_score<=0:
            return False
        else:
            return super().split_is_better(new_score)
        
#---------------------------------------------------------------------------------
# class for handling davies_bouldin score
# # we should look for the minimum score 
#---------------------------------------------------------------------------------
class DaviesBouldin_Score(Score):
    def __init__(self):
        super().__init__('Davies Bouldin', 0)
        
    #    Computes the davie bouldin score for given clusters
    def compute_score(self, X, labels, verbose=False):
        return davies_bouldin_score(X, labels)

def create_scorer(scorer_type):
    scorer=None
    if scorer_type=='WG':
        scorer=WG_Score()
    elif scorer_type=='CH':
        scorer=CalinskiHarabaz_Score()
    elif scorer_type=='SIL':
        scorer=Silhouette_Score()
    elif scorer_type=='DB':
        scorer=DaviesBouldin_Score()
    elif scorer_type=='BIC':
        scorer=BIC_Score()
    else:
        raise Exception('Unknown scorer type'+scorer_type)
    return scorer
    

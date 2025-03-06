import pdb
from scores import *
import numpy as np
import matplotlib.pyplot as plt

scorer_names=['WG','CH','SIL','DB','BIC']
nscorers=len(scorer_names)
scorers=[None]*nscorers
for s in range(nscorers):
    scorers[s]=create_scorer(scorer_names[s])
scores=np.zeros(nscorers)
# we will create nsamples data points with data split into 2 separated but close sets
# we will check scores for completely random labels
# we will then check scores for labels where a 3rd cluster has been made from the
# points between the samples
# we will then check scores for matching labels
nsamples=1000
nsamples_set0=300
nsamples_set1=nsamples-nsamples_set0
ix_set0=np.arange(nsamples_set0)
ix_set1=np.arange(nsamples_set1)+nsamples_set0
nvars=3
X=np.zeros((nsamples,nvars))
for v in range(nvars):
    X[ix_set0,v]=np.random.uniform(size=nsamples_set0)
    X[ix_set1,v]=np.random.uniform(size=nsamples_set1)+0.8
nrows=3
ncols=3
size=2
fig = plt.figure(figsize=(12,12*nrows/ncols))
# completely random labels- should get low score
#-----------------------------------------------
labels=np.random.randint(0,2,size=nsamples)
score_str=[]
halfway_s=np.ceil(round(nscorers/2))
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,labels)
    scorers[s].add_score(scores[s])
    if s==halfway_s:
        score_str.append('\n'+scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
    else:
        score_str.append(scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
score_str=', '.join(score_str)
print('random label scores:')
print('    '+'            '.join(scorer_names)+'\n', scores)
#assert(scores[2]<=0.002) # silhouette score likley to be <=0
# plot random data
ix0=np.where(labels==0)[0]
ix1=np.where(labels==1)[0]
ax=fig.add_subplot(nrows,ncols,1,projection='3d')
ax.set_title('random\n'+score_str, fontsize=10)
ax.set_xlabel('x',fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_zlabel('z', fontsize=10)
x=X[ix0,0]
y=X[ix0,1]
z=X[ix0,2]
ax.scatter(x,y,z, marker='.', s=size, color='r')
x=X[ix1,0]
y=X[ix1,1]
z=X[ix1,2]
ax.scatter(x,y,z, marker='.', s=size, color='b')

# partly matching labels
#-----------------------------------------------
good_labels=np.zeros(nsamples, int)
good_labels[ix_set1]=1
# switch some random labels around compared to good_labels
part_labels=np.copy(good_labels)
l0_labels=part_labels[ix_set0]
l1_labels=part_labels[ix_set1]
nto_shift_from0=int(nsamples_set0*0.1)
nto_shift_from1=int(nsamples_set1*0.1)
clf = NearestCentroid(metric='euclidean')
clf.fit(X, good_labels)
# for samples in cluster 0 find those closest to cluster 1 and vice versa
dist1_to0=np.sum(distance.cdist(X[ix_set1,:], [clf.centroids_[0][:]], 'sqeuclidean'),axis=1)
dist0_to1=np.sum(distance.cdist(X[ix_set0,:], [clf.centroids_[1][:]], 'sqeuclidean'),axis=1)
ix=np.argsort(dist1_to0)
l1_labels[ix[:nto_shift_from1]]=2
ix=np.argsort(dist0_to1)
l0_labels[ix[:nto_shift_from0]]=2
part_labels[ix_set0]=l0_labels
part_labels[ix_set1]=l1_labels
score_str=[]
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,part_labels)
    if s==halfway_s:
        score_str.append('\n'+scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
    else:
        score_str.append(scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
    assert(scorers[s].split_is_better(scores[s]))
    scorers[s].add_score(scores[s])
score_str=', '.join(score_str)
print('partly good labels scores:')
print('    '+'            '.join(scorer_names)+'\n', scores)
# plot partly good labels
ix0=np.where(part_labels==0)[0]
ix1=np.where(part_labels==1)[0]
ix2=np.where(part_labels==2)[0]
ax=fig.add_subplot(nrows,ncols,2,projection='3d')
ax.set_title('part\n'+score_str, fontsize=10)
ax.set_xlabel('x',fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_zlabel('z', fontsize=10)
x=X[ix0,0]
y=X[ix0,1]
z=X[ix0,2]
ax.scatter(x,y,z, marker='.', s=size, color='r')
x=X[ix1,0]
y=X[ix1,1]
z=X[ix1,2]
ax.scatter(x,y,z, marker='.', s=size, color='b')
x=X[ix2,0]
y=X[ix2,1]
z=X[ix2,2]
ax.scatter(x,y,z, marker='.', s=size, color='g')

# good labels
#-----------------------------------------------
score_str=[]
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,good_labels)
    if s==halfway_s:
        score_str.append('\n'+scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
    else:
        score_str.append(scorer_names[s]+'={sc:.2f} '.format(sc=scores[s]))
    assert(scorers[s].split_is_better(scores[s]))
    scorers[s].add_score(scores[s])
score_str=', '.join(score_str)
print('good label scores:')
print('    '+'            '.join(scorer_names)+'\n', scores)
# plot good labels
ax=fig.add_subplot(nrows,ncols,3,projection='3d')
ax.set_title('good labels\n'+score_str, fontsize=10)
ax.set_xlabel('x',fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_zlabel('z', fontsize=10)
x=X[ix_set0,0]
y=X[ix_set0,1]
z=X[ix_set0,2]
ax.scatter(x,y,z, marker='.', s=size, color='r')
x=X[ix_set1,0]
y=X[ix_set1,1]
z=X[ix_set1,2]
ax.scatter(x,y,z, marker='.', s=size, color='b')

xticks=np.arange(3)
for s in range(nscorers):
    ax=fig.add_subplot(nrows,ncols,4+s)
    ax.scatter(xticks,scorers[s].scores, marker='*')
    ax.set_xticks(xticks, labels=['random','part', 'good'])
    plt.title(scorer_names[s])

plt.gcf().subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()
plt.close()

# add some scores and check they plot properly
scorer_wg=create_scorer('WG')
scorer_wg.add_score(0.2,2)
scorer_wg.add_score(0.3,3)
scorer_wg.add_score(0.25,4)
print('check score plotting')
print('WG scores',scorer_wg.scores)
print('nclusters',scorer_wg.nclusters)
#do the same but with just one score
scorer_wg.plot_scores(./testing/')
scorer_wg=create_scorer('WG')
scorer_wg.add_score(0.3,3)
scorer_wg.plot_scores('./testing/second_')

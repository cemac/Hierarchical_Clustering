import pdb
from scores import *
from numpy import *

scorer_wg=create_scorer('WG')
scorer_ch=create_scorer('CH')
scorer_sil=create_scorer('SIL')
scorer_db=create_scorer('DB')
scorers=[scorer_wg, scorer_ch, scorer_sil, scorer_db]
nscorers=len(scorers)
scores=np.zeros(nscorers)
nsamples=1000
nsamples_set1=300
nvars=4
# completely random data - should get low score
print('test random data')
X=np.zeros((nsamples,nvars))
labels=np.asarray([np.random.randint(0,9) for i in range(nsamples)])
for v in range(nvars):
    X[:,v]=np.random.uniform(size=nsamples)
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,labels)
    scorers[s].add_score(scores[s])
print('WG {}, CH {}, SIL {}, DB {}'.format(scores[0], scores[1], scores[2], scores[3]))
assert(scores[2]<0) # silhouette score likley to be <0
print('test separated data')
labels[:nsamples_set1]=np.asarray([0 for i in range(nsamples_set1)])
labels[nsamples_set1:]=np.asarray([1 for i in range(nsamples-nsamples_set1)])
for v in range(nvars):
    X[:nsamples_set1,v]=np.random.uniform(size=nsamples_set1)
    X[nsamples_set1:,v]=np.random.uniform(size=nsamples-nsamples_set1)+5
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,labels)
    assert(scorers[s].split_is_better(scores[s]))
    scorers[s].add_score(scores[s])
print('WG {}, CH {}, SIL {}, DB {}'.format(scores[0], scores[1], scores[2], scores[3]))
print('test partly separated data')
for v in range(nvars):
    X[:nsamples_set1,v]=np.random.uniform(size=nsamples_set1)
    X[nsamples_set1:,v]=np.random.uniform(size=nsamples-nsamples_set1)+0.5
for s in range(nscorers):
    scores[s]=scorers[s].compute_score(X,labels)
    assert(scorers[s].split_is_better(scores[s])==False)
print('WG {}, CH {}, SIL {}, DB {}'.format(scores[0], scores[1], scores[2], scores[3]))
# add some scores and check they plot properly
scorer_wg=create_scorer('WG')
scorer_wg.add_score(0.2,2)
scorer_wg.add_score(0.3,3)
scorer_wg.add_score(0.25,4)
print('scores',scorer_wg.scores)
print('nclusters',scorer_wg.nclusters)
#do the same but with just one score
scorer_wg.plot_scores('./')
scorer_wg=create_scorer('WG')
scorer_wg.add_score(0.3,3)
scorer_wg.plot_scores('./second_')

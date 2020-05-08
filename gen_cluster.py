import collections
import itertools
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from functools import reduce
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import sparse
import tensorly as tl
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorly import tensor as tensor_dense
from tensorly import unfold as unfold_dense
from tensorly.contrib.sparse import tensor, unfold
from tensorly.tenalg import inner, mode_dot, multi_mode_dot
from tqdm.autonotebook import tqdm


def get_random_samples_higherorder(ll, p):
    assert p <= 1
    ll_len = list(map(lambda x: len(x), ll))
    n_prod = int(reduce(lambda a,b: a * b, ll_len) * p)
#     print('sample {} from {}'.format(n_prod, [len(i) for i in ll]))
    edge_indices = []
    for l in tqdm(ll):
        edge_indices.append(np.array(l)[np.random.randint(len(l),size=n_prod)].tolist())
    return edge_indices

def gen_cluster(block_sizes, dim_sizes, bipartite, ps, q):
    print('check valid')
    # check valid
    if not bipartite:
        for i in block_sizes:
            assert i[-1] == i[-2] 
        assert dim_sizes[-1] == dim_sizes[-2]

    
    assert np.array(block_sizes).shape[1] == len(dim_sizes)
    assert (np.array(block_sizes).sum(axis=0) <= np.array(dim_sizes)).all()

    assert 1 >= max(ps) >= min(ps) > q >= 0
    
#     print('gen blocks')
    # gen
    back_size = deepcopy(dim_sizes)
    for i in block_sizes:
        for ind, j in enumerate(i):
            back_size[ind] -= j

    cluster_id2idx = {}
    c = 0
    ind = -1
    # for ind, i in enumerate(block_sizes + [back_size]):
    for i in block_sizes:
        ind += 1
        cluster_id2idx[ind] = []
        for ind_j, j in enumerate(i):
            if ind_j == len(i) - 2:
                cluster_id2idx[ind].append(list(range(c, c + j)))
                c += j
            elif ind_j == len(i) - 1:
                if bipartite:
                    cluster_id2idx[ind].append(sample(range(dim_sizes[ind_j]), j))
                else:
                    cluster_id2idx[ind].append(deepcopy(cluster_id2idx[ind][-1]))
            else:
                cluster_id2idx[ind].append(sample(range(dim_sizes[ind_j]), j))
    
#     print('gen edges')
    

    all_edge_indices = []
    c = -1
    for i in cluster_id2idx:
        c += 1
        ll = cluster_id2idx[i]
        p = ps[c]
        all_edge_indices.append(get_random_samples_higherorder(ll, p))
        ll_neg = [ll[0]]
        ind = 1
        for j in ll[1:]:
            ll_neg.append(list(set(range(dim_sizes[ind])) - set(j)))
            ind += 1
#         all_edge_indices.append(get_random_samples_higherorder(ll_neg, q)) # no specifc negative sampling
    all_edge_indices.append(get_random_samples_higherorder(list(map(lambda x: list(range(x)), dim_sizes)), q))

    
    all_edge_indices_ = [[] for _ in range(len(dim_sizes))]
    for ei in tqdm(all_edge_indices):
        for ind, ei_ in enumerate(ei):
            all_edge_indices_[ind] += ei_
    
    return {
        'cluster_id2idx': cluster_id2idx, 
        'edge_indices': all_edge_indices,
        'tensor': gen_tensor(all_edge_indices_, dim_sizes)
    }

def gen_tensor(all_edge_indices, dim_sizes):

    data = [1.] * len(all_edge_indices[0])
#     print('gen sparse')
    s = sparse.COO(all_edge_indices, data, shape=tuple(dim_sizes)).astype(int)
#     print('gen tensor')
    return tensor(s)

import os
import pickle
import sys
sys.path.append('./fraudar')
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorly import unfold as unfold_dense
from tensorly.contrib.sparse import tensor, unfold

from gen_cluster import gen_cluster

from baselines import eval_dcube, eval_fraudar, eval_midas, eval_mzoom

def run_method(method_name, hashtag_tensor, user_map, input_folder, numberOfBlocks):
    if method_name == 'dcube':
        detected_blocks = eval_dcube(hashtag_tensor, user_map, input_folder, numberOfBlocks)
        detected_users = []
        for i, j in detected_blocks.items():
            for user in j:
                detected_users.append(j)

    elif method_name == 'mzoom':
        detected_blocks = eval_mzoom(hashtag_tensor, user_map, input_folder, numberOfBlocks)
        detected_users = []
        for i, j in detected_blocks.items():
            for user in j:
                detected_users.append(j)

    elif method_name == 'midas':
        list_of_users = eval_midas(hashtag_tensor, user_map, input_folder, numberOfBlocks)
        detected_users = [i[0] for i in list_of_users]

    elif method_name == 'fraudar':
        detected_users = eval_fraudar(hashtag_tensor, user_map, input_folder, numberOfBlocks)

    else:
        raise NotImplementedError

    return detected_users

if __name__ == "__main__":
    # Define where to save intermediate baseline files
    result_folder = "results/"
    input_folder = "input/"
    folders = [result_folder, input_folder]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    all_perm = {}
    for run in range(5):

        all_perm[run] = {}

        for num_user in [500, 5000, 50000]:
            num_hashtag = num_user * 2
            num_time = num_user // 2
            block_size = 40
            block_num = 5

            # Generate identity maps for hashtag_map and user_map
            hashtag_map = {str(i): i for i in range(num_hashtag)}
            user_map = {str(i): i for i in range(num_user)}

            q = .05

            config = {

                'hashtag': {
                    'block_sizes': [[block_size, block_size, block_size] for _ in range(block_num)], 
                    'dim_sizes': [num_time, num_user, num_hashtag], 
                    'bipartite': True, 
                    'ps': [.4,.35,.3, .25, .2], # [p] * block_num, 
                    'q': q
                }
            }

            for i in config:
                if not config[i]['bipartite']:
                    for j in config[i]['block_sizes']:
                        assert j[-1] == j[-2]

            cluster_info = {}

            for i in config:
                print(i)
                cluster_info[i] = gen_cluster(**config[i])
                print('\n')

            data_config = {
                'hashtag': {'axis': ['time', 'user', 'hashtag'], 'relweight': 1},
            }

            data_input = {i: cluster_info[i]['tensor'] for i in data_config}

            labs = np.array([0] * cluster_info['hashtag']['tensor'].shape[1])
            for i in cluster_info['hashtag']['cluster_id2idx']:
                labs[np.array(cluster_info['hashtag']['cluster_id2idx'][i][1])] = i + 1
            
            # Generate the true labels
            true_labels = deepcopy(labs)
            true_labels[true_labels > 0] = 1

            # Retrieve the time, user, hashtag tensor
            hashtag_tensor = data_input['hashtag']

            perm = {}

            for rseed in [100, 200, 300, 400]:
                perm[rseed] = {}

                print('seed', rseed)
                for r in [3, 10, 20]:
                    print('rank', r)
                    perm[rseed][r] = {}

                    for ind, method in enumerate(['dcube', 'mzoom', 'midas', 'fraudar']):
                        method = 'fraudar'
                        detected_users = run_method(method, hashtag_tensor, user_map, input_folder, block_num)
                        pred_labels = np.array([1 if i in detected_users else 0 for i in user_map.values()])
                        nmi = normalized_mutual_info_score(true_labels, pred_labels)
                        perm[rseed][r][method] = nmi
                        
                    print(perm)
            all_perm[run][num_user] = deepcopy(perm)
            pickle.dump(perm, open('perm_{}_{}.pkl'.format(run, num_user), 'wb'))
    pickle.dump(all_perm, open('all_perm.pkl', 'wb'))

import collections
import datetime
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from functools import reduce
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sparse
import tensorly as tl
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorly import tensor as tensor_dense
from tensorly import unfold as unfold_dense
from tensorly.contrib.sparse import tensor, unfold
from tensorly.tenalg import inner, mode_dot, multi_mode_dot
from tqdm import tqdm

from fraudar.greedy import logWeightedAveDegree

sys.path.append(os.path.abspath('../'))
warnings.filterwarnings("ignore")

# try running the serial version, not hadoop (not accelerated)
def eval_dcube(hashtag_tensor, user_map, input_folder, numberOfBlocks):
    print("start processing data with dcube")
    move_to_dcube = "./dcube/"
    move_from_dcube = "../"
    mzoom_output_file = "dcube_hashtag_result.txt"
    # same format as mzoom
    mzoom_input_file = "mzoom_hashtag_tensor.txt"

    nz = hashtag_tensor.nonzero()

    # build input files fitting format
    print("creating input files for dcube")
    with open(input_folder + mzoom_input_file, 'w') as f:
        for t, user, hashtag in zip(*nz):
            # time, user, hashtag, measure value (1)
            line = str(t)+","+str(user)+","+str(hashtag)+","+str(1)+"\n"
            f.write(line)
    f.close()

    os.chdir(move_to_dcube)
    print("initiating dcube process")
    dcube_start_time = time.time()
    input_path = move_from_dcube + input_folder + mzoom_input_file
    output_path = "output/"
    os.makedirs(output_path, exist_ok=True)

    print("removing the output folder's content")
    try:
        subprocess.call("rm -R ./{}*".format(output_path))
    except:
        pass
    dim = 3 # (has three dimensions time, user, hashtag)
    density_measure = "GEO"
    policy = "DENSITY"

    print('running dcube...')
    os.system("./run_single.sh {} {} {} {} {} {}".format(
        input_path, output_path, dim, density_measure, policy, numberOfBlocks))
    ts = time.time() - dcube_start_time

    below_threshold_blocks = dict()
    blocks = dict()
    block_count = 0
    users_threshold = 500	# max number of users to be added is 50000
    # load all users that were retrieved => blocks are too big
    for file in os.listdir(output_path):
        if ".tuples" in file:
            current_users = set()
            current_block = list()
            with open(output_path+file, 'r') as f:
                for line in f:
                    line = line.split(",")
                    uid = int(line[1])
                    tim = int(line[0])
                    hashtagid = int(line[2])
                    current_users.add(uid)
                    current_block.append((tim, uid, hashtagid))
                f.close()
            if len(current_users) <= users_threshold:
                final_uids = list()
                for map_id in current_users:
                    final_uids.append(user_map[str(map_id)])
                below_threshold_blocks[str(block_count)] = final_uids
                blocks[str(block_count)] = current_block
            print("from {} have found {} users for block {}".format(file, len(current_users), block_count))
            block_count += 1

    return below_threshold_blocks

def eval_mzoom(hashtag_tensor, user_map, input_folder, numberOfBlocks):
    print("start processing data with mzoom")
    move_to_mzoom = "./mzoom/"
    move_from_mzoom = "../"
    mzoom_output_file = "mzoom_hashtag_result.txt"
    mzoom_input_file = "mzoom_hashtag_tensor.txt"

    nz = hashtag_tensor.nonzero()

    # build input files fitting format
    print("creating input files for mzoom")
    with open(input_folder + mzoom_input_file, 'w') as f:
        for t, user, hashtag in zip(*nz):
            # time, user, hashtag, measure value (1)
            line = str(t)+","+str(user)+","+str(hashtag)+","+str(1)+"\n"
            f.write(line)
    f.close()

    os.chdir(move_to_mzoom)
    print("initiating mzoom process")
    mzoom_start_time = time.time()
    input_path = move_from_mzoom + input_folder + mzoom_input_file
    output_path = "output_mzoom/"
    print("emptying output folder from past results")
    try:
        subprocess.call("rm -R ./{}*".format(output_path))
    except:
        pass
    dim = 3 # (has three dimensions time, user, hashtag)
    density_measure = "GEO"
    os.system("./run_mzoom.sh {} {} {} {} {}".format(
        input_path, output_path, dim, density_measure, numberOfBlocks))
    ts = time.time() - mzoom_start_time
    
    below_threshold_blocks = dict()
    blocks = dict()
    block_count = 0
    users_threshold = 500	# max number of users to be added is 50000
    # load all users that were retrieved => blocks are too big
    for file in os.listdir(output_path):
        if ".tuples" in file:
            current_users = set()
            current_block = list()
            with open(output_path+file, 'r') as f:
                for line in f:
                    line = line.split(",")
                    uid = int(line[1])
                    tim = int(line[0])
                    hashtagid = int(line[2])
                    current_users.add(uid)
                    current_block.append((tim, uid, hashtagid))
                f.close()
            if len(current_users) <= users_threshold:
                final_uids = list()
                for map_id in current_users:
                    final_uids.append(user_map[str(map_id)])
                below_threshold_blocks[str(block_count)] = final_uids
                blocks[str(block_count)] = current_block
            print("from {} have found {} users for block {}".format(file, len(current_users), block_count))
            block_count += 1

    try:
        subprocess.call("rm -R ./output_mzoom/*")
    except:
        pass

    return below_threshold_blocks

def eval_midas(hashtag_tensor, user_map, input_folder, numberOfBlocks):
    # now run baselines for MIDAS
    print("start processing data with midas")
    midas_output = "midas.json"
    midas_mtx_input = "midas_input_uh.txt"
    midas_ten_input = "midas_input_uht.txt"

    nz = hashtag_tensor.nonzero()

    # build input files fitting format
    print("building user hashtag time matrix for midas")
    with open(input_folder + midas_ten_input, 'w') as f:
        for t, user, hashtag in zip(*nz):
            # time, user, hashtag, measure value (1)
            line = str(t) + "," + str(user) + "," + str(hashtag) + "," + str(1) + "\n"
            f.write(line)
    f.close()

    print("building user hashtag matrix input file for midas")
    # saving data for the user hashtag matrix for midas
    with open(input_folder + midas_mtx_input, 'w') as f:
        for user, hashtag in zip(nz[1], nz[2]):
            # user, hashtag
            line = str(user) + "," + str(hashtag) + "," + str(0) + "\n"
            f.write(line)
    f.close()

    print("running midas algorithm over the user hashtag tensor")
    move_to_midas = "./MIDAS"
    from_midas = "../"
    midas_mtx_output = "midas_matrix_results.txt"
    midas_ten_output = "midas_tensor_results.txt"
    os.chdir(move_to_midas)
    
    result_folder = 'results/'

    # note that the time recorded in this case includes loading data
    midas_start_uht = time.time()
    os.system("./midas -i {} -o {}".format(
        from_midas+input_folder+midas_ten_input,
        from_midas+result_folder+midas_ten_output))
    midas_uht_ts = time.time() - midas_start_uht

    midas_start_uh = time.time()
    os.system("./midas -i {} -o {}".format(
        from_midas+input_folder+midas_mtx_input,
        from_midas+result_folder+midas_mtx_output))
    midas_uh_ts = time.time() - midas_start_uh
    os.chdir(from_midas)
    
    # we will select the 1% highest portion of the score
    midas_edge_anomalies = np.zeros(len(nz[0]))
    total_user_anomaly_score = dict()
    avg_user_anomaly_score = dict()

    counter = 0
    # edges with the higher score at the most anomalous
    print("loading anomaly scores for midas tensor edges")
    with open(result_folder + midas_ten_output, 'r') as f:
        for line in f:
            # if want to compute the average anomaly score for all users
            # or maybe the total anomaly score for a user across the graph
            anomalyScore = float(line.replace("\n",""))
            midas_edge_anomalies[counter] = anomalyScore
            
            user_index = nz[1][counter]
            uid = user_map[str(user_index)]
            
            #hashtag_index = unfolded_hashtag_tensor[str(counter)][2]
            #hashtag = hashtag_map[str(hashtag_index)]
            if str(uid) not in avg_user_anomaly_score:
                avg_user_anomaly_score[str(uid)] = {"avgScore": 0, "count": 0}
            
            count = avg_user_anomaly_score[str(uid)]["count"]

            if avg_user_anomaly_score[str(uid)]["count"] > 1:
                avg_user_anomaly_score[str(uid)]["avgScore"] *= count
                avg_user_anomaly_score[str(uid)]["avgScore"] += anomalyScore
                avg_user_anomaly_score[str(uid)]["avgScore"] /= (count+1)
            else:
                avg_user_anomaly_score[str(uid)]["avgScore"] = anomalyScore
            
            avg_user_anomaly_score[str(uid)]["count"] += 1
            counter += 1

    threshold = np.percentile(midas_edge_anomalies, 0)
    # now retrieve the list of top most anomalous users
    dense_users = list()
    print("retrieving .1% most suspicious users found by MIDAS")
    for user in tqdm(avg_user_anomaly_score):
        if avg_user_anomaly_score[user]["avgScore"] > threshold:
            dense_users.append((user, avg_user_anomaly_score[user]["avgScore"]))

    return dense_users

def retrieve_coo(hashtag_matrix):
    # transforming to COO matrix enabling transformation
    coo_hashtag = scipy.sparse.coo_matrix(([1]*len(hashtag_matrix[0]),
        (hashtag_matrix[0], hashtag_matrix[1])),
        shape=(max(hashtag_matrix[0])+1, max(hashtag_matrix[1])+1))
    return coo_hashtag

def eval_fraudar(hashtag_tensor, user_map, input_folder, numberOfBlocks):    
    # Create user-hashtag matrix from hashtag tensor
    tmp = hashtag_tensor.sum(0).to_scipy_sparse()

    # coo_hashtag = retrieve_coo(hashtag_matrix)
    
    start_time = time.time()
    blocks = logWeightedAveDegree(tmp)
    ts = time.time() - start_time
    
    print("finished fraudar in {0:.2f} with average score: {1}".format(ts, blocks[1]))
    fraudar_output = "fraudar.json"

    # retrieve values for user ids and hashtags
    fraudar_block_hashtags = []
    fraudar_block_uids = []
    for _id in blocks[0][0]:
        fraudar_block_uids.append(user_map[str(_id)])

    return fraudar_block_uids

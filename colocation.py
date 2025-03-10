import random
import math
import json
from abc import ABC, abstractmethod
import time
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter


class Space(ABC):
    """
    Abstract base class defining the interface for a 'space'
    where nodes can live. Subclasses must implement:
      - sample_point()
      - distance(p1, p2)
    """

    @abstractmethod
    def sample_point(self):
        pass

    @abstractmethod
    def distance(self, p1, p2):
        pass

    def get_area(self):
        pass

    def get_max_dist(self):
        pass

class SphericalSpace(Space):
    """
    Sample points on (or near) the unit sphere.
    distance() returns geodesic distance (great-circle distance).
    """
    def sample_point(self):
        # Sample (x, y, z) from Normal(0, 1),
        # then normalize to lie on the unit sphere.
        while True:
            x = random.gauss(0, 1)
            y = random.gauss(0, 1)
            z = random.gauss(0, 1)
            r2 = x*x + y*y + z*z
            if r2 > 1e-12:
                scale = 1.0 / math.sqrt(r2)
                return (x*scale, y*scale, z*scale)

    def distance(self, p1, p2):
        # On a unit sphere, distance = arc length = arccos(dot(p1,p2))
        dotp = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
        # numerical safety clamp
        dotp = max(-1.0, min(1.0, dotp))
        return math.acos(dotp)
    
    def get_area(self):
        return 4*np.pi
    def get_max_dist(self):
        return 2*np.pi
    
def init_distance_matrix(positions, space):
    """
    Build the initial distance matrix for all node pairs.
    Returns a 2D list (or NumPy array) of shape (n, n).
    """
    n = len(positions)
    dist_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = space.distance(positions[i], positions[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def update_distance_matrix_for_node(dist_matrix, positions, space, moved_idx):
    """
    After node 'moved_idx' has changed its position,
    recalc only row [moved_idx] and column [moved_idx].
    """
    n = len(positions)
    i = moved_idx
    for j in range(n):
        if j == i:
            dist_matrix[i][j] = 0.0
        else:
            d = space.distance(positions[i], positions[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

def all_reward(dist_matrix, node_thresh):

    n = len(dist_matrix)
    scores = [0]*n
    radii = [0]*n
    
    # For each node i, find the gamma-th closest node distance
    # Then increment +1 for all nodes j within that radius.
    for i in range(n):
        # gather all distances from i -> others
        dist_list = sorted(dist_matrix[i])
        radius = dist_list[node_thresh-1]  # gamma-th (1-based => index node_thresh-1)
        radii[i] = radius
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                scores[j] += 1
    return scores, radii

def fixed_reward(dist_matrix, node_thresh):

    #pretty sure this doesn't do anything 
    reward=node_thresh
    n = len(dist_matrix)
    scores = [0]*n
    radii = [0]*n
    
    # For each node i, find the gamma-th closest node distance
    # Then increment +1 for all nodes j within that radius.
    for i in range(n):
        # gather all distances from i -> others
        dist_list = sorted(dist_matrix[i])
        radius = dist_list[node_thresh-1]  # gamma-th (1-based => index node_thresh-1)
        radii[i] = radius
        cnt = 0
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                cnt = cnt+1
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                scores[j] += reward/cnt
    return scores, radii

def prop_latency_reward(dist_matrix, node_thresh):

    #pretty sure this doesn't do anything 
    reward=node_thresh
    n = len(dist_matrix)
    scores = [0]*n
    radii = [0]*n
    
    # For each node i, find the gamma-th closest node distance
    # Then increment +1 for all nodes j within that radius.
    for i in range(n):
        tot = 0
        for j in range(n):
            if j==i:
                continue
            tot += 1/dist_matrix[i][j]
        for j in range(n):
            if j==i:
                continue
            scores[j] += reward*((1/dist_matrix[i][j])/tot)
    return scores, radii

def dropoff_reward(dist_matrix, node_thresh):

    #pretty sure this doesn't do anything 
    reward=node_thresh
    n = len(dist_matrix)
    scores = [0]*n
    radii = [0]*n
    
    # For each node i, find the gamma-th closest node distance
    # Then increment +1 for all nodes j within that radius.
    for i in range(n):
        # gather all distances from i -> others
        dist_list = sorted(dist_matrix[i])
        radius = dist_list[node_thresh-1]  # gamma-th (1-based => index node_thresh-1)
        radii[i] = radius
        cnt = 0
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                cnt = cnt+1
        if cnt>node_thresh:
            continue
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                scores[j] += reward/cnt
    return scores, radii

def declining_reward(dist_matrix, node_thresh):

    #pretty sure this doesn't do anything 
    reward=node_thresh
    n = len(dist_matrix)
    scores = [0]*n
    radii = [0]*n
    
    # For each node i, find the gamma-th closest node distance
    # Then increment +1 for all nodes j within that radius.
    for i in range(n):
        # gather all distances from i -> others
        dist_list = sorted(dist_matrix[i])
        radius = dist_list[node_thresh-1]  # gamma-th (1-based => index node_thresh-1)
        radii[i] = radius
        cnt = 0
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                cnt = cnt+1
        for j in range(n):
            if dist_matrix[i][j] <= radius:
                scores[j] += (reward/cnt)**2
    return scores, radii


def compute_all_scores(dist_matrix, node_thresh, score_func=all_reward):
    """
    Given a precomputed dist_matrix (n x n),
    compute the final score for each node using the 'gamma-th neighbor' rule.
    """

    scores,radii = score_func(dist_matrix,node_thresh)
    
    return scores, radii

def find_best_location_for_node(positions, space, node_thresh,
                                node_idx, dist_matrix,
                                num_candidates, reward_func):
    """
    Sample random points in the space to find the location that
    yields the highest final score for 'node_idx'.
    We do:
      - Save the old row & column for node_idx,
      - Update the distance matrix to reflect the candidate location,
      - Compute the node's score,
      - Revert the distance matrix,
      - Keep track of the best so far.
    """
    original_loc = positions[node_idx]
    # We'll store the old row & col so we can revert after each test
    n = len(positions)
    old_row = dist_matrix[node_idx][:]         # copy row i
    old_col = [dist_matrix[j][node_idx] for j in range(n)]

    # Score at the original location
    best_loc = original_loc
    best_score = compute_score_single_move_single_node(dist_matrix, node_thresh, node_idx,reward_func)

    for _ in range(num_candidates):
        candidate_loc = space.sample_point()
        # Temporarily place node_idx at candidate_loc
        positions[node_idx] = candidate_loc

        # Update row/col of dist_matrix for node_idx
        for j in range(n):
            if j == node_idx:
                dist_matrix[node_idx][j] = 0.0
            else:
                d = space.distance(candidate_loc, positions[j])
                dist_matrix[node_idx][j] = d
                dist_matrix[j][node_idx] = d

        # Compute node_idx's score with the updated matrix
        cand_score = compute_score_single_move_single_node(dist_matrix, node_thresh, node_idx,reward_func)

        # Check if better
        if cand_score > best_score:
            best_score = cand_score
            best_loc = candidate_loc

        # Revert the distance matrix to old row/col
        for j in range(n):
            dist_matrix[node_idx][j] = old_row[j]
            dist_matrix[j][node_idx] = old_col[j]

        # Revert node in positions
        positions[node_idx] = original_loc

    return best_loc, best_score


    # We'll define a helper that calculates just node_idx's score
def compute_score_single_move_single_node(dist_matrix, node_thresh, node_idx, reward_func):
    """
    Returns the final score of 'node_idx' by building
    the entire scoreboard from the existing distance matrix.
    """
    all_scores,radii = compute_all_scores(dist_matrix, node_thresh, reward_func)
    return all_scores[node_idx]
from statistics import mean, median

def med_mean_distance(dist_matrix):
    """
    Compute the mean of all pairwise distances in dist_matrix.
    Assumes dist_matrix is NxN, symmetric, and has zeroes on the diagonal.
    """
    n = len(dist_matrix)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(dist_matrix[i][j])
    if distances:
        return median(distances), mean(distances)
    return 0.0,0.0

import numpy as np
import math

def count_distances(dist_matrix, space):

    # Flatten the list and count frequencies
    frequency_dict = Counter(round(x*100/space.get_max_dist()) for row in dist_matrix for x in row)
    frequency_dict[0] -= len(dist_matrix) #remove the 0's on the diagonal
    divided_frequency = {key: value / 2 for key, value in frequency_dict.items()}
    return divided_frequency



def run_simulation(node_count=10, gamma=0.3, rounds_per_node=3, checkpoint_count=100, num_candidates=50, reward_func_name="all_reward",seed=0,experiment_name="general_data"):
    #if no seed provided, randomise
    if not seed:
        seed = round((time.time()*1000000)%1000000)
    
    random.seed(seed)
    space = SphericalSpace()  # or your chosen space

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    rounds_count = node_count*rounds_per_node
    checkpoint_rounds = max(int(rounds_count/checkpoint_count),1)
    all_checkpoints=[]
    positions = [space.sample_point() for _ in range(node_count)]
    dist_matrix = init_distance_matrix(positions, space)
    node_thresh = int(node_count*gamma)
    db = DBSCAN(eps=0.2, min_samples=10, metric='precomputed')
            
    reward_funcs = {
        "all_reward": all_reward,
        "fixed_reward":fixed_reward,
        "dropoff_reward":dropoff_reward,
        "declining_reward":declining_reward,
        "prop_latency_reward":prop_latency_reward
    }
    reward_func = reward_funcs[reward_func_name]
    output_json_file= experiment_name+"/"+reward_func_name+"_"+str(rounds_per_node)+"_"+str(node_count)+"_"+str(round(gamma,3))+"_"+str(round(seed))+".json"

    params = {
        "node_count": node_count,
        "gamma": gamma,
        "rounds_per_node":rounds_per_node,
        "reward_function":reward_func_name,
        "seed":seed
    }

    def record_checkpoint(round_idx, space):
        # compute scores for all nodes
        scores, radii = compute_all_scores(dist_matrix, node_thresh, reward_func)
        # store them
        median_distance,mean_distance  = med_mean_distance(dist_matrix)
        clusters = db.fit_predict(dist_matrix)
        unique_clusters = np.unique(clusters)
        cp = {
            "round_idx": round_idx,
            "positions": [list(p) for p in positions],  # convert tuples to lists
            "distances": count_distances(dist_matrix, space),
            "scores": scores,
            "radii": radii,
            "median_distance":median_distance,
            "mean_distance":mean_distance,
            "clustering": clusters.tolist(),
            "cluster_count": len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) -1
        }
        all_checkpoints.append(cp)
        return 1 if len(unique_clusters)==1 and -1 not in unique_clusters else 0

    # 1) Initialize positions and distance matrix

    record_checkpoint(0,space)
    for round_idx in range(1, rounds_count+1):
        selected_node = random.randint(0, node_count-1)
        # 2) Find best location for that node
        best_loc, best_score = find_best_location_for_node(
            positions, space, node_thresh,
            node_idx=selected_node,
            dist_matrix=dist_matrix,
            num_candidates=num_candidates,
            reward_func=reward_func
        )
        # 3) Permanently move the node
        positions[selected_node] = best_loc

        # 4) Update the distance matrix row/column for this node
        update_distance_matrix_for_node(dist_matrix, positions, space, selected_node)

        # Possibly record a checkpoint or print progress
        if (round_idx % checkpoint_rounds) == 0:
            stationary = record_checkpoint(round_idx,space)
            print(f"Round {round_idx} done;"+str((round_idx/checkpoint_rounds)*(100/checkpoint_count))+"% done")
            if stationary:
                break

    output_data = {
        "params": params,
        "checkpoints": all_checkpoints,
        "node_thresh": node_thresh
    }

    Path(__location__+"/"+experiment_name).mkdir(parents=True, exist_ok=True)
    
    with open(__location__ + "/" + output_json_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(__location__+"/files.json")
    with open(__location__+"/files.json", "r+") as f:
        files = json.load(f)
        files.append(output_json_file)
        f.seek(0)
        f.truncate()
        json.dump(files, f)

    print(f"Simulation complete. Data written to {output_json_file}")
    record_checkpoint(round_idx,space)

    print("Simulation complete.")
    # positions now final

if __name__ == "__main__":
    node_count=250
    gamma=0.15
    rounds_per_node=24
    checkpoint_count=50
    num_candidates=50
    seed=0
    experiment_name="friday_data"

    for i in range(5):
        run_simulation(
            node_count=node_count,
            gamma=gamma*(i+1),
            rounds_per_node=rounds_per_node,
            checkpoint_count=checkpoint_count,
            num_candidates=num_candidates,
            reward_func_name="all_reward",
            seed=seed,
            experiment_name=experiment_name
        )


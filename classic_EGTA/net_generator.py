import numpy as np
import random
import pickle
import os

def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)

def save_pkl(obj,path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result

def generate_networks(n):
    external_asset = np.random.uniform(low=0.0, high=40, size=n)
    Lk_r = list(np.random.randint(low=0, high=n-1, size=n))
    rand_bankrupts = np.random.randint(low=0, high=n)
    shock_id = random.sample(range(n), rand_bankrupts)
    for s in shock_id:
        external_asset[s] = 0

    adj = np.zeros((n, n))
    for i in range(len(Lk_r)):
        alist = list(range(0, n))
        alist.remove(i)
        Cdt_index = np.array(random.sample(alist, Lk_r[i]))
        for j in Cdt_index:
            homo_lb = [10, 20, 35]
            adj[i][j] = np.random.randint(0, random.choice(homo_lb))
    num_edges = np.array(Lk_r).sum()
    return external_asset, adj, num_edges

def generate_all_networks(num_instance, num_banks, save_path="./instances/networks.pkl"):
    networks = []
    for i in range(num_instance):
        external_asset, adj, num_edges = generate_networks(num_banks)
        net = {}
        net["external_asset"] = external_asset
        net["adj"] = adj
        net["num_edges"] = num_edges
        networks.append(net)

    save_pkl(networks, save_path)


if __name__ == "__main__":
    generate_all_networks(num_instance=1000,
                          num_banks=10,
                          save_path="./instances/networks_10banks_1000ins.pkl")
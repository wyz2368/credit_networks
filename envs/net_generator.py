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

def generate_networks(n,
                      ext_low,
                      ext_high,
                      default_frac=0.5):
    external_asset = np.random.uniform(low=ext_low, high=ext_high, size=n)
    # Lk_r = list(np.random.randint(low=int((n-1)/2), high=n-1, size=n))
    Lk_r = [n-1 for _ in range(n)] # Fully connected.
    rand_bankrupts = np.random.randint(low=0, high=int(default_frac * n))
    shock_id = random.sample(range(n), rand_bankrupts)
    for s in shock_id:
        external_asset[s] = 0

    adj = np.zeros((n, n))
    # print("LK:", Lk_r)
    for i in range(len(Lk_r)):
        alist = list(range(0, n))
        alist.remove(i)
        Cdt_index = np.array(random.sample(alist, Lk_r[i]))
        # print(Cdt_index)
        for j in Cdt_index:
            homo_lb = [10, 20, 35]
            adj[i][j] = np.random.randint(0, random.choice(homo_lb))
    num_edges = np.array(Lk_r).sum()
    return external_asset, adj, num_edges

def generate_all_networks(num_instance,
                          num_banks,
                          ext_low,
                          ext_high,
                          save_path="../instances/"):
    networks = []
    for i in range(num_instance):
        external_asset, adj, num_edges = generate_networks(num_banks,
                                                           ext_low,
                                                           ext_high)
        net = {}
        net["external_asset"] = external_asset
        net["adj"] = adj
        net["num_edges"] = num_edges
        networks.append(net)
        # break

    save_path += "networks_10banks_" + str(num_instance) + "ins_" + str(ext_low) + str(ext_high) + "ext_fc"
    save_pkl(networks, save_path + ".pkl")


if __name__ == "__main__":
    generate_all_networks(num_instance=1000,
                          num_banks=10,
                          ext_low=70,
                          ext_high=100,
                          save_path="../instances/")
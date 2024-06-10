import numpy as np
import random
import pickle
import os
from classic_EGTA.clearing import clearing

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
    Lk_r = list(np.random.randint(low=int(n-2), high=n-1, size=n))
    # Lk_r = [n-1 for _ in range(n)] # Fully connected.
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


def generate_networks_with_sink_nodes(n,
                                      ext_low,
                                      ext_high,
                                      rand_bankrupts=7):
    # Add a sink node.
    total_n = n + 1
    external_asset = np.random.uniform(low=ext_low, high=ext_high, size=total_n)
    # Lk_r = list(np.random.randint(low=1, high=n-1, size=n))
    Lk_r = [n-1 for _ in range(n)] # Fully connected.
    shock_id = random.sample(range(n), rand_bankrupts)
    for s in shock_id:
        external_asset[s] = ext_low

    adj = np.zeros((total_n, total_n))
    for i in range(len(Lk_r)):
        alist = list(range(0, n))
        alist.remove(i)
        Cdt_index = np.array(random.sample(alist, Lk_r[i]))
        for j in Cdt_index:
            homo_lb = [10, 20, 35]
            adj[i][j] = np.random.randint(0, random.choice(homo_lb))

        adj[i][total_n-1] = 20

    num_edges = np.array(Lk_r).sum()
    return external_asset, adj, num_edges

def generate_all_networks(num_instance,
                          num_banks,
                          ext_low,
                          ext_high,
                          save_path="../instances/tests/"):
    networks = []
    for i in range(num_instance):
        # external_asset, adj, num_edges = generate_networks_with_sink_nodes(num_banks,
        #                                                                    ext_low,
        #                                                                    ext_high)

        external_asset, adj, num_edges = generate_networks(num_banks,
                                                           ext_low,
                                                           ext_high)
        net = {}
        net["external_asset"] = external_asset
        net["adj"] = adj
        net["num_edges"] = num_edges
        networks.append(net)

        # print(external_asset)
        # print(adj)
        # print(external_asset + np.sum(adj, axis=0) - np.sum(adj, axis=1))
        #
        # ALPHA = BETA = 0.5
        #
        # payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(external_asset, adj, ALPHA, BETA)
        # print(payments_matrix)
        # print(Bank_equity)
        # print(Bank_asset)
        # print(SW_equity)
        # print(SW_asset)
        # print(Default_bank)
        # print(Recover_rate)
        #
        #
        # break

    save_path += "networks_10banks_" + str(num_instance) + "ins_" + str(ext_low) + str(ext_high) + "ext"
    save_pkl(networks, save_path + ".pkl")


if __name__ == "__main__":
    generate_all_networks(num_instance=1000,
                          num_banks=10,
                          ext_low=40,
                          ext_high=70,
                          save_path="../instances/merge/")
                          # save_path="../instances/prepayments/tests/")

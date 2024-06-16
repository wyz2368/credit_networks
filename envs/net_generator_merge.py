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


def split_number(number, k):
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Generate k-1 random points
    points = sorted(random.uniform(0, number) for _ in range(k - 1))

    # Calculate the k parts
    parts = [points[0]] + [points[i] - points[i - 1] for i in range(1, k - 1)] + [number - points[-1]]

    return parts

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

    # Generate payments matrix: adj[i, j] means i owes j.
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

    # Generate shareholding (Assume the number of players equals the number of banks)
    # (i, j): player i, bank j
    # TODO: need to guarantee each firm has a shareholder.
    total_assets = np.squeeze(np.sum(adj, axis=0)) + external_asset
    # print("total_assets", total_assets)
    shareholding = np.zeros((n, n))
    all_selected_players = set()
    for bank in range(n):
        selected_players = np.random.choice(a=list(range(n)), size=max(1, int(np.random.uniform(0,1)*0.5*n)), replace=False)
        all_selected_players.update(selected_players)
        # Randomly split the total asset.
        total_asset = total_assets[bank]
        if len(selected_players) == 1:
            shareholding[selected_players[0], bank] = total_asset
        else:
            split_parts = split_number(number=total_asset, k=len(selected_players))
            for i in range(len(selected_players)):
                shareholding[selected_players[i], bank] = split_parts[i]

    non_selected_players = set(range(n)).difference(all_selected_players)
    if len(non_selected_players) != 0:
        # print("Exist non-selected player:", non_selected_players)
        for player in non_selected_players:
            selected_bank = np.random.choice(a=list(range(n)), size=1)
            total_asset = total_assets[selected_bank]
            current_players = np.where(shareholding[:, selected_bank] > 0)[0]
            # print("current_players:", current_players, "selected_bank:", selected_bank)
            split_parts = split_number(number=total_asset, k=len(current_players)+1)
            for i in range(len(current_players) + 1):
                if i == len(current_players):
                    shareholding[player, selected_bank] = split_parts[i]
                    break
                shareholding[current_players[i], selected_bank] = split_parts[i]

    # Sample utility functional form parameters.
    params = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Generate a random value between 0 and 1
            value = np.random.choice([1, 1.1])
            # Add both (i, j) and (j, i) pairs to the dictionary with the same value
            params[i, j] = value
            params[j, i] = value

    return external_asset, adj, num_edges, shareholding, params


def generate_all_networks(num_instance,
                          num_banks,
                          ext_low,
                          ext_high,
                          save_path="../instances/tests/"):
    networks = []
    for i in range(num_instance):
        external_asset, adj, num_edges, shareholding, params = generate_networks(num_banks,
                                                                        ext_low,
                                                                        ext_high)
        net = {}
        net["external_asset"] = external_asset
        net["adj"] = adj
        net["num_edges"] = num_edges
        net["shareholding"] = shareholding
        net["params"] = params
        networks.append(net)

        # print("external_asset:", external_asset)
        # print("adj", adj)
        # print(external_asset + np.sum(adj, axis=0) - np.sum(adj, axis=1))

        # print("shareholding", shareholding)
        # print("params", params)

        ALPHA = BETA = 0.5

        # payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(external_asset, adj, ALPHA, BETA)
        # print(payments_matrix)
        # print(Bank_equity)
        # print(Bank_asset)
        # print(SW_equity)
        # print(SW_asset)
        # print(Default_bank)
        # print(Recover_rate)


        # break

    save_path += "networks_merge10banks_" + str(num_instance) + "ins_" + str(ext_low) + str(ext_high) + "ext"
    save_pkl(networks, save_path + ".pkl")


if __name__ == "__main__":
    generate_all_networks(num_instance=1000,
                          num_banks=10,
                          ext_low=40,
                          ext_high=100,
                          save_path="../instances/merge/")


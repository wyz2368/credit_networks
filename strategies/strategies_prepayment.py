import numpy as np
from itertools import combinations
import random

# random.seed(10)
# np.random.seed(10)

SINK_NODE = True

def subset_sum_with_ids(numbers, target):
    # Generate all possible subsets of (ID, number) pairs
    all_subsets = [combo for i in range(1, len(numbers) + 1) for combo in combinations(numbers, i)]

    # Filter subsets based on the sum condition
    valid_subsets = [subset for subset in all_subsets if sum(pair[1] for pair in subset) <= target]

    # Extract IDs from the valid subsets
    # all_ids = [[pair[0] for subset in valid_subsets for pair in subset]
    if len(valid_subsets) == 0:
        return []

    all_ids = []
    for subset in valid_subsets:
        selected_ids = []
        for pair in subset:
            selected_ids.append(pair[0])
        all_ids.append(selected_ids)

    return all_ids


def greedy_sampling(feasible_set, liabilities, external_asset):
    numbers_with_ID = []
    for id in feasible_set:
        numbers_with_ID.append((id, liabilities[id]))
    selected_ids = subset_sum_with_ids(numbers_with_ID, external_asset)
    if len(selected_ids) == 0:
        return []
    return random.choice(selected_ids)

def random_strategy(player, external_assets, adj_matrix):
    liabilities = np.squeeze(adj_matrix[player, :])
    feasible_set = []
    for i, liability in enumerate(liabilities):
        if SINK_NODE and i == len(external_assets)-1:
            continue
        if i == player:
            continue
        if external_assets[player] >= liability and liability > 0:
            feasible_set.append(i)

    return greedy_sampling(feasible_set, liabilities, external_assets[player])

def max_incoming_payment_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    liability = np.squeeze(adj_matrix[player, :])
    pairs = list(zip(range(len(external_assets)), incoming_payment))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    external_asset = external_assets[player]
    selected_banks = []
    for pair in sorted_pairs:
        if pair[1] == 0:
            break
        if liability[pair[0]] <= external_asset and liability[pair[0]] > 0:
            selected_banks.append(pair[0])
            external_asset -= liability[pair[0]]
            break

    return selected_banks

def max_incoming_payment_greedy_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    liability = np.squeeze(adj_matrix[player, :])
    pairs = list(zip(range(len(external_assets)), incoming_payment))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    external_asset = external_assets[player]
    selected_banks = []
    for pair in sorted_pairs:
        if pair[1] == 0:
            break
        if liability[pair[0]] <= external_asset and liability[pair[0]] > 0:
            selected_banks.append(pair[0])
            external_asset -= liability[pair[0]]

    return selected_banks

def heuristic_belief_strategy(player, external_assets, adj_matrix, mu=0, sigma=5, discount=0.1):
    noisy_external_assets = external_assets + np.random.normal(mu, sigma, len(external_assets))
    noisy_external_assets[noisy_external_assets < 0] = 0
    # print(noisy_external_assets)
    liabilities = np.squeeze(adj_matrix[player, :])
    incoming_payment = np.squeeze(adj_matrix[:, player])
    selected_banks = []
    filtered_liability = []

    for i in range(len(external_assets)):
        if i == player:
            continue
        if noisy_external_assets[i] + liabilities[i] >= incoming_payment[i] \
                and incoming_payment[i] > noisy_external_assets[i] + liabilities[i] * discount \
                and external_assets[player] >= liabilities[i] and incoming_payment[i] != 0 and liabilities[i] != 0:
            filtered_liability.append((i, liabilities[i]))

    sorted_pairs = sorted(filtered_liability, key=lambda x: x[1], reverse=True)
    for pair in sorted_pairs:
        if pair[1] == 0:
            break
        selected_banks.append(pair[0])
        break

    # return greedy_sampling(selected_banks, liabilities, external_assets[player])
    return selected_banks

def ultruism_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    liability = np.squeeze(adj_matrix[player, :])
    external_asset = external_assets[player]
    if np.sum(incoming_payment) + external_asset < np.sum(liability): # insolvent.
        selected_banks = []
        if SINK_NODE:
            pairs = list(zip(range(len(external_assets)-1), liability[:-1]))
        else:
            pairs = list(zip(range(len(external_assets)), liability))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        for pair in sorted_pairs:
            if pair[1] == 0:
                break
            if pair[1] <= external_asset:
                selected_banks.append(pair[0])
                external_asset -= pair[1]
        return selected_banks
    else:
        return []

def check_and_random_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    liability = np.squeeze(adj_matrix[player, :])
    external_asset = external_assets[player]
    if np.sum(incoming_payment) + external_asset < np.sum(liability):  # insolvent.
        return random_strategy(player, external_assets, adj_matrix)
    else:
        return []

def check_and_heuristic_belief_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    liability = np.squeeze(adj_matrix[player, :])
    external_asset = external_assets[player]
    if np.sum(incoming_payment) + external_asset < np.sum(liability):  # insolvent.
        return heuristic_belief_strategy(player, external_assets, adj_matrix)
    else:
        return []


def noop_strategy(player, external_assets, adj_matrix):
    return []

PREPAYMENT_STRATEGIES = {
    "noop_strategy":noop_strategy,
    # "random_strategy":random_strategy,
    # "max_incoming_payment_strategy":max_incoming_payment_strategy,
    # "max_incoming_payment_greedy_strategy":max_incoming_payment_greedy_strategy,
    "heuristic_belief_strategy":heuristic_belief_strategy,
    "ultruism_strategy":ultruism_strategy,
    # "check_and_random_strategy":check_and_random_strategy,
    "check_and_heuristic_belief_strategy":check_and_heuristic_belief_strategy
 }


### TEST ###

# external_assets = [2, 0, 5, 0]
#
# adj_m = np.array([
#     [0, 4, 4, 1],
#     [0, 0, 2, 0],
#     [5, 3, 0, 2],
#     [0, 0, 1, 0]])
#
# ALPHA = BETA = 0.5
#
# # ids = random_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# ids = max_incoming_payment_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# ids = max_incoming_payment_greedy_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# ids = heuristic_belief_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m, sigma=0)
# ids = ultruism_strategy(player=0, external_assets=external_assets, adj_matrix=adj_m)
# print(ids)




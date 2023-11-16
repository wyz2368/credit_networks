import numpy as np
from itertools import combinations
import random

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
        if i == player:
            continue
        if external_assets[player] >= liability and liability > 0:
            feasible_set.append(i)

    return greedy_sampling(feasible_set, liabilities, external_assets[player])

def max_incoming_payment_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    if external_assets[player] > incoming_payment:
        return [np.argmax(incoming_payment)]
    else:
        return []

def max_incoming_payment_greedy_strategy(player, external_assets, adj_matrix):
    incoming_payment = np.squeeze(adj_matrix[:, player])
    pairs = list(zip(range(len(external_assets)), incoming_payment))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    external_asset = external_assets[player]
    selected_banks = []
    for pair in sorted_pairs:
        if pair[1] == 0:
            break
        if pair[1] <= external_asset:
            selected_banks.append(pair[0])
            external_asset -= pair[1]
        else:
            break

    return selected_banks

def heuristic_belief_strategy(player, external_assets, adj_matrix, mu=0, sigma=10, discount=0.8):
    noisy_external_assets = external_assets + np.random.normal(mu, sigma, len(external_assets))
    noisy_external_assets[noisy_external_assets < 0] = 0
    print(noisy_external_assets)
    liabilities = np.squeeze(adj_matrix[player, :])
    incoming_payment = np.squeeze(adj_matrix[:, player])
    noisy_estimate = noisy_external_assets * discount + liabilities
    selected_banks = []

    for i, est in enumerate(noisy_estimate):
        if i == player:
            continue
        if est >= incoming_payment[i] \
                and incoming_payment[i] > noisy_external_assets[i] * discount \
                and external_assets[player] >= liabilities[i] and incoming_payment[i] != 0 and liabilities[i] != 0:
            selected_banks.append(i)

    return greedy_sampling(selected_banks, liabilities, external_assets[player])
    # return selected_banks


def noop_strategy(player, external_assets, adj_matrix):
    return []

PREPAYMENT_STRATEGIES = {
    "random_strategy":random_strategy,
    "noop_strategy":noop_strategy,
    "max_incoming_payment_strategy":max_incoming_payment_strategy,
    "max_incoming_payment_greedy_strategy":max_incoming_payment_greedy_strategy,
    "heuristic_belief_strategy":heuristic_belief_strategy
}


### TEST ###

# external_assets = [2, 0, 5, 0]
#
# adj_m = np.array([
#     [0, 4, 4, 4],
#     [0, 0, 2, 0],
#     [5, 3, 0, 2],
#     [0, 0, 1, 0]])
#
# ALPHA = BETA = 0.5
#
# # ids = random_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# # ids = max_incoming_payment_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# # ids = max_incoming_payment_greedy_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m)
# ids = heuristic_belief_strategy(player=2, external_assets=external_assets, adj_matrix=adj_m, sigma=0)
# print(ids)




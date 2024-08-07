import numpy as np

# random.seed(10)
# np.random.seed(10)

def relationship_function(params, i, j, x, y):
    # Complements or substitutes
    # x, y external assets
    # i, j two merged banks
    return (x + y) ** params[(i, j)]

def marginal_relationship_function(params, i, j, x, y):
    # Complements or substitutes
    # x, y external assets
    # i, j two merged banks
    return (x + y) ** params[(i, j)] - (x + y)

def noop_strategy(player, observation):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_firms = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_firms:
        action[i] = -1
    return action

def random_strategy(player, observation):
    # randomly select a player. If the selected player is itself, then do nothing.
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        selected_bank = np.random.choice(num_banks)
        if selected_bank == i:
            action[i] = -1
        else:
            action[i] = selected_bank
    return action


def always_accept_strategy(player, observation):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_firms = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_firms:
        action[i] = -3
    return action

    #-3: always accept, -2: no control of a firm, -1: votes for no.


def max_external_assets_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        combined_external_assets = []
        for j in range(num_banks):
            if i == j:
                combined_external_assets.append(-1)
            else:
                combined_external_assets.append(relationship_function(params=observation["params"],
                                                                      i=i,
                                                                      j=j,
                                                                      x=observation["external_asset"][i],
                                                                      y=noisy_external_assets[j]))

        action[i] = np.argmax(combined_external_assets)

    return action

def max_external_assets_increase_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        combined_external_assets = []
        for j in range(num_banks):
            if i == j:
                combined_external_assets.append(-1)
            else:
                combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                              i=i,
                                                                              j=j,
                                                                              x=observation["external_asset"][i],
                                                                              y=noisy_external_assets[j]))

        action[i] = np.argmax(combined_external_assets)

    return action




def reduced_liability_strategy(player, observation):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        if np.all(observation["adj"][i, :] == 0):
            action[i] = -1
        else:
            biggest_lender = np.argmax(observation["adj"][i, :])
            action[i] = biggest_lender

    return action

def shareholder_influence_strategy(player, observation):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        for j in range(num_banks):
            if i == j:
                continue
            else:
                player_shareholding_after_merger = observation["shareholding"][:, i] + observation["shareholding"][:, j]

            highest_holding = np.argmax(player_shareholding_after_merger)
            if highest_holding == player:
                action[i] = j
            else:
                action[i] = -1

    return action

def max_potential_bank_strategy(player, observation):
    """
    Find which bank has most 1.1 with others.
    """
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    params = observation["params"]
    sum_params = np.sum(params, axis=0)
    for i in holding_banks:
        max_potential_bank = np.argmax(sum_params)
        if i == max_potential_bank:
            # copy = np.copy(sum_params)
            # copy[i] = 0
            # second_highest = np.argmax(copy)
            # action[i] = second_highest
            action[i] = -1
        else:
            action[i] = -1
            # action[i] = max_potential_bank

    return action


def mea_noop_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)
        else:
            action[i] = -1

    return action

def rl_noop_strategy(player, observation):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender
        else:
            action[i] = -1

    return action


def mea_rl_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)
        else:
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender

    return action


def rl_mea_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender
        else:
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)

    return action

def mea_si_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)
        else:
            for j in range(num_banks):
                if i == j:
                    continue
                else:
                    player_shareholding_after_merger = observation["shareholding"][:, i] + observation["shareholding"][:, j]

                highest_holding = np.argmax(player_shareholding_after_merger)
                if highest_holding == player:
                    action[i] = j
                else:
                    action[i] = -1

    return action


def mea_mp_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    params = observation["params"]
    sum_params = np.sum(params, axis=0)

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)
        else:
            max_potential_bank = np.argmax(sum_params)
            if i == max_potential_bank:
                copy = np.copy(sum_params)
                copy[i] = 0
                second_highest = np.argmax(copy)
                action[i] = second_highest
            else:
                action[i] = max_potential_bank

    return action

def rl_si_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2
    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender
        else:
            for j in range(num_banks):
                if i == j:
                    continue
                else:
                    player_shareholding_after_merger = observation["shareholding"][:, i] + observation["shareholding"][:, j]

                highest_holding = np.argmax(player_shareholding_after_merger)
                if highest_holding == player:
                    action[i] = j
                else:
                    action[i] = -1

    return action

def rl_mp_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    params = observation["params"]
    sum_params = np.sum(params, axis=0)

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender
        else:
            max_potential_bank = np.argmax(sum_params)
            if i == max_potential_bank:
                copy = np.copy(sum_params)
                copy[i] = 0
                second_highest = np.argmax(copy)
                action[i] = second_highest
            else:
                action[i] = max_potential_bank

    return action

def mea_aa_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    params = observation["params"]
    sum_params = np.sum(params, axis=0)

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            combined_external_assets = []
            for j in range(num_banks):
                if i == j:
                    combined_external_assets.append(-1)
                else:
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                   i=i,
                                                                                   j=j,
                                                                                   x=observation["external_asset"][i],
                                                                                   y=noisy_external_assets[j]))

            action[i] = np.argmax(combined_external_assets)
        else:
            action[i] = -3
    return action


def rl_aa_strategy(player, observation, mu=0, sigma=5):
    num_banks = len(observation["external_asset"])
    noisy_external_assets = observation["external_asset"] + np.random.normal(mu, sigma, num_banks)
    noisy_external_assets[noisy_external_assets < 0] = 0

    shareholding = observation["shareholding"]
    holding_banks = np.where(shareholding[player, :] > 0)[0]
    action = np.zeros(num_banks) - 2

    params = observation["params"]
    sum_params = np.sum(params, axis=0)

    for i in holding_banks:
        incoming_payment = np.squeeze(observation["adj"][:, i])
        liability = np.squeeze(observation["adj"][i, :])
        external_asset = observation["external_asset"][i]
        if np.sum(incoming_payment) + external_asset < np.sum(liability):
            if np.all(observation["adj"][i, :] == 0):
                action[i] = -1
            else:
                biggest_lender = np.argmax(observation["adj"][i, :])
                action[i] = biggest_lender
        else:
            action[i] = -3

    return action


MERGE_STRATEGIES = {
    # "noop_strategy":noop_strategy,
    # "random_strategy":random_strategy,
    # "max_external_assets_strategy": max_external_assets_strategy,
    "always_accept_strategy": always_accept_strategy,
    # "max_external_assets_increase_strategy": max_external_assets_increase_strategy,
    "reduced_liability_strategy": reduced_liability_strategy,
    "shareholder_influence_strategy": shareholder_influence_strategy,
    # "max_potential_bank_strategy": max_potential_bank_strategy,
    # "mea_noop_strategy": mea_noop_strategy,
    # "rl_noop_strategy": rl_noop_strategy,
    "mea_rl_strategy": mea_rl_strategy,
    "rl_mea_strategy": rl_mea_strategy,
    # "mea_si_strategy": mea_si_strategy,
    # "rl_si_strategy": rl_si_strategy,
    # "rl_mp_strategy": rl_mp_strategy,
    # "mea_mp_strategy": mea_mp_strategy,
    "mea_aa_strategy": mea_aa_strategy,
    "rl_aa_strategy": rl_aa_strategy
}


### TEST ###

# external_assets = np.array([2, 0, 5, 0])
#
# adj_m = np.array([
#     [0, 4, 4, 1],
#     [0, 0, 2, 0],
#     [5, 3, 0, 2],
#     [0, 0, 1, 0]])
#
# params = np.array([
#     [0, 1, 1.1, 1],
#     [1, 0, 1.1, 1.1],
#     [1.1, 1.1, 0, 1],
#     [1, 1.1, 1, 0]])
#
# shareholding = np.array([
#                         [0, 1, 0, 1],
#                         [0, 0, 1, 0],
#                         [1, 1, 0, 1],
#                         [0, 0, 1, 0]])
#
# observation = {}
# observation["params"] = params
# observation["adj"] = adj_m
# observation["external_asset"] = external_assets
# observation["shareholding"] = shareholding

# ids = noop_strategy(player=4, observation=observation)
# ids = random_strategy(player=1, observation=observation)
# ids = max_external_assets_strategy(player=3, observation=observation)
# ids = reduced_liability_strategy(player=3, observation=observation)
# ids = shareholder_influence_strategy(player=2, observation=observation)
# ids = mea_noop_strategy(player=0, observation=observation)

# ids = max_potential_bank_strategy(player=0, observation=observation)
# print(ids)


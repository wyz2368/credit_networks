import numpy as np


def decision_rule(i, j, observation, gamma, merge_cost_factor=0.05):
    params = observation["params"]
    marginal_ext_asst = marginal_relationship_function(params=params,
                                                       i=i,
                                                       j=j,
                                                       x=observation["external_asset"][i],
                                                       y=observation["external_asset"][j])

    ext_asst = relationship_function(params=params,
                                       i=i,
                                       j=j,
                                       x=observation["external_asset"][i],
                                       y=observation["external_asset"][j])

    if marginal_ext_asst - merge_cost_factor * ext_asst > 0:
        incremental_ext_asst = True
    else:
        incremental_ext_asst = False

    claim_i = sum(observation["adj"][:, i]) - observation["adj"][j, i]
    claim_j = sum(observation["adj"][:, j]) - observation["adj"][i, j]

    liability_i = sum(observation["adj"][i, :]) - observation["adj"][i, j]
    liability_j = sum(observation["adj"][j, :]) - observation["adj"][j, i]

    if gamma * (claim_i + claim_j) + ext_asst - (liability_i + liability_j):
        solvency_after_merger = True
    else:
        solvency_after_merger = False

    return incremental_ext_asst and solvency_after_merger


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

"""
Start of strategy factory.
"""
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


def mea_facotry(gamma):
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
                    combined_external_assets.append(marginal_relationship_function(params=observation["params"],
                                                                                  i=i,
                                                                                  j=j,
                                                                                  x=observation["external_asset"][i],
                                                                                  y=noisy_external_assets[j]))


            best_j = np.argmax(combined_external_assets)

            decision = decision_rule(i=i,
                                     j=best_j,
                                     observation=observation,
                                     gamma=gamma)
            if decision:
                action[i] = best_j
            else:
                action[i] = -1

        return action

    return max_external_assets_strategy


def rl_factory(gamma):
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
                decision = decision_rule(i=i,
                                         j=biggest_lender,
                                         observation=observation,
                                         gamma=gamma)
                if decision:
                    action[i] = biggest_lender
                else:
                    action[i] = -1

        return action

    return reduced_liability_strategy


def si_factory(gamma):

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

                decision = decision_rule(i=i,
                                         j=j,
                                         observation=observation,
                                         gamma=gamma)

                highest_holding = np.argmax(player_shareholding_after_merger)
                if highest_holding == player and decision:
                    action[i] = j
                else:
                    action[i] = -1

        return action

    return shareholder_influence_strategy


def mp_facotry(gamma):
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
                action[i] = -1
            else:
                decision = decision_rule(i=i,
                                         j=max_potential_bank,
                                         observation=observation,
                                         gamma=gamma)
                if decision:
                    action[i] = max_potential_bank
                else:
                    action[i] = -1

        return action

    return max_potential_bank_strategy


MERGE_STRATEGIES = {
    # "noop_strategy":noop_strategy,
    # "random_strategy":random_strategy,
    # "always_accept_strategy": always_accept_strategy,
    # MAE
    "max_external_assets_strategy01": mea_facotry(gamma=0.1),
    "max_external_assets_strategy03": mea_facotry(gamma=0.3),
    "max_external_assets_strategy05": mea_facotry(gamma=0.5),
    "max_external_assets_strategy07": mea_facotry(gamma=0.7),
    "max_external_assets_strategy09": mea_facotry(gamma=0.9),
    # # RL
    # "reduced_liability_strategy01": rl_factory(gamma=0.1),
    # "reduced_liability_strategy03": rl_factory(gamma=0.3),
    # "reduced_liability_strategy05": rl_factory(gamma=0.5),
    # "reduced_liability_strategy07": rl_factory(gamma=0.7),
    # "reduced_liability_strategy09": rl_factory(gamma=0.9),
    # # SI
    # "shareholder_influence_strategy01": si_factory(gamma=0.1),
    # "shareholder_influence_strategy03": si_factory(gamma=0.3),
    # "shareholder_influence_strategy05": si_factory(gamma=0.5),
    # "shareholder_influence_strategy07": si_factory(gamma=0.7),
    # "shareholder_influence_strategy09": si_factory(gamma=0.9),
    # # MP
    "max_potential_bank_strategy01": mp_facotry(gamma=0.1),
    "max_potential_bank_strategy03": mp_facotry(gamma=0.3),
    "max_potential_bank_strategy05": mp_facotry(gamma=0.5),
    "max_potential_bank_strategy07": mp_facotry(gamma=0.7),
    "max_potential_bank_strategy09": mp_facotry(gamma=0.9),
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


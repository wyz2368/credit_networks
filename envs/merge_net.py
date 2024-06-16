"""
This is a PettingZoo implementation of a credit network for Merge & Aquisition.
"""

import functools
import random
from collections import defaultdict

from envs.net_generator_prepay import load_pkl

import numpy as np
from gymnasium.spaces import Discrete, Dict, Box

from pettingzoo import ParallelEnv
from classic_EGTA.clearing import clearing


def find_vote_with_highest_weight(votes, weights):
    if len(votes) != len(weights):
        raise ValueError("Votes and weights must be of the same length.")

    vote_weight_map = defaultdict(int)

    for vote, weight in zip(votes, weights):
        vote_weight_map[vote] += weight

    if -2 in vote_weight_map:
        del vote_weight_map[-2]

    if len(vote_weight_map) == 0:
        print("votes:", votes)
        raise ValueError("All -2 actions")

    max_weight_vote = max(vote_weight_map.items(), key=lambda item: item[1])[0]

    return max_weight_vote




class Merge_Net(ParallelEnv):
    """
    The metadata holds environment constants.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"name": "merge_net"}

    def __init__(self,
                 num_banks=10,
                 num_shareholders=10,
                 low_payment=0,
                 high_payment=100,
                 default_cost=0.5,
                 num_rounds=1,
                 merge_cost_factor=0.1,
                 control_bonus_factor=0.2,
                 params_decay=0.99, # decay of the parameter of functional form
                 utility_type="Bank_asset",
                 instance_path="./instances/merge/networks_10banks_1000ins.pkl",
                 sample_type="enum",
                 verbose=True):

        self.num_banks = num_banks
        self.current_num_bank = self.num_banks
        self.num_players = num_shareholders
        self.possible_agents = ["player_" + str(r) for r in range(num_shareholders)]
        self.low_payment = low_payment
        self.high_payment = high_payment

        # A mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Spaces.
        self._action_spaces = {agent: Discrete(self.num_banks + 2, start=-2) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Dict({"adj": Box(low=self.low_payment, high=self.high_payment, shape=(self.current_num_bank, self.current_num_bank)),
                     "external_asset": Box(low=self.low_payment, high=self.high_payment, shape=(1, self.current_num_bank)),
                     "params": Box(low=1, high=1.1, shape=(self.current_num_bank, self.current_num_bank)),
                     "shareholding": Box(low=0, high=1000, shape=(self.current_num_bank, self.current_num_bank))}) for agent in self.possible_agents
        }

        # Networks
        self.network_pool = load_pkl(path=instance_path)
        self.network_pool_iterator = iter(self.network_pool)
        self.default_cost = default_cost
        self.num_rounds = num_rounds
        self.utility_type = utility_type
        self.sample_type = sample_type
        self.merge_cost_factor = merge_cost_factor
        self.control_bonus_factor = control_bonus_factor
        self.params_decay = params_decay  #The parameter used for decaying the params when multiple banks merge.

        self.stats = []
        self.verbose = verbose


    def observation_space(self, agent):
        # The observation space include: external assets, liability, params, shareholding ratio.
        return Dict({"adj": Box(low=self.low_payment, high=self.high_payment, shape=(self.current_num_bank, self.current_num_bank)),
                     "external_asset": Box(low=self.low_payment, high=self.high_payment, shape=(1, self.current_num_bank)),
                     "params": Box(low=0.5, high=1.5, shape=(self.current_num_bank, self.current_num_bank)),
                     "shareholding": Box(low=0, high=1, shape=(self.num_players, self.current_num_bank))})

    def action_space(self, agent):
        return Discrete(self.num_banks + 2, start=-2)

    def sample_network(self):
        if self.sample_type == "random":
            return random.choice(self.network_pool)
        elif self.sample_type == "enum":
            return next(self.network_pool_iterator)
        else:
            raise ValueError("No such sample type.")


    def reset(self, seed=None, options=None):
        self.current_num_banks = self.num_banks #TODO: Check if this assignment correct
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.current_network = self.sample_network()

        observation = {"adj": self.current_network["adj"],
                       "external_asset": self.current_network["external_asset"],
                       "params": self.current_network["params"],
                       "shareholding": self.current_network["shareholding"]}

        observations = {agent: observation for agent in self.agents} # This assumes complete information and can be overloaded.
        infos = {agent: {} for agent in self.agents}
        self.state = observations # All players share the same state.

        return observations, infos


    def reset_netpool_iterator(self):
        self.network_pool_iterator = iter(self.network_pool)

    def update_default_cost(self, new_default_cost):
        self.default_cost = new_default_cost

    def get_stats(self):
        return self.stats

    def get_num_players(self):
        return self.num_players

    def get_current_num_banks(self):
        return self.current_num_banks

    def get_num_banks(self):
        return self.num_banks

    def reset_stats(self):
        self.stats = []


    def apply_actions(self, observation, actions):
        # actions is a list of actions:
        # An action is a list of length of banks. The index is the bank id of the shareholder,
        # the value is the merged bank id.

        current_external_assets = np.copy(observation["external_asset"])
        current_adj_matrix = np.copy(observation["adj"])
        current_params = np.copy(observation["params"])
        current_shareholding = np.copy(observation["shareholding"])
        all_votes = actions
        # print("all_votes", all_votes)

        # Get bank actions.
        bank_actions = [] # -1：No vote, -2: Not a shareholder
        for bank in range(self.current_num_banks):
            shares = current_shareholding[:, bank]
            votes = all_votes[:, bank]
            bank_actions.append(find_vote_with_highest_weight(votes=votes, weights=shares))

        # Get which pairs of banks to merge
        selected_banks = set()
        merged_banks = []
        removed_banks = []
        for i, action in enumerate(bank_actions):
            if action == -1 or i in selected_banks:
                continue
            for j in range(i+1, len(bank_actions)):
                if j in selected_banks:
                    continue
                if bank_actions[i] == j and bank_actions[j] == i:
                    merged_banks.append((i, j))
                    removed_banks.append(j)
                    selected_banks.add(i)
                    selected_banks.add(j)

        if self.verbose:
            if len(merged_banks) == 0:
                print("NO MERGE!#########################################################")
            else:
                print("MERGE!***********************************************************", merged_banks)

        refused_pairs = []

        # Rule of merging banks: reducing the dimension of adj_matrix, external assets, shareholding
        # by designating i as the merged bank and j as the removed bank.
        for i, j in merged_banks:
            # Adjust external assets.
            new_external_asset = self.relationship_function(params=current_params,
                                                                    i=i,
                                                                    j=j,
                                                                    x=current_external_assets[i],
                                                                    y=current_external_assets[j])

            total_asset_with_claims = np.sum(current_adj_matrix[:, i]) + current_external_assets[i] + np.sum(
                current_adj_matrix[:, j]) + current_external_assets[j]
            if new_external_asset - self.merge_cost_factor * total_asset_with_claims < 0:
                # Refuse the merger.
                removed_banks.remove(j)
                refused_pairs.append((i, j))
                if self.verbose:
                    print("REFUSE THE MERGER.", (i, j))
                continue
            else:
                current_external_assets[i] = new_external_asset
                current_external_assets[i] -= self.merge_cost_factor * total_asset_with_claims
                if self.verbose:
                    print("ACCEPT THE MERGER.", (i, j))


            # Adjust payment matrix.
            for bank in range(self.current_num_banks):
                if bank == i:
                    continue
                elif bank == j:
                    current_adj_matrix[bank, i] = 0
                    current_adj_matrix[i, bank] = 0
                else:
                    current_adj_matrix[bank, i] += current_adj_matrix[bank, j]
                    current_adj_matrix[i, bank] += current_adj_matrix[j, bank]


            # Adjust shareholding.
            current_shareholding[:, i] += current_shareholding[:, j]

            # Adjust params
            self.params_adjust(params=current_params,
                               i=i,
                               j=j,
                               x=current_external_assets[i],
                               y=current_external_assets[j])

        # Remove j
        # Adjust external assets.
        current_external_assets = np.delete(current_external_assets, removed_banks)
        current_adj_matrix = np.delete(current_adj_matrix, removed_banks, axis=0)
        current_adj_matrix = np.delete(current_adj_matrix, removed_banks, axis=1)
        current_shareholding = np.delete(current_shareholding, removed_banks, axis=1)
        current_params = np.delete(current_params, removed_banks, axis=0)
        current_params = np.delete(current_params, removed_banks, axis=1)

        self.current_num_banks -= len(merged_banks) + len(refused_pairs)

        new_observation = {}
        new_observation["adj"] = current_adj_matrix
        new_observation["external_asset"] = current_external_assets
        new_observation["params"] = current_params
        new_observation["shareholding"] = current_shareholding

        if len(merged_banks) == len(refused_pairs):
            early_termination = True
        else:
            early_termination = False

        return new_observation, early_termination


    def clearing(self, external_assets, adj_matrix):
        """
        Clearing the current system.
        """
        payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(
            external_assets, adj_matrix, self.default_cost, self.default_cost)

        stat = {
            "payments_matrix": payments_matrix,
            "Bank_equity": Bank_equity,
            "Bank_asset": Bank_asset,
            "SW_equity": SW_equity,
            "SW_asset": SW_asset,
            "Default_bank": Default_bank,
            "Recover_rate": Recover_rate
        }

        return stat

    def clearing_current_state(self):
        observation = self.state[self.agents[0]]
        adj_m = observation["adj"]
        external_assets = observation["external_asset"]

        self.clearing(external_assets=external_assets, adj_matrix=adj_m)

    def check_no_actions(self, actions): #TODO: check if no action is correct
        return np.isin(actions, [-1, -2]).all()


    def step(self, actions, pure=False):
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        observation = self.state[self.agents[0]]
        adj_m = observation["adj"]
        external_assets = observation["external_asset"]
        shareholding = observation["shareholding"]
        actions = np.array(actions)
        all_zeros = self.check_no_actions(actions)

        self.num_moves += 1
        env_truncation = self.num_moves > self.num_rounds
        truncations = {agent: env_truncation for agent in self.agents}

        # Pull up the apply_action earlier to create the early termination if the only merger was rejected.
        new_observation, early_termination = self.apply_actions(observation, actions)

        if all_zeros or env_truncation or early_termination:
            assert np.all(external_assets >= 0)
            stat = self.clearing(external_assets=external_assets, adj_matrix=adj_m)
            self.stats.append(stat)

            rewards = {}
            utilities_from_banks = self.split_utility(total_assets=stat[self.utility_type],
                                                      shareholding=shareholding)

            control_bonus = self.compute_control_bonus(total_assets=stat[self.utility_type],
                                                      shareholding=shareholding)

            for i, agent in enumerate(self.agents):
                rewards[agent] = utilities_from_banks[i] + control_bonus[i]

            if env_truncation or early_termination:
                terminations = {agent: True for agent in self.agents}
            else:
                terminations = {agent: False for agent in self.agents}

            observations = self.state

        else:

            rewards = {}
            for agent in self.agents:
                rewards[agent] = 0

            terminations = {agent: False for agent in self.agents}

            # current observation is set to be same as state. However, strategies only use
            # partial of full information and are restricted to local info.
            observations = {
                self.agents[i]: new_observation for i in range(len(self.agents))
            }
            self.state = observations


        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation or all_zeros or early_termination:
            self.agents = []

        return observations, rewards, terminations, truncations, infos


    ####### Special to merge net ############
    def relationship_function(self, params, i, j, x, y):
        # Complements or substitutes
        # x, y external assets
        # i, j two merged banks
        return (x + y) ** params[(i, j)]

    def params_adjust(self, params, i, j, x, y):
        """
        Adjust the params between the new merged bank and others based on a weighted-based rule.
        :return:
        """
        ratio_x = x / (x + y)
        ratio_y = y / (x + y)

        params[:, i] = ratio_x * params[:, i] + ratio_y * params[:, j]
        params[i, :] = params[:, i]
        params[i, i] = 0



    def split_utility(self, total_assets, shareholding):
        percentage_shareholding = shareholding / np.sum(shareholding, axis=0)
        utilities_over_banks = percentage_shareholding * total_assets
        utilities = np.squeeze(np.sum(utilities_over_banks, axis=1))
        return utilities

    def compute_control_bonus(self, total_assets, shareholding):
        control_bonus = np.zeros(self.num_players)
        biggest_shareholders = np.argmax(shareholding, axis=0)
        for i, holder in enumerate(biggest_shareholders):
            control_bonus[holder] += total_assets[i] * self.control_bonus_factor

        return control_bonus


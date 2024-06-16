"""
This is a PettingZoo implementation of a credit network for prepayment.
"""

import functools
import random

from envs.net_generator_prepay import load_pkl

import numpy as np
from gymnasium.spaces import MultiBinary, Box

from pettingzoo import ParallelEnv
# from pettingzoo.utils import parallel_to_aec
# from pettingzoo.utils import wrappers
from envs.customized_wrappers import Customized_AssertOutOfBoundsWrapper
from envs.customized_wrappers import Customized_OrderEnforcingWrapper
from envs.customized_wrappers import parallel_to_aec
from classic_EGTA.clearing import clearing


def create_env(env):
    env = raw_env(env)
    # this wrapper helps error handling for discrete action spaces
    env = Customized_AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = Customized_OrderEnforcingWrapper(env)
    return env

def raw_env(env):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_to_aec(env)
    return env


class Prepayment_Net(ParallelEnv):
    """
    The metadata holds environment constants.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"name": "prepayment_net"}

    def __init__(self,
                 num_banks=10,
                 low_payment=0,
                 high_payment=100,
                 default_cost=0.5,
                 num_rounds=1,
                 utility_type="Bank_asset",
                 instance_path="./instances/networks_10banks_1000ins.pkl",
                 sample_type="enum"):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """
        self.num_players = num_banks
        self.possible_agents = ["player_" + str(r) for r in range(num_banks)]
        self.low_payment = low_payment
        self.high_payment = high_payment

        # A mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # We can define the observation and action spaces here as attributes to be used in their corresponding methods
        # The observation shape is the shape of (adj_matrix + external asset).
        self._action_spaces = {agent: MultiBinary(self.num_players) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Box(low=low_payment, high=high_payment, shape=(self.num_players + 1, self.num_players)) for agent in self.possible_agents
        }

        self.network_pool = load_pkl(path=instance_path)
        self.network_pool_iterator = iter(self.network_pool)
        self.default_cost = default_cost
        self.num_rounds = num_rounds
        self.utility_type = utility_type
        self.sample_type = sample_type

        self.stats = []


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=self.low_payment, high=self.high_payment, shape=(self.num_players + 1, self.num_players))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiBinary(self.num_players)

    def sample_network(self):
        if self.sample_type == "random":
            return random.choice(self.network_pool)
        elif self.sample_type == "enum":
            return next(self.network_pool_iterator)
        else:
            raise ValueError("No such sample type.")

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.current_network = self.sample_network()
        # print("current networks:", self.current_network)
        adj_matrix_asset = np.vstack([self.current_network["adj"], self.current_network["external_asset"]])
        observations = {agent: adj_matrix_asset for agent in self.agents} # This assumes perfect information and can be overloaded.
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

    def reset_stats(self):
        self.stats = []


    def apply_actions(self, adj_matrix, external_assets, actions):
        new_external_assets = np.copy(external_assets)
        new_adj_matrix = np.copy(adj_matrix)
        for player_name, action in actions.items():
            player = self.agent_name_mapping[player_name]
            feasible_set = np.where(action == 1)[0]
            for j in feasible_set:
                new_external_assets[player] -= new_adj_matrix[player][j]
                new_external_assets[j] += new_adj_matrix[player][j]
                new_adj_matrix[player, j] = 0

        return new_external_assets, new_adj_matrix

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
        adj_matrix_asset = self.state[self.agents[0]]
        adj_m = adj_matrix_asset[:-1, :]
        external_assets = adj_matrix_asset[-1, :]

        self.clearing(external_assets=external_assets, adj_matrix=adj_m)

    def zero_actions(self, actions):
        all_zeros = True
        for action in actions.values():
            if np.any(action):
               all_zeros = False

        return all_zeros


    def step(self, actions, pure=False):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        adj_matrix_asset = self.state[self.agents[0]]
        adj_m = adj_matrix_asset[:-1, :]
        external_assets = adj_matrix_asset[-1,:]
        all_zeros = self.zero_actions(actions)

        self.num_moves += 1
        env_truncation = self.num_moves > self.num_rounds
        truncations = {agent: env_truncation for agent in self.agents}

        if all_zeros or env_truncation:
            assert np.all(external_assets >= 0)
            stat = self.clearing(external_assets=external_assets, adj_matrix=adj_m)
            self.stats.append(stat)

            rewards = {}
            for i, agent in enumerate(self.agents):
                rewards[agent] = stat[self.utility_type][i]

            if env_truncation:
                terminations = {agent: True for agent in self.agents}
            else:
                terminations = {agent: False for agent in self.agents}

            observations = self.state

        else:
            # if pure:
            #     print("before:\n", external_assets)
            #     print("before:\n", adj_m)
            new_external_assets, new_adj_matrix = self.apply_actions(adj_m, external_assets, actions)

            # if pure:
            #     print("after:\n",new_external_assets)
            #     print("after:\n",new_adj_matrix)

            rewards = {}
            for agent in self.agents:
                rewards[agent] = 0

            terminations = {agent: False for agent in self.agents}

            # current observation is set to be same as state. However, strategies only use
            # partial of full information and are restricted to local info.
            adj_matrix_asset = np.vstack([new_adj_matrix, new_external_assets])
            observations = {
                self.agents[i]: adj_matrix_asset for i in range(len(self.agents))
            }
            self.state = observations


        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation or all_zeros:
            self.agents = []

        return observations, rewards, terminations, truncations, infos



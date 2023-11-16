"""
The main solver for EGTA with player reduction.
"""
import numpy as np
from classic_EGTA.symmetric_utils import create_profiles, find_pure_equilibria
from classic_EGTA.player_reduction import deviation_preserve_reduction

def to_binary_action(num_actions, vanilla_action):
    binary_action = np.zeros(num_actions)
    if len(vanilla_action) == 0:
        return binary_action
    for i in vanilla_action:
        binary_action[i] = 1
    return binary_action

def average_stats(stats):
    average_result = {
            "Bank_equity": 0,
            "Bank_asset": 0,
            "SW_equity": 0,
            "SW_asset": 0,
    }

    num_instances = len(stats)

    for stat in stats:
        for key in average_result:
            average_result[key] += stat[key]

    for key in average_result:
        average_result[key] /= num_instances

    return average_result


def average_payoff_per_policy(average_result, profile):
    payoffs = []
    start = 0
    for count in profile:
        if count == 0:
            payoffs.append(None)
            continue
        payoffs.append(sum(average_result[start:start+count]) / count)
        start += count

    return payoffs


class EGTASolver:
    def __init__(self,
                 env,
                 sim_per_profile,
                 initial_strategies,
                 reduce_num_players):

        self.env = env
        self.num_players = self.env.num_players
        self.policies = initial_strategies
        self.num_policies = len(initial_strategies)
        self.reduce_num_players = reduce_num_players
        self.sim_per_profile = sim_per_profile

        self.profiles = create_profiles(num_players=self.num_players, num_strategies=len(self.policies))
        self.reduced_profiles = create_profiles(num_players=self.reduce_num_players, num_strategies=len(self.policies))
        self.reduced_game = self.init_reduced_game(self.reduced_profiles)
        self.equilibria = []

    def init_reduced_game(self, profiles):
        """
        Initialize a symmetric game with reduced number of players.
        :param profiles: a list of profiles.
        """
        payoffs = {}
        for profile in profiles:
            payoffs[tuple(profile)] = []
        return payoffs

    def assign_policies(self, profile):
        """
        Assigning strategies in a profile to agents.
        """
        assert sum(profile) == len(self.env.possible_agents)
        current_policies = {}
        j = 0
        for i, count in enumerate(profile):
            for _ in range(count):
                agent = self.env.possible_agents[j]
                current_policies[agent] = self.policies[i]
                j += 1

        return current_policies

    def simulation(self, profile):
        current_policies = self.assign_policies(profile)
        averaged_rewards = []
        for i in range(self.sim_per_profile):
            observations, infos = self.env.reset()
            total_rewards = []
            while self.env.agents:
                actions = {}
                for id, agent in enumerate(self.env.agents):
                    adj_matrix_asset = observations[agent]
                    adj_m = adj_matrix_asset[:-1, :]
                    external_assets = adj_matrix_asset[-1, :]
                    current_policy = current_policies[agent]
                    vanilla_action = current_policy(player=id,
                                                     external_assets=external_assets,
                                                     adj_matrix=adj_m)
                    binary_action = to_binary_action(self.num_players, vanilla_action)
                    actions[agent] = binary_action

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                total_rewards.append([rewards[agent] for agent in self.env.possible_agents])

            averaged_rewards.append(np.sum(total_rewards, axis=0))

        return np.mean(averaged_rewards, axis=0)

    def update_reduced_game_states(self):
        for reduced_profile in self.reduced_profiles:
            original_profiles = deviation_preserve_reduction(reduced_profile=reduced_profile,
                                                             num_players=self.num_players,
                                                             reduce_num_players=self.reduce_num_players)
            for original_profile in original_profiles:
                averaged_rewards = self.simulation(original_profile)
                payoffs = average_payoff_per_policy(average_result=averaged_rewards,
                                                    profile=original_profile)
                self.reduced_game[tuple(reduced_profile)] = payoffs


    def update_meta_strategies(self, meta_strategies):
        self.meta_strategies = meta_strategies

    def run(self):
        self.update_reduced_game_states()
        self.equilibria = find_pure_equilibria(self.reduced_game)
        return self.equilibria

    def get_stats(self):
        return self.env.get_stats()

    def get_reduced_game(self):
        return self.reduced_game













"""
The main solver for EGTA with player reduction.
"""


import numpy as np
from classic_EGTA.symmetric_utils import create_profiles, find_pure_equilibria, convert_to_full_game
from classic_EGTA.player_reduction import deviation_preserve_reduction
from classic_EGTA.nash_solvers.pygambit_solver import pygbt_solve_matrix_games
from classic_EGTA.nash_solvers.replicator_dynamics import replicator_dynamics
from classic_EGTA.clearing import save_pkl

def to_binary_action(num_actions, vanilla_action):
    """
    Convert a set of actions to a binary vector. [3, 5] -> [0,0,1,0,1]
    """
    binary_action = np.zeros(num_actions)
    if len(vanilla_action) == 0:
        return binary_action
    for i in vanilla_action:
        binary_action[i] = 1
    return binary_action

def average_stats(stats):
    """
    Average stats over instances. Each stat contains a clearing result of a stable network.
    """
    average_result = {
            "Bank_equity": 0,
            "Bank_asset": 0,
            "SW_equity": 0,
            "SW_asset": 0
    }

    num_instances = len(stats)
    for stat in stats:
        for key in average_result:
            average_result[key] += stat[key]

    for key in average_result:
        average_result[key] /= num_instances

    return average_result


def average_payoff_per_policy(average_result, original_profile, reduced_profile):
    """
    Compute the averaged payoff for each policy. This is achieved by averaging the payoffs
    of players playing the same policy.
    """
    payoffs = []
    start = 0
    for count in original_profile:
        if count == 0:
            payoffs.append(None)
            continue
        # This works because policies are assigned sequentially to the list of players.
        payoffs.append(sum(average_result[start:start+count]) / count)
        start += count

    return payoffs

def check_symmetric_strategies(profile):
    """
    Check whether all players play the same strategy in the profile.
    """
    for i in range(len(profile)-1):
        if not np.allclose(profile[i], profile[i+1]):
            return False
    return True

def sample_pure_profile(mixed_strategy, num_players):
    num_strategies = len(mixed_strategy)
    profile = np.zeros(num_strategies, dtype=int)
    sampled_strategies = np.random.choice(num_strategies, num_players, p=mixed_strategy)
    for strategy_id in sampled_strategies:
        profile[strategy_id] += 1

    return profile



class EGTASolver:
    def __init__(self,
                 env,
                 sim_per_profile,
                 initial_strategies,
                 reduce_num_players,
                 checkpoint_dir):

        self.env = env
        self.num_players = self.env.get_num_players()
        self.policies = initial_strategies
        self.num_policies = len(initial_strategies)
        self.reduce_num_players = reduce_num_players
        self.sim_per_profile = sim_per_profile
        self.checkpoint_dir = checkpoint_dir

        self.reduced_profiles = create_profiles(num_players=self.reduce_num_players, num_strategies=len(self.policies))
        self.reduced_game = self.init_reduced_game(self.reduced_profiles)
        self.reduced_game_stats = self.init_reduced_game(self.reduced_profiles)
        self.summary_stats = self.init_reduced_game_stats(self.reduced_profiles)
        self.equilibria = []

        # print("Begin Running EGTA.")
        # print("Initial Reduced Profiles:", self.reduced_profiles, len(self.reduced_profiles))
        # print("Initial Reduced Game:", self.reduced_game)

    def init_reduced_game(self, profiles):
        """
        Initialize a symmetric game with reduced number of players.
        :param profiles: a list of profiles.
        """
        payoffs = {}
        for profile in profiles:
            payoffs[tuple(profile)] = []
        return payoffs

    def init_reduced_game_stats(self, profiles):
        """
        Initialize a symmetric game with reduced number of players.
        :param profiles: a list of profiles.
        """
        payoffs = {}
        measures = ["total_assets", "total_equity", "num_default", "recover_rate"]
        types = ["benefit", "neural", "harm"]
        for profile in profiles:
            payoffs[tuple(profile)] = {}
            for measure in measures:
                payoffs[tuple(profile)][measure] = {}
                for type in types:
                    payoffs[tuple(profile)][measure][type] = 0
        return payoffs

    def assign_policies(self, profile):
        """
        Assigning policies in a profile to players.
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
        """
        Simulate the payoffs for strategies in a pure-strategy profile (how many players play each strategy).
        """
        current_policies = self.assign_policies(profile)
        averaged_rewards = []
        # When network instance is sampled from the generator, env.reset will sample a new instance.
        # When sample_type is "enum", it returns an instance given by an iterator.
        # When sample_type is "random", it returns an instance randomly sampled from the generator.
        for i in range(self.sim_per_profile):
            observations, infos = self.env.reset()
            traj_rewards = []
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

                # print("actions1:", actions)

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                traj_rewards.append([rewards[agent] for agent in self.env.possible_agents])

            # Sum of immediate rewards, not discounted.
            averaged_rewards.append(np.sum(traj_rewards, axis=0))
            # print("%%averaged_rewards:", averaged_rewards)

        # Average over instances.
        return np.mean(averaged_rewards, axis=0)

    def update_reduced_game_states(self):
        """
        Simulate all profiles in the reduced game.
        """
        # print("In update_reduced_game_states")
        for reduced_profile in self.reduced_profiles:
            self.reset_stats()
            # Compute the corresponding profile in the original game.
            # original_profiles is a list of profiles, one for each deviating strategy.
            original_profiles = deviation_preserve_reduction(reduced_profile=reduced_profile,
                                                             num_players=self.num_players,
                                                             reduce_num_players=self.reduce_num_players)
            # print("reduced_profile:", reduced_profile)
            # print("original_profiles:", original_profiles)

            # print("For each original profile:")
            payoffs_over_original_profiles = []
            for original_profile in original_profiles:
                # print("original_profile:", original_profile, "reduced:", reduced_profile)
                averaged_rewards = self.simulation(original_profile)
                # print("averaged_rewards:", averaged_rewards)
                payoffs = average_payoff_per_policy(average_result=averaged_rewards,
                                                    original_profile=original_profile,
                                                    reduced_profile=reduced_profile)
                payoffs_over_original_profiles.append(payoffs)
                # print("payoffs:", payoffs)

            j = 0 # index for payoffs_over_original_profiles
            for i, count in enumerate(reduced_profile):
                if count == 0:
                    self.reduced_game[tuple(reduced_profile)].append(None)
                else:
                    strategy_payoff = payoffs_over_original_profiles[j][i]
                    self.reduced_game[tuple(reduced_profile)].append(strategy_payoff)
                    j += 1

            # print("self.reduced_game[tuple(reduced_profile)]:", self.reduced_game[tuple(reduced_profile)])

            stats = self.get_stats()
            self.reduced_game_stats[tuple(reduced_profile)].append(stats)
            save_pkl(self.reduced_game_stats, path=self.checkpoint_dir + "/reduced_game_stats.pkl")

            # print("---------")

    def regret_in_full_game(self, profile, num_iterations):
        assert check_symmetric_strategies(profile)
        mixed_strategy = profile[0]
        num_strategies = len(mixed_strategy)
        cumulative_payoffs = np.zeros(num_strategies)
        for iter in range(num_iterations):
            pure_strategy_profile = sample_pure_profile(mixed_strategy=mixed_strategy,
                                                        num_players=self.num_players)
            payoffs = self.simulation(pure_strategy_profile)
            cumulative_payoffs += payoffs

        expected_payoffs = np.sum(cumulative_payoffs * mixed_strategy)
        regrets = cumulative_payoffs - expected_payoffs
        return max(regrets)


    def run(self):
        self.update_reduced_game_states()
        self.pure_equilibria = find_pure_equilibria(self.reduced_game)
        full_meta_games = convert_to_full_game(num_players=self.reduce_num_players,
                                             num_policies=self.num_policies,
                                             symmetric_game=self.reduced_game)
        # self.equilibria = pygbt_solve_matrix_games(full_meta_games, method="simpdiv", mode="all")
        self.equilibria = [replicator_dynamics(full_meta_games)]


        return self.pure_equilibria, self.equilibria

    def get_stats(self):
        return self.env.get_stats()

    def reset_stats(self):
        return self.env.reset_stats()

    def get_reduced_game(self):
        return self.reduced_game

    def observe(self):
        pass











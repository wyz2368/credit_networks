"""
The main solver for EGTA with player reduction.
"""

import numpy as np
from classic_EGTA.symmetric_utils import create_profiles, find_pure_equilibria, convert_to_full_game
from classic_EGTA.player_reduction import deviation_preserve_reduction
from classic_EGTA.nash_solvers.pygambit_solver import pygbt_solve_matrix_games
from classic_EGTA.nash_solvers.replicator_dynamics import replicator_dynamics
from envs.net_generator_prepay import save_pkl
from classic_EGTA.evaluation import evaluate

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


def average_payoff_per_policy(average_result, original_profile):
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


def is_pure_symmetric(profile, num_players):
    if len(np.where(np.array(profile) == num_players)[0]) != 0:
        return True
    return False


class EGTASolver:
    def __init__(self,
                 env,
                 sim_per_profile,
                 initial_strategies,
                 reduce_num_players,
                 checkpoint_dir):

        self.env = env
        self.num_players = self.env.get_num_players() #TODO: Check this consistent with M/A
        self.policies = initial_strategies
        self.num_policies = len(initial_strategies)
        self.reduce_num_players = reduce_num_players
        self.sim_per_profile = sim_per_profile
        self.checkpoint_dir = checkpoint_dir

        self.reduced_profiles = create_profiles(num_players=self.reduce_num_players, num_strategies=len(self.policies))
        self.reduced_game = self.init_reduced_game(self.reduced_profiles)
        self.equilibria = []

        # For evaluation:
        full_game_profiles = create_profiles(num_players=self.num_players,
                                             num_strategies=len(self.policies))

        self.full_symmetric_game = {}
        for profile in full_game_profiles:
            self.full_symmetric_game[tuple(profile)] = []
        self.game_stats = self.init_reduced_game(full_game_profiles)

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
        measures = ["Bank_asset", "Bank_equity", "Default_bank", "Recover_rate"]
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

    def simulation(self, profile, binary_action_flag=False):
        """
        Simulate the payoffs for strategies in a pure-strategy profile (how many players play each strategy).
        """
        current_policies = self.assign_policies(profile)
        averaged_rewards = []
        self.env.reset_netpool_iterator()

        # When network instance is sampled from the generator, env.reset will sample a new instance.
        # When sample_type is "enum", it returns an instance given by an iterator.
        # When sample_type is "random", it returns an instance randomly sampled from the generator.
        non_empty_actions = []
        for i in range(self.sim_per_profile):
            # print("------ Sim {} -------".format(i))
            observations, infos = self.env.reset()
            # print("params:", observations["player_0"]["params"])
            traj_rewards = []
            k = 0
            while self.env.agents:
                # print(k, "observation", observations["player_1"]["shareholding"])
                k += 1
                if binary_action_flag:
                    actions = {}
                else:
                    actions = []
                for id, agent in enumerate(self.env.agents):
                    if binary_action_flag:
                        adj_matrix_asset = observations[agent]
                        adj_m = adj_matrix_asset[:-1, :]
                        external_assets = adj_matrix_asset[-1, :]
                        current_policy = current_policies[agent]
                        vanilla_action = current_policy(player=id,
                                                         external_assets=external_assets,
                                                         adj_matrix=adj_m)

                        binary_action = to_binary_action(self.num_players, vanilla_action)
                        actions[agent] = binary_action
                    else:
                        observation = observations[agent]
                        current_policy = current_policies[agent]
                        vanilla_action = current_policy(player=id,
                                                        observation=observation) # Raw int [1,2,3,4,5]
                        actions.append(vanilla_action)

                # print("Observation:", observations["player_0"])
                # print("actions:", actions)

                observations, rewards, terminations, truncations, infos = self.env.step(actions, is_pure_symmetric(profile, 10))
                traj_rewards.append([rewards[agent] for agent in self.env.possible_agents])
            # Sum of immediate rewards, not discounted.
            if k > 1:
                non_empty_actions.append(i)
            averaged_rewards.append(np.sum(traj_rewards, axis=0))

        # Average over instances.
        # print(averaged_rewards)
        # save_pkl(averaged_rewards, path=self.checkpoint_dir + "/averaged_rewards.pkl")
        # save_pkl(non_empty_actions, path=self.checkpoint_dir + "/valid_ins_idx.pkl")
        return np.mean(averaged_rewards, axis=0)

    def update_reduced_game_states(self):
        """
        Simulate all profiles in the reduced game.
        """
        for reduced_profile in self.reduced_profiles:
            # print("==================================================")
            # print("Current Reduced Profile:", reduced_profile)
            # Compute the corresponding profile in the original game.
            # original_profiles is a list of profiles, one for each deviating strategy.
            original_profiles = deviation_preserve_reduction(reduced_profile=reduced_profile,
                                                             num_players=self.num_players,
                                                             reduce_num_players=self.reduce_num_players)

            payoffs_over_original_profiles = []
            for original_profile in original_profiles:
                # print("---------------")
                self.reset_stats()
                averaged_rewards = self.simulation(original_profile)

                # print("rewards:", averaged_rewards)

                payoffs = average_payoff_per_policy(average_result=averaged_rewards,
                                                    original_profile=original_profile)
                payoffs_over_original_profiles.append(payoffs)


                # For evaluation
                self.full_symmetric_game[tuple(original_profile)] = payoffs
                stats = self.get_stats()
                self.game_stats[tuple(original_profile)].append(stats.copy())


            j = 0 # index for payoffs_over_original_profiles
            for i, count in enumerate(reduced_profile):
                if count == 0:
                    self.reduced_game[tuple(reduced_profile)].append(None)
                else:
                    strategy_payoff = payoffs_over_original_profiles[j][i]
                    self.reduced_game[tuple(reduced_profile)].append(strategy_payoff)
                    j += 1

        save_pkl(self.reduced_game, path=self.checkpoint_dir + "/reduced_game.pkl")


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

    def observe(self, logger):
        logger.info("==== Begin Evaluation ====")
        if len(self.pure_equilibria) == 0:
            logger.info("==== No pure-strategy NE ====")

        ne_set = []
        for ne in self.pure_equilibria:
            if not is_pure_symmetric(ne, self.reduce_num_players):
                continue
            mixed_strategy = np.copy(ne)
            mixed_strategy[mixed_strategy > 0] = 1
            ne_set.append(mixed_strategy)

        ne_set.append(self.equilibria[0][0])

        evaluate(self, ne_set, logger)















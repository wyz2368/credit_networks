import numpy as np
from classic_EGTA.symmetric_utils import create_profiles

measures = ["Bank_asset", "Bank_equity", "Default_bank", "Recover_rate"]
types = ["benefit", "neural", "harm"]

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

def evaluate(solver, ne_set, logger):
    """
    The main entry for evaluating the output of an EGTA solver.
    """
    full_symmetric_game, social_optimum, optimum_profile, game_stats = get_full_game(solver)
    for ne in ne_set:
        ne_expected_payoff = mixed_strategy_expected_payoffs(full_symmetric_game, ne)
        poa = price_of_anarchy(social_optimum, ne_expected_payoff * solver.num_players)
        summary_stats = num_beneficial_banks(solver, game_stats, ne)
        logger.info("The current NE is {}".format(ne))
        logger.info("The expected payoff per player is {}".format(ne_expected_payoff))
        logger.info("The social optimum profile is {}".format(optimum_profile))
        logger.info("The social optimum is {}".format(social_optimum))
        logger.info("The PoA is {}".format(poa))
        logger.info("The summary stats: {}".format(summary_stats))
        logger.info("-------------------------")


def get_full_game(solver):
    """
    Simulate the full game with a compact symmetric-game representation.
    """
    full_game_profiles = create_profiles(num_players=solver.num_players,
                                         num_strategies=len(solver.policies))

    full_symmetric_game = {}
    for profile in full_game_profiles:
        full_symmetric_game[tuple(profile)] = []

    game_stats = solver.init_reduced_game(full_game_profiles)

    social_optimum = -np.inf
    optimum_profile = None
    for profile in full_game_profiles:
        solver.reset_stats()
        averaged_rewards = solver.simulation(profile)
        payoffs = average_payoff_per_policy(average_result=averaged_rewards,
                                            original_profile=profile)

        full_symmetric_game[tuple(profile)] = payoffs
        sw = np.sum(averaged_rewards)
        if sw > social_optimum:
            social_optimum = sw
            optimum_profile = profile

        stats = solver.get_stats()
        game_stats[tuple(profile)].append(stats.copy())


    return full_symmetric_game, social_optimum, optimum_profile, game_stats


def price_of_anarchy(social_optimum, social_welfare_ne):
    """Compute price of anarchy approximated by the current NE."""
    return social_optimum / social_welfare_ne

def mixed_strategy_expected_payoffs(full_game, mixed_strategy):
    """
    Compute the expected payoff of a mixed strategy in symmetric games.
    :param full_game: a compact representation of a symmetric game.
    :param mixed_strategy: a numpy array of probabilities.
    """
    assert np.sum(mixed_strategy) == 1
    assert len(mixed_strategy) == len(list(full_game.keys())[0])
    num_strategies = len(mixed_strategy)
    deviation_payoffs = np.zeros(num_strategies)
    for profile in full_game:
        for j in range(num_strategies):
            opponent_profile = list(profile)
            if mixed_strategy[j] == 0: # impossible to be a pivot.
                continue
            if opponent_profile[j] == 0: # impossible to be a pivot.
                continue
            opponent_profile[j] -= 1
            coef = multinomial(opponent_profile)
            prob = 1
            for i, p in enumerate(mixed_strategy):
                prob *= p ** opponent_profile[i]

            deviation_payoffs[j] += coef * prob * full_game[profile][j]

    expected_payoff = np.sum(deviation_payoffs * mixed_strategy)
    return expected_payoff

def multinomial(lst):
    """
    Compute the mulitnomial coefficient.
    :param lst: a list of counts.
    """
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res


def num_beneficial_banks(solver, game_stats, mixed_strategy):
    """
    Compute how many banks are beneficial, neural, and losing under a mixed strategy.
    """
    noop_profile = tuple([solver.num_players] + [0 for _ in range(solver.num_policies - 1)])
    noop_stats = game_stats[noop_profile][0]
    summary_stats = init_summary_stats()

    for profile in game_stats:
        coef = multinomial(list(profile))
        prob = 1
        for i, p in enumerate(mixed_strategy):
            prob *= p ** profile[i]
        coef *= prob
        compare_stats(noop_stats, game_stats[profile][0], summary_stats, coef)

    return summary_stats



def init_summary_stats():
    """
    Initialize a summary container.
    """
    summary_stats = {}
    for measure in measures:
        summary_stats[measure] = {}
        for type in types:
            summary_stats[measure][type] = 0
    return summary_stats


def compare_stats(baseline_stats, stats, overall_stats=None, coef=1):
    """
    Compare the clearing statistics of credit networks given by two profiles (one is the baseline).
    :param baseline_stats: a list of stats of credit networks
    :param stats: a list of stats of credit networks
    :param overall_stats: a dict for summarization.
    :param coef: probability of the profile generating stats.
    """
    summary_stats = init_summary_stats()
    for i, baseline_stat in enumerate(baseline_stats):
        stat = stats[i]
        for measure in measures:
            if measure in ["Bank_asset", "Bank_equity"]:
                if np.sum(baseline_stat[measure]) - np.sum(stat[measure]) > 0.2:
                    summary_stats[measure]["harm"] += 1
                elif np.sum(baseline_stat[measure]) - np.sum(stat[measure]) < 0.2:
                    summary_stats[measure]["benefit"] += 1
                else:
                    summary_stats[measure]["neural"] += 1
            elif measure in ["Default_bank"]:
                if len(baseline_stat[measure]) < len(stat[measure]):
                    summary_stats[measure]["harm"] += 1
                elif len(baseline_stat[measure]) > len(stat[measure]):
                    summary_stats[measure]["benefit"] += 1
                else:
                    summary_stats[measure]["neural"] += 1
            else:
                if len(baseline_stat["Default_bank"]) > 0:
                    average_recover_rate_baseline = 0
                    for default_bank in baseline_stat["Default_bank"]:
                        average_recover_rate_baseline += baseline_stat[measure][default_bank]
                    average_recover_rate_baseline /= len(baseline_stat["Default_bank"])
                else:
                    average_recover_rate_baseline = 1

                if len(stat["Default_bank"]) > 0:
                    average_recover_rate = 0
                    for default_bank in stat["Default_bank"]:
                        average_recover_rate += stat[measure][default_bank]
                    average_recover_rate /= len(stat["Default_bank"])
                else:
                    average_recover_rate = 1

                if average_recover_rate_baseline > average_recover_rate:
                    summary_stats[measure]["harm"] += 1
                elif average_recover_rate_baseline < average_recover_rate:
                    summary_stats[measure]["benefit"] += 1
                else:
                    summary_stats[measure]["neural"] += 1

    for measure in measures:
        for type in types:
            summary_stats[measure][type] *= coef
            if overall_stats is not None:
                overall_stats[measure][type] += summary_stats[measure][type]

    if overall_stats is not None:
        for measure in measures:
            for type in types:
                overall_stats[measure][type] = np.round(overall_stats[measure][type], 4)
    return summary_stats




# game = {}
# game[(2, 0, 0)] = [1, None, None]
# game[(0, 2, 0)] = [None, 5, None]
# game[(0, 0, 2)] = [None, None, 9]
# game[(1, 1, 0)] = [2, 2, None]
# game[(1, 0, 1)] = [3, None, 3]
# game[(0, 1, 1)] = [None, 6, 6]
#
#
# exp = mixed_strategy_expected_payoffs(game, np.array([0.3, 0.3, 0.4]))
# print(exp)





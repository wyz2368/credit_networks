import numpy as np

def evaluate(reduced_game, ne_set, logger, num_players=4):
    """
    The main entry for evaluating the output of an EGTA solver.
    """
    full_symmetric_game, social_optimum, optimum_profile, game_stats = get_full_game(solver)
    for ne in ne_set:
        social_optimum, optimum_profile = get_social_optimum(reduced_game)

        poa = price_of_anarchy(social_optimum, ne_expected_payoff * num_players)
        logger.info("The current NE is {}".format(ne))
        logger.info("The expected payoff per player is {}".format(ne_expected_payoff))
        logger.info("The social optimum profile is {}".format(optimum_profile))
        logger.info("The social optimum is {}".format(social_optimum))
        logger.info("The PoA is {}".format(poa))
        logger.info("-------------------------")


def price_of_anarchy(social_optimum, social_welfare_ne):
    """Compute price of anarchy approximated by the current NE."""
    return social_optimum / social_welfare_ne


def get_social_optimum(reduced_game):
    social_optimum = -np.inf
    optimum_profile = None

    for profile in reduced_game:
        payoffs = reduced_game[profile]
        sw = 0
        for i, cnt in enumerate(list(profile)):
            if cnt == 0:
                continue
            sw += cnt * payoffs[i]

        if sw > social_optimum:
            social_optimum = sw
            optimum_profile = profile

    return social_optimum, optimum_profile


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

def mixed_strategy_expected_payoffs(reduced_game, mixed_strategy):
    """
    Compute the expected payoff of a mixed strategy in symmetric games.
    :param full_game: a compact representation of a symmetric game.
    :param mixed_strategy: a numpy array of probabilities.
    """
    assert np.sum(mixed_strategy) == 1

    num_strategies = len(mixed_strategy)
    deviation_payoffs = np.zeros(num_strategies)
    for profile in reduced_game:
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

            deviation_payoffs[j] += coef * prob * reduced_game[profile][j]

    expected_payoff = np.sum(deviation_payoffs * mixed_strategy)
    return expected_payoff





import numpy as np
import itertools
from itertools import product
import copy

from classic_EGTA.nash_solvers.pygambit_solver import pygbt_solve_matrix_games

def create_profiles(num_players, num_strategies):
    strategy_index = range(num_strategies)
    # Generate all combinations of fruit choices for the given number of people
    combinations = product(strategy_index, repeat=num_players)

    ordered_list = []
    for combo in combinations:
        ordered_list.append(tuple(sorted(list(combo))))
    ordered_list = set(ordered_list)

    # Initialize a list to store the counts for each combination
    combination_counts = []

    # Iterate through each combination and append the counts for each fruit to the list
    for combo in ordered_list:
        counts = []
        for id in strategy_index:
            counts.append(combo.count(id))
        combination_counts.append(counts)

    return combination_counts


def find_pure_equilibria(reduced_game):
    """
    Compute the pure-strategy equilibria in symmetric games.
    :param reduced_game: a dictionary. Keys are profiles and values are payoffs.
    """
    def compute_deviation_profiles(profile):
        deviations = []
        pivotals = []
        n = len(profile)
        for i, count in enumerate(profile):
            if count == 0:
                continue
            for j in range(n):
                if j == i:
                    continue
                deviation = copy.copy(list(profile))
                deviation[j] += 1
                deviation[i] -= 1
                deviations.append(tuple(deviation))
                pivotals.append(tuple([i, j]))

        return deviations, pivotals

    equilibria = []
    for profile in reduced_game:
        current_payoff = reduced_game[profile]
        deviations, pivotals = compute_deviation_profiles(profile)
        no_deviation = True
        for pos, dev in enumerate(deviations):
            i, j = pivotals[pos]
            if reduced_game[dev][j] > current_payoff[i]:
                no_deviation = False
                break

        if no_deviation:
            equilibria.append(profile)

    return equilibria


def convert_to_full_game(num_players, num_policies, symmetric_game):
    total_number_policies = [num_policies for _ in range(num_players)]
    meta_games = [np.full(tuple(total_number_policies), np.nan) for _ in range(num_players)]

    range_iterators = [range(total_number_policies[k]) for k in range(num_players)]

    for current_index in itertools.product(*range_iterators):
        indice = list(current_index)
        profile = [0 for _ in range(num_policies)]
        for i in indice:
            profile[i] += 1
        payoffs = symmetric_game[tuple(profile)]

        for player in range(num_players):
            player_strategy = indice[player]
            meta_games[player][tuple(indice)] = payoffs[player_strategy]

    return meta_games



# game = {}
# game[(2, 0)] = [1, None]
# game[(1, 1)] = [0, 0]
# game[(0, 2)] = [None, 1]

# print(find_pure_equilibria(game))


# game = {}
# game[(3, 0)] = [0.5, None]
# game[(2, 1)] = [1, 2]
# game[(1, 2)] = [2, 1]
# game[(0, 3)] = [None, 0.5]
#
# print("Pure-strategy equilibria:", find_pure_equilibria(game))
# meta_games = convert_to_full_game(num_players=3, num_policies=2, symmetric_game=game)
# # ne = pygbt_solve_matrix_games(meta_games, method="gnm", mode="all")
# print(meta_games)
# print("NE given by gnm:", ne)
#
# imitation_game = [np.array([[[0.5, 1.],
#                              [1., -1.]],
#
#                             [[3., 1.],
#                              [1., 1]]]),
#                   np.array([[[1., 1.],
#                              [0., 0.]],
#
#                             [[0., 0.],
#                              [1., 1.]]]),
#                   np.array([[[1., 0.],
#                              [1., 0.]],
#
#                             [[0., 1.],
#                              [0., 1.]]])]
#
# ne = pygbt_solve_matrix_games(imitation_game, method="simpdiv", mode="all")
# print("NE given by the imitation game", ne)
# print("Pure-strategy equilibria:", pygbt_solve_matrix_games(imitation_game, method="enumpure", mode="all"))

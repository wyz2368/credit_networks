import numpy as np
from itertools import product
import copy

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

# game = {}
# game[(2, 0)] = [1, 1]
# game[(1, 1)] = [0, 0]
# game[(0, 2)] = [1, 1]
#
# print(find_pure_equilibria(game))

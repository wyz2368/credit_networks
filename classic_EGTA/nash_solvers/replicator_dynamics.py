"""
This replicator dynamics solver is to approximate NE in symmetric games.
It is implemented with lazy evaluation.
"""

import numpy as np

def _approx_simplex_projection(updated_strategy, gamma=0.0):
  """Approximately projects the distribution in updated_x to have minimal probabilities.

  Minimal probabilities are set as gamma, and the probabilities are then
  renormalized to sum to 1.

  Args:
    updated_strategy: New distribution value after being updated by update rule.
    gamma: minimal probability value when divided by number of actions.

  Returns:
    Projected distribution.
  """
  # Epsilon approximation of L2-norm projection onto the Delta_gamma space.
  updated_strategy[updated_strategy < gamma] = gamma
  updated_strategy = updated_strategy / np.sum(updated_strategy)
  return updated_strategy


def replicator_step(current_strategy, dev_payoff_function, dt):
    """
    One step of replicator update.
    :param current_strategy: A numpy probability distribution on one player's strategy set.
    :param dev_payoff_function: A deviation payoff simulator.
    :param dt: step size.
    :return: a updated probability distribution.
    """
    current_strategy = current_strategy
    deviation_payoff = dev_payoff_function(current_strategy)
    updated_strategy = current_strategy + dt * deviation_payoff
    updated_strategy = _approx_simplex_projection(updated_strategy, 0.0)

    return updated_strategy


def replicator_dynamics(num_strategies,
                        dev_payoff_function,
                        num_iterations=int(1e4),
                        dt=1e-3,
                        average_over_last_n_strategies=None,
                        initial_strategies=None):
    """
    Only work for symmetric games.
    Run RD given the specified initial strategies.
    :param: dev_payoff_function: A deviation payoff function for lazy evaluation.
    :param: num_iterations: the number of RD iterations.
    :param: dt: step size.
    :param: average_over_last_n_strategies: averaged strategies window size.
    :param num_strategies: a scale.
    :param initial_strategies: a list of probability distributions, one for each player
    :return: a list of probability distributions, one for each player
    """
    new_strategies = initial_strategies or [np.ones(num_strategies) / num_strategies]

    average_over_last_n_strategies = average_over_last_n_strategies or num_iterations

    meta_strategy_window = []
    for i in range(num_iterations):
        new_strategies = replicator_step(new_strategies, dev_payoff_function , dt)
        if i >= num_iterations - average_over_last_n_strategies:
            meta_strategy_window.append(new_strategies)
    average_new_strategies = np.mean(meta_strategy_window, axis=0)

    return average_new_strategies


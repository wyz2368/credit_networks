import numpy as np
#
# a = np.array([[1,2,3], [2,3,4]])
# b = np.array([5,6,7])
# c = np.vstack([a, b])
# print(c[:-1, :])
# print(c[-1, :])

# from collections import Counter
#
# agents = ["1", "2", "3", "4"]
profile = tuple([2, 0, 2])
# policies = ["a", "b", "c"]
# average_result = [1,2,3,4]
#
# a = Counter({"a":1, "b":2, "c":3})
# b = Counter({"a":1, "b":2, "c":3})
# c = {"a":1, "b":2, "c":3}
# d = [a, b, c]
#
#
# def average_payoff_per_policy(average_result, profile):
#     payoffs = []
#     start = 0
#     for count in profile:
#         if count == 0:
#             payoffs.append(None)
#             continue
#         payoffs.append(sum(average_result[start:start + count]) / count)
#         start += count
#
#     return payoffs
#
# print(average_payoff_per_policy(average_result, profile))

def zero_actions(actions):
    all_zeros = True
    for action in actions.values():
        if np.any(action):
            all_zeros = False
    return all_zeros

actions = {}
actions["1"] = np.zeros(5)
actions["2"] = np.zeros(5) + 1

print(zero_actions(actions))
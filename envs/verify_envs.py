import numpy as np


from envs.prepayment_net import Prepayment_Net
from classic_EGTA.strategies import PREPAYMENT_STRATEGIES
from classic_EGTA.clearing import load_pkl

net_pool = load_pkl("../instances/networks_10banks_1000ins_4070ext.pkl")

i = iter(net_pool[:3])
for j in range(11):
    # print(j)
    print(next(i))

# # Load game. This should be adaptive to different environments.
# prepayment_network = Prepayment_Net(num_banks=10,
#                                     default_cost=0.5,
#                                     num_rounds=1,
#                                     utility_type="Bank_asset",
#                                     instance_path="../classic_EGTA/instances/networks_10banks_1000ins.pkl",
#                                     sample_type="enum")
#
# env = prepayment_network
#
# observations, infos = env.reset()
#
# adj_matrix_asset = observations["player_1"]
# adj_m = adj_matrix_asset[:-1, :]
# external_assets = adj_matrix_asset[-1, :]
#
# print("adj_m:", adj_m)
# print("external_assets:", external_assets)
#
# actions = {'player_0': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_1': np.array([0., 0., 0., 0. , 1., 0., 0., 0., 0., 0.]), 'player_2': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_3': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_4': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_5': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), 'player_6': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_7': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_8': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'player_9': np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])}
#
# print("+++++++++++ After taking actions: ")
# observations, rewards, terminations, truncations, infos = env.step(actions)
# adj_matrix_asset = observations["player_1"]
# adj_m = adj_matrix_asset[:-1, :]
# external_assets = adj_matrix_asset[-1, :]
#
# print("adj_m:", adj_m)
# print("external_assets:", external_assets)
#
# observations, rewards, terminations, truncations, infos = env.step(actions)
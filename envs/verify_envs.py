import numpy as np


from envs.prepayment_net import Prepayment_Net
from classic_EGTA.strategies import PREPAYMENT_STRATEGIES
from classic_EGTA.clearing import load_pkl

net_pool = load_pkl("../instances/networks_10banks_1000ins_4070ext.pkl")

i = iter(net_pool[:3])
for j in range(3):
    print(next(i))
    break


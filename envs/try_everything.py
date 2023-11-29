import numpy as np
from classic_EGTA.clearing import clearing

# ext_ast = np.array([ 6.    ,      39.94104039 , 44.97132485 , 57.70845025 , 53.2862818,
#    4.  ,       105.11468192 ,  2.     ,     48.53456724 ,  7.       ])
#
# adj_m = np.array([[ 0,  6, 20,  0, 19,  0,  0,  0, 18,  0],
#  [ 0,  0,  0, 17,  0,  0,  0,  1,  8, 18],
#  [ 0, 14,  0, 13,  0,  0,  1, 19,  8,  0],
#  [ 6,  0,  8,  0,  9,  0,  0,  5,  6,  7],
#  [ 0,  0,  4,  0,  0,  0,  0,  0, 26,  1],
#  [ 0, 13, 28, 32,  0,  0,  0,  3,  0,  0],
#  [15,  2,  0,  0, 18, 17,  0,  0, 31,  4],
#  [ 0,  6,  0,  0,  3,  0, 18,  0,  0,  6],
#  [ 0,  0, 18,  0,  0,  4,  0,  3,  0, 13],
#  [ 2,  0, 22, 11,  0, 18,  4, 27,  0,  0]])

# payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(ext_ast, adj_m, 0.5, 0.5)
# print(Bank_asset)

# [ 27.667       65.90904039  92.40732485  97.90145025  87.4582818
  # 31.         125.44768192  39.612      131.48656724  56.        ]


def is_pure(profile, num_players):
 if len(np.where(np.array(profile) == num_players)[0]) != 0:
  return True
 return False

profile = tuple([0,0,3])
print(is_pure(profile, 3))
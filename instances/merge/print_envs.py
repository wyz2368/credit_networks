from envs.net_generator_prepay import load_pkl
import numpy as np

envs = load_pkl("./networks_merge10banks_1000ins_2070ext.pkl")
# envs = load_pkl("./networks_merge10banks_1000ins_4070ext.pkl")
# envs = load_pkl("./networks_merge10banks_1000ins_40100ext.pkl")
for i, env in enumerate(envs):
    array = env["params"]
    print(array)
    break
    # has_zero_row = np.any(np.all(array == 0, axis=1))
    # if has_zero_row:
    #     print("YES")
    #     break
    # # print(env["params"])



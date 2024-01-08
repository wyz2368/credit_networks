import numpy as np
from scipy import optimize
# from envs.net_generator import load_pkl, save_pkl

def clearing(external_asset, adj_m, ALPHA, BETA):
    bank_number = len(adj_m)  # the number of banks in the financial network
    entry_number = bank_number * bank_number  # the number of entry(element) in adjact matrix

    A_s_h = np.zeros((entry_number, entry_number),
                     dtype=np.float32)  # The structure of the initial coefficient matrix of LP

    b_s_h = []
    for i in range(entry_number):
        x = i // bank_number
        y = i % bank_number
        # print(a[x][y])
        total_liability = np.sum(adj_m, axis=1)[x]
        if total_liability > 0:
            weight = adj_m[x][y] / total_liability
        else:
            weight = 0
        # print('weight: {0}'.format(weight))

        # 等式右边的得数
        b = external_asset[i // bank_number] * weight
        b_s_h.append(b)
        B_s_h = np.array(b_s_h, dtype=np.float32)

        # 系数矩阵
        for j in range(bank_number):
            A_s_h[i][i // bank_number + j * bank_number] = np.zeros((entry_number, entry_number))[i][
                                                               i // bank_number + j * bank_number] - weight
    A_s_h[np.isnan(A_s_h)] = 0

    names = globals()
    names['whether_default_' + str(0)] = np.ones(bank_number, dtype=bool).tolist()
    # print( names['whether_default_' + str(0)])
    for n in range(bank_number + 1):  # 算法最多要跑n rounds 就会收敛，每一轮最少一个银行倒闭
        # 第一次开始round 记为 round=1,n个banks 最多需要n次，所以需要rang(n+1）
        whether_default_0 = [True for i in range(bank_number)]
        new_default = []

        A = np.vstack(
            (np.identity(entry_number, dtype=np.float32), A_s_h + np.identity(entry_number, dtype=np.float32)))
        B = np.vstack((adj_m.reshape(-1, 1), B_s_h.reshape(-1, 1)))
        # print('看看B的变化 {}'.format(B))
        # 线性规划求解
        c = np.ones((1, entry_number))
        res = optimize.linprog(-c, A, B)
        payments_matrix = res.x.reshape(bank_number, bank_number)
        payments_matrix = np.round(payments_matrix, 3)
        payments_matrix = np.where(payments_matrix == 0.0, 0.0, payments_matrix)

        # print('payment matrix is:{}'.format(payments_matrix))

        names['whether_default_' + str(n + 1)] = []
        for i in range(bank_number):  # 对每个银行进行判别
            # False means in default
            if_default = (external_asset[i] + np.sum(payments_matrix, axis=0)[i] + 0.0000001) >= np.sum(adj_m, axis=1)[
                i]
            names['whether_default_' + str(n + 1)].append(if_default)

        if names['whether_default_' + str(n + 1)] != names['whether_default_' + str(n)]:
            for j in range(bank_number):  # 对比新旧default set,找到最新的default bank，然后据此更新矩阵参数
                if names['whether_default_' + str(n + 1)][j] == names['whether_default_' + str(n)][j]:
                    pass
                else:
                    new_default.append(j)
                    if isinstance(BETA, list):
                        A_s_h[bank_number * j: (bank_number * j + bank_number)] = A_s_h[bank_number * j:(
                                bank_number * j + bank_number)] * BETA[j]
                    else:
                        A_s_h[bank_number * j: (bank_number * j + bank_number)] = A_s_h[bank_number * j:(
                                bank_number * j + bank_number)] * BETA
                    if isinstance(ALPHA, list):
                        B_s_h[bank_number * j: (bank_number * j + bank_number)] = B_s_h[bank_number * j:(
                                bank_number * j + bank_number)] * ALPHA[j]
                    else:
                        B_s_h[bank_number * j: (bank_number * j + bank_number)] = B_s_h[bank_number * j:(
                                bank_number * j + bank_number)] * ALPHA
        #             print(names['whether_default_' + str(n + 1)])
        else:
            Bank_equity = []
            Bank_asset = []
            Default_bank = []
            Recover_rate = []
            for b in range(bank_number):
                if adj_m[b, :].sum() == 0:
                    recover_ratio = 1
                else:
                    recover_ratio = payments_matrix[b, :].sum() / adj_m[b, :].sum()
                Recover_rate.append(recover_ratio)
                if not names['whether_default_' + str(n + 1)][b]:
                    equity = 0
                    Default_bank.append(b)
                else:
                    equity = max((payments_matrix[:, b].sum() + external_asset[b] - payments_matrix[b, :].sum()), 0)
                asset = payments_matrix[:, b].sum() + external_asset[b]
                Bank_equity.append(equity)
                Bank_asset.append(asset)
            SW_equity = sum(Bank_equity)
            SW_asset = payments_matrix.sum()
            #             print(names['whether_default_' + str(n + 1)])

            return payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate


# def clearing_instances(load_path="./instances/networks_10banks_1000ins.pkl",
#                        save_path="./instances/",
#                        ALPHA=0.5,
#                        BETA=0.5):
#     networks = load_pkl(load_path)
#     all_stats = []
#     for network in networks:
#         external_assets = network["external_asset"]
#         adj_m = network["adj"]
#
#         payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(
#             external_assets, adj_m, ALPHA, BETA)
#         stat = {
#             "payments_matrix": payments_matrix,
#             "Bank_equity": Bank_equity,
#             "Bank_asset": Bank_asset,
#             "SW_equity": SW_equity,
#             "SW_asset": SW_asset,
#             "Default_bank": Default_bank,
#             "Recover_rate": Recover_rate
#         }
#         all_stats.append(stat)
#
#     save_path += "networks_10banks_1000ins_" + str(ALPHA) + "_solution.pkl"
#
#     save_pkl(all_stats, save_path)
#
#     return all_stats

# if __name__ == "__main__":
#     for alpha in np.linspace(0, 1, 11):
#         alpha = np.round(alpha, 1)
#         clearing_instances(ALPHA=alpha)




# external_assets = [2, 0, 5, 0]
#
# adj_m = np.array([
#     [0, 4, 4, 4],
#     [0, 0, 2, 0],
#     [5, 3, 0, 2],
#     [0, 0, 1, 0]])

external_assets = [5, 5]

adj_m = np.array([
    [0, 2],
    [0, 0]])

# external_assets = [2, 4]
#
# adj_m = np.array([
#     [0, 0],
#     [5, 0]])

# external_assets = [0, 4, 6, 0]
#
# adj_m = np.array([
#     [0, 0, 0, 12],
#     [0, 0, 2, 2],
#     [8, 0, 0, 0],
#     [0, 0, 0, 0]])
#
#
ALPHA = BETA = 0.5

payments_matrix, Bank_equity, Bank_asset, SW_equity, SW_asset, Default_bank, Recover_rate = clearing(external_assets, adj_m, ALPHA, BETA)
print(payments_matrix)
print(Bank_equity)
print(Bank_asset)
print(SW_equity)
print(SW_asset)
print(Default_bank)
print(Recover_rate)
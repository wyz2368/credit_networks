from envs.net_generator_prepay import load_pkl

def compute_max_sw(reduced_game):
    max_sw = -1000
    for profile in reduced_game:
        sw = 0
        for i, cnt in enumerate(profile):
            if cnt == 0:
                continue
            sw += cnt * reduced_game[tuple(profile)][i]

        if sw > max_sw:
            max_sw = sw

    return max_sw

#
# default_cost = ["1", "3", "5", "7", "9"]
# path = "./reduced_game_40100_0"
#
# for cost in default_cost:
#     print(cost)
#     complete_path = path + cost + '.pkl'
#     reduced_game = load_pkl(complete_path)
#     max_sw = compute_max_sw(reduced_game)
#     ne_sw = reduced_game[(0,0,0,0,4,0)]
#     print(ne_sw / max_sw)

reduced_game = load_pkl("./reduced_game.pkl")
print(reduced_game[(0, 0, 3, 1, 0)])
print(reduced_game[(0, 0, 0, 4, 0)])
print(reduced_game[(0, 0, 0, 0, 4)])



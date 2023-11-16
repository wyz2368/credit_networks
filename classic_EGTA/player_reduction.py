
def deviation_preserve_reduction(reduced_profile, num_players, reduce_num_players):
    if sum(reduced_profile) != reduce_num_players:
        raise ValueError("sum(reduced_profile) != reduce_num_players")
    num_strategies = len(reduced_profile)
    original_profiles = []
    coef = (num_players - 1)/(reduce_num_players - 1)

    for i in range(num_strategies):
        original_profile = []
        for j, reduced_count in enumerate(reduced_profile):
            if reduced_count == 0:
                original_profile.append(0)
                continue
            if i == j:
                original_cnt = coef * (reduced_count - 1) + 1
            else:
                original_cnt = coef * reduced_count
            original_profile.append(original_cnt)

        original_profiles.append(original_profile)

    return original_profiles



# print(deviation_preserve_reduction([3,0,2], 25, 5))
from pettingzoo.utils.wrappers import OrderEnforcingWrapper, AssertOutOfBoundsWrapper
from pettingzoo.utils.conversions import parallel_to_aec_wrapper, aec_to_parallel_wrapper
def parallel_to_aec(par_env):
    """Converts a Parallel environment to an AEC environment.

    In the case of an existing AEC environment wrapped using a `aec_to_parallel_wrapper`, this function will return the original AEC environment.
    Otherwise, it will apply the `parallel_to_aec_wrapper` to convert the environment.
    """
    if isinstance(par_env, aec_to_parallel_wrapper):
        raise ValueError("The env is an AEC.")
    else:
        aec_env = Customized_parallel_to_aec_wrapper(par_env)
        ordered_env = Customized_OrderEnforcingWrapper(aec_env)
        return ordered_env

class Customized_parallel_to_aec_wrapper(parallel_to_aec_wrapper):
    def __init__(self, parallel_env):
        super().__init__(parallel_env)

    def get_stats(self):
        return self.env.get_stats()

    def get_num_players(self):
        return self.env.get_num_players()

class Customized_OrderEnforcingWrapper(OrderEnforcingWrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_stats(self):
        return self.env.get_stats()

    def get_num_players(self):
        return self.env.get_num_players()

class Customized_AssertOutOfBoundsWrapper(AssertOutOfBoundsWrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_stats(self):
        return self.env.get_stats()

    def get_num_players(self):
        return self.env.get_num_players()




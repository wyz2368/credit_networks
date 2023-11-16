import datetime
import os
from absl import app
from absl import flags
import logging
import numpy as np

from envs.prepayment_net import create_env, Prepayment_Net
from classic_EGTA.egta_solver import EGTASolver
from classic_EGTA.strategies import PREPAYMENT_STRATEGIES
from classic_EGTA.clearing import save_pkl



FLAGS = flags.FLAGS
# Game-related
flags.DEFINE_string("game_name", "prepayment_game", "Game name.")
flags.DEFINE_integer("num_banks", 10, "The number of players.")
flags.DEFINE_integer("sim_per_profile", 1000, "The number of simulations per profile.")
flags.DEFINE_integer("reduce_num_players", 4, "The number of players in the reduced game.")
flags.DEFINE_integer("reduce_num_players", 4, "The number of players in the reduced game.")
flags.DEFINE_float("default_cost", 0.5, "Default cost")
flags.DEFINE_string("utility_type", "Bank_asset", "Options: Bank_asset, Bank_equity")


# General
flags.DEFINE_string("root_result_folder", 'root_result_psro', "root directory of saved results")
flags.DEFINE_integer("seed", None, "Seed.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")

def init_logger(logger_name, checkpoint_dir):
  # Set up logging info.
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(checkpoint_dir + "/data.log")
  file_handler.setLevel(logging.INFO)
  logger.addHandler(file_handler)

  return logger


def init_strategies():
     return list(PREPAYMENT_STRATEGIES.values())

def egta_runner(env, checkpoint_dir):
    logger = init_logger(logger_name=__name__,
                         checkpoint_dir=checkpoint_dir)

    initial_strategies = init_strategies()

    egta_solver = EGTASolver(env=env,
                             sim_per_profile=FLAGS.sim_per_profile,
                             initial_strategies=initial_strategies,
                             reduce_num_players=FLAGS.reduce_num_players)

    equilibria = egta_solver.run()
    logger.info("Equilibria: {}".format(equilibria))
    stats = egta_solver.get_stats()
    save_pkl(stats, checkpoint_dir + "/stats.pkl")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    seed = np.random.randint(0, 10000)

    # Load game. This should be adaptive to different environments.
    prepayment_network = Prepayment_Net(num_banks=FLAGS.num_banks,)
    env = create_env(prepayment_network)

    # Set up working directory.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir = checkpoint_dir + "_oracle_" + FLAGS.oracle_type + '_se_' + str(seed) + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    egta_runner(env, checkpoint_dir=checkpoint_dir)



if __name__ == "__main__":
    app.run(main)
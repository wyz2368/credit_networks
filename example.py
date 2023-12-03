import datetime
import os
from absl import app
from absl import flags
import logging
import numpy as np

from envs.prepayment_net import create_env, Prepayment_Net
from classic_EGTA.egta_solver import EGTASolver
from classic_EGTA.strategies import PREPAYMENT_STRATEGIES


FLAGS = flags.FLAGS
# Game-related
flags.DEFINE_string("game_name", "prepayment_game", "Game name.")
flags.DEFINE_integer("num_banks", 10, "The number of players.")
flags.DEFINE_integer("sim_per_profile", 1, "The number of simulations per profile.")
flags.DEFINE_integer("reduce_num_players", 4, "The number of players in the reduced game.")
flags.DEFINE_integer("num_rounds", 1, "The max number of time steps for truncation.")
flags.DEFINE_float("default_cost", 0.5, "Default cost")
flags.DEFINE_string("utility_type", "Bank_asset", "Options: Bank_asset, Bank_equity")
flags.DEFINE_string("sample_type", "enum", "Options: random, enum")
flags.DEFINE_string("instance_path", "../instances/networks_10banks_1000ins_4070ext.pkl", "Path to instances.")

#networks_10banks_1000ins.pkl

# General
flags.DEFINE_string("root_result_folder", 'root_result', "root directory of saved results")


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


def logging_game_info(logger):
    logger.info("game_name: {}".format(FLAGS.game_name))
    logger.info("num_banks: {}".format(FLAGS.num_banks))
    logger.info("sim_per_profile: {}".format(FLAGS.sim_per_profile))
    logger.info("reduce_num_players: {}".format(FLAGS.reduce_num_players))
    logger.info("num_rounds: {}".format(FLAGS.num_rounds))
    logger.info("default_cost: {}".format(FLAGS.default_cost))
    logger.info("sample_type: {}".format(FLAGS.sample_type))
    logger.info("utility_type: {}".format(FLAGS.utility_type))
    logger.info("instance_path: {}".format(FLAGS.instance_path))
    logger.info("======= Begin Running EGTA =======")



def egta_runner(env, checkpoint_dir):
    logger = init_logger(logger_name=__name__,
                         checkpoint_dir=checkpoint_dir)
    logging_game_info(logger)

    initial_strategies = init_strategies()

    egta_solver = EGTASolver(env=env,
                             sim_per_profile=FLAGS.sim_per_profile,
                             initial_strategies=initial_strategies,
                             reduce_num_players=FLAGS.reduce_num_players,
                             checkpoint_dir=checkpoint_dir)

    pure_equilibria, RD_equilibria = egta_solver.run()
    logger.info("Pure Equilibria: {}".format(pure_equilibria))
    logger.info("RD Equilibria: {}".format(RD_equilibria))

    # Evaluation
    egta_solver.observe(logger)





def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    seed = np.random.randint(0, 10000)
    # random.seed(10)
    # np.random.seed(10)

    # Load game. This should be adaptive to different environments.
    prepayment_network = Prepayment_Net(num_banks=FLAGS.num_banks,
                                        default_cost=FLAGS.default_cost,
                                        num_rounds=FLAGS.num_rounds,
                                        utility_type=FLAGS.utility_type,
                                        instance_path=FLAGS.instance_path,
                                        sample_type=FLAGS.sample_type)
    env = prepayment_network

    # Set up working directory.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir = checkpoint_dir + '_ut_' + FLAGS.utility_type + '_dfcost_' + str(FLAGS.default_cost) + "_ins_" + FLAGS.instance_path[-11:-4] + '_se_' + str(seed) + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    egta_runner(env, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    app.run(main)


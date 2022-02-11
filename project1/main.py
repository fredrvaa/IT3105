import argparse

from environments.environment import Environment
from environments.gambler import Gambler
from learner.actor_critic import ActorCritic
from utils.config_parser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/config/file', required=True)
args = parser.parse_args()

config_parser = ConfigParser(args.config)

environment: Environment = config_parser.environment

actor_critic: ActorCritic = config_parser.actor_critic

fit_kwargs = config_parser.fit_parameters

actor_critic.fit(**fit_kwargs)
actor_critic.visualize_fit()
actor_critic.run(visualize=True)
if type(environment) is Gambler:
    actor_critic.actor.visualize_strategy()

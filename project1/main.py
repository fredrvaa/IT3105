import argparse

import numpy as np
import torch

from environments.environment import Environment
from environments.gambler import Gambler
from learner.actor_critic import ActorCritic
from utils.config_parser import ConfigParser
import random

# Set seed for reproducibility
SEED = 14
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/config/file', required=True)
parser.add_argument('-v', '--visualize', action='store_true', help='Flag used to get visualizations.')
args = parser.parse_args()

config_parser = ConfigParser(args.config)

environment: Environment = config_parser.environment

actor_critic: ActorCritic = config_parser.actor_critic

fit_parameters = config_parser.fit_parameters
visualization_parameters = config_parser.visualization_parameters
show = visualization_parameters['show']
vis_sleep = visualization_parameters['vis_sleep']

print('---FITTING MODEL---')
actor_critic.fit(**fit_parameters)
if show:
    actor_critic.visualize_fit()

print('---RUNNING MODEL---')
actor_critic.run(visualize=show, vis_sleep=vis_sleep)
if type(environment) is Gambler and show:
    actor_critic.actor.visualize_strategy()

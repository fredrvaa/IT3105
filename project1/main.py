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

fit_kwargs = config_parser.fit_parameters

print('---FITTING MODEL---')
actor_critic.fit(**fit_kwargs)
if args.visualize:
    actor_critic.visualize_fit()

print('---RUNNING MODEL---')
actor_critic.run(visualize=args.visualize)
if type(environment) is Gambler and args.visualize:
    actor_critic.actor.visualize_strategy()

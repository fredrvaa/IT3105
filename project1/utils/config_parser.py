import inspect

import yaml

from environments.environment import Environment
from environments.cartpole import CartPole
from environments.towers_of_hanoi import TowersOfHanoi
from environments.gambler import Gambler
from learner.actors.actor import Actor
from learner.critics.critic import Critic
from learner.critics.table_critic import TableCritic
from learner.critics.network_critic import NetworkCritic
from learner.actor_critic import ActorCritic

STRING_EXCEPTIONS = ['name', 'checkpoint_folder']


class ConfigParser:
    def __init__(self, config_file: str):
        with open(config_file, "r") as stream:
            self._config = yaml.safe_load(stream)

        self.environment: Environment = self._get_environment()
        self.actor_critic: ActorCritic = self._get_actor_critic()
        self.fit_parameters: dict = self._get_fit_parameters()

    def _parse_config(self, config) -> dict:
        parsed_config = {}
        for k, v in config.items():
            if v is not None:
                if type(v) is dict:
                    parsed_config[k] = self._parse_config(v)
                elif type(v) is str and k not in STRING_EXCEPTIONS:
                    print(v)
                    parsed_config[k] = eval(v)
                else:
                    parsed_config[k] = v
                parsed_config
        return parsed_config

    def _get_environment(self) -> Environment:
        environment = eval(self._config['environment_type'])
        kwargs = self._parse_config(self._config['environment_params'])
        return environment(**kwargs)

    def _get_actor(self) -> Actor:
        actor = eval(self._config['actor_type'])
        kwargs = self._parse_config(self._config['actor_params'])
        return actor(environment=self.environment, **kwargs)

    def _get_critic(self) -> Critic:
        critic = eval(self._config['critic_type'])
        kwargs = self._parse_config(self._config['critic_params'])
        return critic(environment=self.environment, **kwargs)

    def _get_actor_critic(self) -> ActorCritic:
        actor = self._get_actor()
        critic = self._get_critic()
        return ActorCritic(environment=self.environment, actor=actor, critic=critic)

    def _get_fit_parameters(self) -> dict:
        return self._parse_config(self._config['fit'])

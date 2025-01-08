from gymnasium.envs.registration import register

#from vrp
from typing import TypeVar
from .tsp import TSPEnv 

register(
    id="tsp/GridWorld-v0",
    entry_point="tsp.envs:GridWorldEnv",
)

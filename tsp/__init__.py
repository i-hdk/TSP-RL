from gymnasium.envs.registration import register

register(
    id="tsp/TSP-v0",
    entry_point="tsp.envs:TSPEnv",
)

#copied from custom env website
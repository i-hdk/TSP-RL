from gymnasium.envs.registration import register

register(
    id="tsp/GridWorld-v0",
    entry_point="tsp.envs:GridWorldEnv",
)

from gymnasium.envs.registration import register

register(
     id="gym_examples/PgEnv-v0",
     entry_point="gym_examples.envs:PgEnv",
     max_episode_steps=30
)
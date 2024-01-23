import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
env.reset()
env.render()


def move_n(n, _action):
    _obs, _reward, _terminated, _truncated, _info = env.step(_action)
    for _ in range(0, n-1):
        print(np.degrees(_obs[2]))
        if _terminated or _truncated:
            break
        _obs, _reward, _terminated, _truncated, _info = env.step(_action)
    env.render()
    return _obs, _reward, _terminated, _truncated, _info


obs, reward, terminated, truncated, info = env.step(1)

while True:
    k = abs(int(np.degrees(obs[2]))-2)//3
    if np.degrees(obs[2]) >= 0:
        action = 1
    else:
        action = 0
    obs, reward, terminated, truncated, info = move_n(k, action)
    if terminated or truncated:
        break

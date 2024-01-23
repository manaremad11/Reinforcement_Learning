import gymnasium as gym
import numpy as np

seed = 2348769034

# iterative
T = 8
n_states = 2 ** T
policies = np.zeros(shape=(n_states, T))

## find all permutations
for i in range(T):
    k = 0
    c = 2 ** (T - i - 1)
    while k < n_states:
        k += c
        for j in range(k, k + c):
            policies[j][i] = 1
        k += c

print(policies.shape)

policies = np.append(policies, np.zeros(shape=(n_states, 1)), axis=1)

env = gym.make("CartPole-v1", render_mode="human")

for i in range(n_states):
    env.reset(seed=seed)
    env.render()
    k = 0

    while k < T:
        action = int(policies[i][k])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
        k += 1
    policies[i][T] = k

mx = 0
mx_index = 0

for i in range(n_states):
    if policies[i][T] > mx:
        mx_index = i
        mx = policies[i][T]

env.reset(seed=seed)
env.render()
k = 0

while k < T:
    action = int(policies[mx_index][k])
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
    k += 1

print(policies[mx_index][:T])
print(policies[mx_index][T])

# q-value way

iterations = 150
T = 10
epsilon = 0

policies = np.zeros(shape=(iterations, T + 1))

for i in range(iterations):
    env.reset(seed=seed)
    env.render()
    k = 0
    while k < T:
        num = np.random.random(1)[0]
        if num >= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(iterations[:i], axis=0)[k]
        policies[i][k] = action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
        k += 1
        if k % 100 == 0:
            epsilon = np.min(epsilon + .15, 0.75)
    policies[i][T] = k

mx = 0
mx_index = 0

for i in range(iterations):
    if policies[i][T] > mx:
        mx_index = i
        mx = policies[i][T]

env.reset(seed=seed)
env.render()

k = 0
while k < T:
    action = int(policies[mx_index][k])
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
    k += 1

print(policies[mx_index][:T])
print(policies[mx_index][T])

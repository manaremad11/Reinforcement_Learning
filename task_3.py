import gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class MountainCarV0Agent:
    def __init__(self, buckets=(40, 40, 3), num_episodes=1000, lr=0.1, min_explore=0.1, discount=1.0):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.lr = lr
        self.min_explore = min_explore
        self.discount = discount
        self.env = gym.make('MountainCar-v0')
        self.Q = np.random.uniform(low=0, high=1, size=self.buckets)
        self.policy = np.zeros(self.buckets[:-1], dtype='int')
        self.N = np.zeros(self.buckets)
        diff = (self.env.observation_space.high - self.env.observation_space.low)
        self.num_states = tuple(np.floor(self.buckets[:-1]/diff).astype(int))

    def discretize_state(self, state):
        result_state = tuple(np.round((state - self.env.observation_space.low) * self.num_states).astype(int))
        return result_state

    def get_explore_rate(self, t):
        return max(1-2*(t/self.num_episodes), self.min_explore)

    def choose_action(self, state, t):
        if np.random.uniform() > self.get_explore_rate(t):
            return np.argmax(self.Q[state])
        else:
            return self.env.action_space.sample()

    def generate_episode(self):
        episode = []
        score = 0
        done = False
        state = self.discretize_state(self.env.reset()[0])

        while not done and score > -1000:
            action = self.policy[state]
            next_state, reward, done, truncated, info = self.env.step(action)
            episode.append((state, action, reward))
            state = self.discretize_state(next_state)
            score += reward

        return episode, score

    def monte_carlo_Q_evaluation(self, episode):
        episode = np.array(episode, dtype='object')
        state_actions = [(state, action) for state, action, _ in episode]
        G = 0
        for (state, action, reward), i in enumerate(reversed(episode)):
            G = self.discount * G + reward
            if (state, action) not in state_actions[:-(i+1)]:
                self.N[state][action] += 1
                self.Q[state][action] += (1/self.N[state][action]) * (G - self.Q[state][action])

    def monte_carlo_policy_evaluation(self, t):
        for state in np.ndindex(*self.buckets[:-1]):
            self.policy[state] = self.choose_action(state, t)

    def update_policy(self):
        for state in np.ndindex(*self.buckets[:-1]):
            self.policy[state] = np.argmax(self.Q[state])

    def monte_carlo_train(self):
        episode_rewards = []

        self.monte_carlo_policy_evaluation(0)
        for t in tqdm(range(self.num_episodes)):
            episode, score = self.generate_episode()
            self.monte_carlo_Q_evaluation(episode)
            self.monte_carlo_policy_evaluation(t)
            episode_rewards.append(score)

        self.update_policy()

        return episode_rewards

    def QLearning_train(self):
        episode_rewards = []
        for t in tqdm(range(self.num_episodes)):
            done = False
            score = 0
            state = self.discretize_state(self.env.reset()[0])

            while not done and score > -1000:
                action = self.choose_action(state, t)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = self.discretize_state(next_state)
                self.Q[state][action] = self.Q[state][action] + \
                    self.lr * (reward + self.discount * np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state
                score += reward
            episode_rewards.append(score)

        self.update_policy()
        return episode_rewards

    def sarsa_train(self):
        episode_rewards = []
        for t in tqdm(range(self.num_episodes)):
            done = False
            score = 0
            state = self.discretize_state(self.env.reset()[0])
            action = self.choose_action(state, t)
            while not done and score > -1000:
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = self.discretize_state(next_state)
                next_action = self.choose_action(next_state, t)
                self.Q[state][action] = self.Q[state][action] + \
                    self.lr * (reward + self.discount * self.Q[next_state][next_action] - self.Q[state][action])
                state, action = next_state, next_action
                score += reward
            episode_rewards.append(score)

        self.update_policy()
        return episode_rewards

    def test(self):
        env_test = gym.make('MountainCar-v0', render_mode='human')
        state = self.discretize_state(env_test.reset()[0])
        score = 0
        done = False
        while not done and score > -1000:
            env_test.render()
            action = self.policy[state]
            next_state, reward, done, truncated, info = env_test.step(action)
            state = self.discretize_state(next_state)
            score += reward
        return score


def draw_score_plot(returns, plot_name):
    mean_rewards = []
    for i in range(len(returns)):
        mean_rewards.append(np.mean(returns[max(0, i - 50):(i + 1)]))
    plt.plot(mean_rewards, label=plot_name)


def main():
    # QLearning
    q_agent = MountainCarV0Agent(buckets=(20, 20, 3), num_episodes=1000, lr=.1,
                                 min_explore=0.01, discount=.99)
    returns = q_agent.QLearning_train()
    q_loss = np.mean(returns)
    q_agent.test()
    draw_score_plot(returns, 'QLearning')

    # Sarsa
    sarsa_agent = MountainCarV0Agent(buckets=(20, 20, 3), num_episodes=5000, lr=.1,
                                 min_explore=0.01, discount=.99)
    returns = sarsa_agent.sarsa_train()
    sarsa_loss = np.mean(returns)
    draw_score_plot(returns, 'Sarsa')

    # Monte Carlo
    mc_agent = MountainCarV0Agent(buckets=(20, 20, 3), num_episodes=5000, lr=.1,
                                 min_explore=0.01, discount=.99)
    returns = mc_agent.QLearning_train()
    mc_loss = np.mean(returns)
    draw_score_plot(returns, 'MC')

    # save results
    print('QLearning loss', q_loss)
    print('sarsa loss', sarsa_loss)
    print('mc loss', mc_loss)
    plt.legend()
    plt.savefig("plot_name")
    plt.show()


if __name__ == '__main__':
    main()


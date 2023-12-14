import numpy as np
import gym

# Q-learning params
ALPHA = 0.1  # learning rate
GAMMA = 0.99  # reward discount
EPSILON = 0.1  
LEARNING_COUNT = 5000
TEST_COUNT = 100
TURN_LIMIT = 1000

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def select_action(self, state):
        if np.random.rand() < EPSILON:
            return self.env.action_space.sample()  
        else:
            return np.argmax(self.q_table[state])  

    def learn(self):
        state = self.env.reset()

        for t in range(TURN_LIMIT):
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # Q-learning update rule
            self.q_table[state][action] += ALPHA * (reward + GAMMA * np.max(self.q_table[next_state]) - self.q_table[state][action])

            if done:
                break
            else:
                state = next_state

    def test(self):
        state = self.env.reset()
        total_reward = 0.0

        for t in range(TURN_LIMIT):
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                break
            else:
                state = next_state

        return total_reward

def main():
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(env)

    print("###### LEARNING #####")
    for i in range(LEARNING_COUNT):
        agent.learn()

    print("###### TEST #####")
    total_reward = 0.0
    for i in range(TEST_COUNT):
        total_reward += agent.test()

    print("Average reward during testing: {:.2f}".format(total_reward / TEST_COUNT))

if __name__ == "__main__":
    main()
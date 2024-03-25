
import gym
from agent import Agent
env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(action_size, state_size)

episodes = 600
batch_size = 32

scores = []
for e in range(episodes):
  state = env.reset()


  for i in range(500):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    reward = reward if not done else -10
    agent.remember(state, action, next_state, reward, done)

    state = next_state
    if done:
      print("episode: {}/{}, score: {}, epsilon: {:.2}"
            .format(e, episodes, i, agent.epsilon))

      break
  agent.learn(batch_size)

  if e % 10 == 0:
    agent.update_target_network()
    print("target network updated")





from collections import deque
import numpy as np
import torch

def ddpg(env, agent, n_episodes=1000, max_t=2000, print_every=100):
    run(env, agent, n_episodes=n_episodes, max_t=max_t, print_every=print_every)

def run(env, agent, n_episodes=1000, max_t=2000, print_every=100):

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores_deque = deque(maxlen=print_every)
    scores = []

    for i_episode in range(1, n_episodes+1):

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        # get the current state
        state = env_info.vector_observations[0]

        agent.reset()

        score = 0

        for t in range(max_t):

            action = agent.act(state)

            # send the action to the environment
            env_info = env.step(action)[brain_name]

            # get the next state
            next_state = env_info.vector_observations[0]

            # get the reward
            reward = env_info.rewards[0]

            # see if episode has finished
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque)>= 30.0:

            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-print_every, np.mean(scores_deque)))

            agent.save_checkpoint()

            break

    return scores

# Report

### Learning Algorithm

This is an implementation of the DDQN algorithm whose respective paper is referenced in the README.md file. I used the code from the ddqn-pendulumn project as a starting point, which is located here: https://github.com/udacity/deep-reinforcement-learning/tree/master/ddqn-pendulum.

I modified the code to work with the Reacher environment, and began testing different model aritechture's and hyperparameters.

### Model Architecture

The final model aritechture uses an Actor/Critic setup.

The Actor is a simple neural network consisting of 3 fully connected layers. The model takes as input a vector of the state space (33 states), runs it through a first layer of512 connected units, then a second layer of 256 units again, and finally through a classification layer that outputs a vector of probabilities the respective action space (4 in this case).

The Critic is a simple neural network with the same architecture, only it outputs a single continuous value.

### Hyper parameters

I started out with the default parameters in the starter code:
BUFFER_SIZE = int(1e5) # replay buffer size
BATCH_SIZE = 128 # minibatch size
GAMMA = 0.99 # discount factor
TAU = 1e-3 # for soft update of target parameters
LR_ACTOR = 1e-4 # learning rate
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

UPDATE_EVERY = 4 # how often to update the network

### Submission parameters:

The best result was acheived using the following hyperparameters:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.001

UPDATE_EVERY = 4        # how often to update the network

You can look at `Explore.ipynb` for a list of the experiments I tried, changing hyperparameter's in code files between different runs just trying to replicate the baseline results in the 'where to start' project section. I was unable to get the average reward above 5 for many experiments and was stuck for a long time until I finally tried setting the Actor and Critic learning rates to both be 1e-4. This finally lead to meaningful increase and average reward, at which point I ran the environment finally against enough episodes to pass the environment.

You can see a plot of rewards for my final submission in the last code cell of `Explore.ipynb`.

### Implementation notes:

I also implemented gradient clipping as described in the project description, and only update the network weights every 4 timestep's to control learning stability.

### Ideas for Future Work:
- Try solving the mulitiple Reacher arm environment.
- Try PPO.
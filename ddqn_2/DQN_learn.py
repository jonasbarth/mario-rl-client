"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from replay_buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}


class DDQNAgent:

    def __init__(
            self,
            env,
            num_episodes,
            q_func, 
            optimizer_spec, 
            exploration, 
            replay_buffer_size, 
            batch_size, 
            gamma, 
            learning_starts,
            learning_freq,
            frame_history_len,
            target_update_freq,
            max_steps
            ):

        self.env = env
        self.q_func = q_func
        self.optimizer_spec = optimizer_spec
        self.exploration = exploration
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.num_episodes = num_episodes

        self.img_h, self.img_w, self.img_c = env.preprocessor.scaled_height, env.preprocessor.scaled_width, env.preprocessor.n_channels
        self.input_arg = self.frame_history_len * self.img_c
        #num_actions = env.action_space.shape
        self.num_actions = len(env.all_actions)
        self.model_name = "DQN"
        self.max_steps = max_steps


    def select_epilson_greedy_action(self, model, obs, t):
        sample = random.random()
        eps_threshold = self.exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return model(Variable(obs, volatile=True)).data.max(1)[1].view(-1, 1).cpu()
        else:
            return torch.IntTensor([[random.randrange(self.num_actions)]])

    def status_to_bool(self, game_status):
        map = {"RUNNING": True, "TIME_OUT": False, "WIN": False, "LOSE": False}
        return map[game_status]


    def train(self):
       
        comment = f'_model={self.model_name} replay_buffer_size={self.replay_buffer_size} batch_size={self.batch_size} \
        gamma={self.gamma} learning_starts={self.learning_starts} learning_freq={self.learning_freq} \
        target_update_freq={self.target_update_freq} \
        num_episodes={self.num_episodes} level={self.env.level_name()} egocentric={self.env.egocentric} frame_skip={self.env.frame_skip}'

        # Writer for Tensorboard
        tb = SummaryWriter(comment=comment)


        # Initialize target q function and q function
        Q = self.q_func(self.input_arg, self.num_actions).type(dtype)
        target_Q = self.q_func(self.input_arg, self.num_actions).type(dtype)

        # Construct Q network optimizer function
        optimizer = self.optimizer_spec.constructor(Q.parameters(), **self.optimizer_spec.kwargs)

        # Construct the replay buffer
        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len)

        # The total loss over the training episodes
        total_loss = np.array([])
        total_reward = np.array([])
        total_steps = 0


        for i_episode in range(self.num_episodes):
            print("Starting episode", i_episode)
            ###############
            # RUN ENV     #
            ###############
            num_param_updates = 0
            mean_episode_reward = -float('nan')
            best_mean_episode_reward = -float('inf')
            last_obs, reward, game_status = self.env.start_state()
            game_status = self.status_to_bool(game_status)
            LOG_EVERY_N_STEPS = 10000
            
            # The loss and for the episode
            i_loss = np.array([])
            i_reward = np.array([])

            

            for t in count():
                #print("\tStep", t)
                ### Step the env and store the transition
                # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                last_idx = replay_buffer.store_frame(last_obs)
                # encode_recent_observation will take the latest observation
                # that you pushed into the buffer and compute the corresponding
                # input that should be given to a Q network by appending some
                # previous frames.
                recent_observations = replay_buffer.encode_recent_observation()
                
                action = None
                # Choose random action if not yet start learning
                if total_steps > self.learning_starts:
                    action = self.select_epilson_greedy_action(Q, recent_observations, total_steps)
                   
                else:
                    action = torch.IntTensor([[random.randrange(self.num_actions)]])
                    
                # Advance one step
                
                obs, reward, game_status = self.env.step(action)
                game_status = self.status_to_bool(game_status)
                #print(reward)
                tb.add_scalar("Reward per timestep", reward, total_steps)
                i_reward = np.append(i_reward, reward)
               
                # Store other info in replay memory
                replay_buffer.store_effect(last_idx, action, reward, game_status)
                # Resets the environment when reaching an episode boundary.
                if not game_status:
                    break
                    #obs, reward, game_status = env.start_state()
                    #game_status = self.status_to_bool(game_status)

                # The last observartion becomes the current observation
                last_obs = obs

                ### Perform experience replay and train the network.
                # Note that this is only done if the replay buffer contains enough samples
                # for us to learn something useful -- until then, the model will not be
                # initialized and random actions should be taken
                # 
                if (total_steps > self.learning_starts and
                        total_steps % self.learning_freq == 0 and
                        replay_buffer.can_sample(self.batch_size)):

                    print("\tTraining network")
                    # Use the replay buffer to sample a batch of transitions
                    # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                    # in which case there is no Q-value at the next state; at the end of an
                    # episode, only the current state reward contributes to the target
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(self.batch_size)
                    # Convert numpy nd_array to torch variables for calculation
                    obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
                    act_batch = Variable(torch.from_numpy(act_batch).long())
                    rew_batch = Variable(torch.from_numpy(rew_batch))
                    next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
                    not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

                    if USE_CUDA:
                        act_batch = act_batch.cuda()
                        rew_batch = rew_batch.cuda()

                       
                    # Compute current Q value, q_func takes only state and output value for every state-action pair
                    # We choose Q based on action taken.
                    current_Q_values = Q(obs_batch).gather(1, act_batch.view(-1, 1))
                    """
                    # DQN
                    # Compute next Q value based on which action gives max Q values
                    # Detach variable from the current graph since we don't want gradients for next Q to propagated
                    next_max_q = target_Q(next_obs_batch).detach().max(1)[0].view(-1, 1)
                    next_Q_values = not_done_mask.view(-1, 1) * next_max_q
                    """
                    next_argmax_action = Q(next_obs_batch).max(1)[1].view(-1, 1)
                    next_q = target_Q(next_obs_batch).detach().gather(1, next_argmax_action)
                    next_Q_values = not_done_mask.view(-1, 1) * next_q 
                    # Compute the target of the current Q values
                    target_Q_values = rew_batch.view(-1, 1) + (self.gamma * next_Q_values)
                    """
                    # Compute Bellman error
                    bellman_error = target_Q_values - current_Q_values
                    # clip the bellman error between [-1 , 1]
                    clipped_bellman_error = bellman_error.clamp(-1, 1)
                    # Note: clipped_bellman_delta * -1 will be right gradient
                    d_error = clipped_bellman_error * -1.0
                
                    # Clear previous gradients before backward pass
                    optimizer.zero_grad()
                    # run backward pass
                    current_Q_values.backward(d_error.data)
                    """
                    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

                    # Add loss per timestep to the tensorboard
                    tb.add_scalar("Loss per timestep", loss.item(), total_steps)

                    # Save the loss of this episode
                    i_loss = np.append(i_loss, loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    for param in Q.parameters():
                        param.grad.data.clamp(-1, 1)
                    # Perfom the update
                    optimizer.step()
                    num_param_updates += 1

                    # Periodically update the target network by Q network to target Q network
                    if num_param_updates % self.target_update_freq == 0:
                        target_Q.load_state_dict(Q.state_dict())

                total_steps += 1

            # Add the cumulative loss of the episode to tensorboard
            tb.add_scalar("Cumulative Loss per episode", i_loss.sum(), i_episode)
            tb.add_scalar("Cumulative Reward per episode", i_reward.sum(), i_episode)
            tb.add_scalar("Average reward per episode", i_reward.mean(), i_episode)
            tb.add_scalar("Average Loss per episode", i_loss.mean(), i_episode)
            

            # Append the average loss over episode i to the total loss
            print("Loss", i_loss)
            total_loss = np.append(total_loss, i_loss.mean())
            i_loss = np.array([])
            print("Episode", i_episode, "finished")

        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.env.directory("./models/" + dt)
        self.env.save_parameters("./models/" + dt + "/hyperparameters.json")
        policy_path = "./models/" + dt + "/eps_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "policy.pt"
        target_path = "./models/" + dt + "/eps_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "target.pt"
        self.env.save_model(Q, policy_path)
        self.env.save_model(target_Q, target_path)
        self.env.zip_directory("./models/" + dt, dt)
        print("Training finished")


    def train_steps(self, policy=None, target=None):
        
       
        comment = f'_model={self.model_name} replay_buffer_size={self.replay_buffer_size} batch_size={self.batch_size} \
        gamma={self.gamma} learning_starts={self.learning_starts} learning_freq={self.learning_freq} \
        target_update_freq={self.target_update_freq} \
        num_eps={self.num_episodes} level={self.env.level_name()} ego={self.env.egocentric} frame_skip={self.env.frame_skip}\
        max_steps={self.max_steps}'

        # Writer for Tensorboard
        tb = SummaryWriter(comment=comment)


        # Initialize target q function and q function    
        Q = self.q_func(self.input_arg, self.num_actions).type(dtype)
        target_Q = self.q_func(self.input_arg, self.num_actions).type(dtype)

        if policy != None and target != None:
            Q.load_state_dict(policy)
            target_Q.load_state_dict(target)


        # Construct Q network optimizer function
        optimizer = self.optimizer_spec.constructor(Q.parameters(), **self.optimizer_spec.kwargs)

        # Construct the replay buffer
        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len)

        # The total loss over the training episodes
        total_loss = np.array([])
        total_reward = np.array([])
        total_steps = 0
        total_eps = 0

        while total_steps < self.max_steps:
        
            print("Starting episode", total_eps)
            ###############
            # RUN ENV     #
            ###############
            num_param_updates = 0
            mean_episode_reward = -float('nan')
            best_mean_episode_reward = -float('inf')
            last_obs, reward, game_status = self.env.start_state()
            game_status = self.status_to_bool(game_status)
            LOG_EVERY_N_STEPS = 10000
            
            # The loss and for the episode
            i_loss = np.array([])
            i_reward = np.array([])

            

            for t in count():
                #print("\tStep", t)
                ### Step the env and store the transition
                # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                last_idx = replay_buffer.store_frame(last_obs)
                # encode_recent_observation will take the latest observation
                # that you pushed into the buffer and compute the corresponding
                # input that should be given to a Q network by appending some
                # previous frames.
                recent_observations = replay_buffer.encode_recent_observation()
                
                action = None
                # Choose random action if not yet start learning
                if total_steps > self.learning_starts:
                    action = self.select_epilson_greedy_action(Q, recent_observations, total_steps)
                   
                else:
                    action = torch.IntTensor([[random.randrange(self.num_actions)]])
                    
                # Advance one step
                
                obs, reward, game_status = self.env.step(action)
                game_status = self.status_to_bool(game_status)
                #print(reward)
                tb.add_scalar("Reward per timestep", reward, total_steps)
                i_reward = np.append(i_reward, reward)
               
                # Store other info in replay memory
                replay_buffer.store_effect(last_idx, action, reward, game_status)
                # Resets the environment when reaching an episode boundary.
                if not game_status:
                    break
                    #obs, reward, game_status = env.start_state()
                    #game_status = self.status_to_bool(game_status)

                # The last observartion becomes the current observation
                last_obs = obs

                ### Perform experience replay and train the network.
                # Note that this is only done if the replay buffer contains enough samples
                # for us to learn something useful -- until then, the model will not be
                # initialized and random actions should be taken
                # 
                if (total_steps > self.learning_starts and
                        total_steps % self.learning_freq == 0 and
                        replay_buffer.can_sample(self.batch_size)):

                    print("\tTraining network")
                    # Use the replay buffer to sample a batch of transitions
                    # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                    # in which case there is no Q-value at the next state; at the end of an
                    # episode, only the current state reward contributes to the target
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(self.batch_size)
                    # Convert numpy nd_array to torch variables for calculation
                    obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
                    act_batch = Variable(torch.from_numpy(act_batch).long())
                    rew_batch = Variable(torch.from_numpy(rew_batch))
                    next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
                    not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

                    if USE_CUDA:
                        act_batch = act_batch.cuda()
                        rew_batch = rew_batch.cuda()

                       
                    # Compute current Q value, q_func takes only state and output value for every state-action pair
                    # We choose Q based on action taken.
                    current_Q_values = Q(obs_batch).gather(1, act_batch.view(-1, 1))
                    """
                    # DQN
                    # Compute next Q value based on which action gives max Q values
                    # Detach variable from the current graph since we don't want gradients for next Q to propagated
                    next_max_q = target_Q(next_obs_batch).detach().max(1)[0].view(-1, 1)
                    next_Q_values = not_done_mask.view(-1, 1) * next_max_q
                    """
                    next_argmax_action = Q(next_obs_batch).max(1)[1].view(-1, 1)
                    next_q = target_Q(next_obs_batch).detach().gather(1, next_argmax_action)
                    next_Q_values = not_done_mask.view(-1, 1) * next_q 
                    # Compute the target of the current Q values
                    target_Q_values = rew_batch.view(-1, 1) + (self.gamma * next_Q_values)
                    """
                    # Compute Bellman error
                    bellman_error = target_Q_values - current_Q_values
                    # clip the bellman error between [-1 , 1]
                    clipped_bellman_error = bellman_error.clamp(-1, 1)
                    # Note: clipped_bellman_delta * -1 will be right gradient
                    d_error = clipped_bellman_error * -1.0
                
                    # Clear previous gradients before backward pass
                    optimizer.zero_grad()
                    # run backward pass
                    current_Q_values.backward(d_error.data)
                    """
                    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

                    # Add loss per timestep to the tensorboard
                    tb.add_scalar("Loss per timestep", loss.item(), total_steps)

                    # Save the loss of this episode
                    i_loss = np.append(i_loss, loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    for param in Q.parameters():
                        param.grad.data.clamp(-1, 1)
                    # Perfom the update
                    optimizer.step()
                    num_param_updates += 1

                    # Periodically update the target network by Q network to target Q network
                    if num_param_updates % self.target_update_freq == 0:
                        target_Q.load_state_dict(Q.state_dict())

                total_steps += 1
            

            # Add the cumulative loss of the episode to tensorboard
            tb.add_scalar("Cumulative Loss per episode", i_loss.sum(), total_eps)
            tb.add_scalar("Cumulative Reward per episode", i_reward.sum(), total_eps)
            tb.add_scalar("Average reward per episode", i_reward.mean(), total_eps)
            tb.add_scalar("Average Loss per episode", i_loss.mean(), total_eps)
            

            # Append the average loss over episode i to the total loss
            print("Loss", i_loss)
            total_loss = np.append(total_loss, i_loss.mean())
            i_loss = np.array([])
            print("Episode", total_eps, "finished")
            total_eps += 1

        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.env.directory("./models/" + dt)
        self.env.save_parameters("./models/" + dt + "/hyperparameters.json")
        policy_path = "./models/" + dt + "/eps_" + str(self.max_steps) + "_" + self.env.level_path[15:-4] + "_" + "policy.pt"
        target_path = "./models/" + dt + "/eps_" + str(self.max_steps) + "_" + self.env.level_path[15:-4] + "_" + "target.pt"
        self.env.save_model(Q, policy_path)
        self.env.save_model(target_Q, target_path)
        self.env.zip_directory("./models/" + dt, dt)
        print("Training finished")
        
        
    def play(self, model):
        # Initialize target q function and q function
        Q = self.q_func(self.input_arg, self.num_actions).type(dtype)
        Q.load_state_dict(model)
        
        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len)



        for i_episode in range(self.num_episodes):
            print("Starting play episode", i_episode)
            ###############
            # RUN ENV     #
            ###############
            
            last_obs, reward, game_status = self.env.start_state()
            game_status = self.status_to_bool(game_status)

         
            # The loss for the episode
            i_loss = np.array([])

            for t in count():

                ### Step the env and store the transition
                # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                last_idx = replay_buffer.store_frame(last_obs)
                # encode_recent_observation will take the latest observation
                # that you pushed into the buffer and compute the corresponding
                # input that should be given to a Q network by appending some
                # previous frames.
                recent_observations = replay_buffer.encode_recent_observation()
                obs = torch.from_numpy(recent_observations).type(dtype).unsqueeze(0) / 255.0

                with torch.no_grad():
                    action = Q(Variable(obs, volatile=True)).data.max(1)[1].view(-1, 1).cpu()
              
                    
                # Advance one step
                
                obs, reward, game_status = self.env.step(action)
                game_status = self.status_to_bool(game_status)
                print(reward)
               
                # Store other info in replay memory
                replay_buffer.store_effect(last_idx, action, reward, game_status)
                # Resets the environment when reaching an episode boundary.
                if not game_status:
                    break
                    #obs, reward, game_status = env.start_state()
                    #game_status = self.status_to_bool(game_status)

                # The last observartion becomes the current observation
                last_obs = obs

                ### Perform experience replay and train the network.
                # Note that this is only done if the replay buffer contains enough samples
                # for us to learn something useful -- until then, the model will not be
                # initialized and random actions should be taken
                





    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    
   

        ### 4. Log progress and keep track of statistics
        
        episode_rewards = env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
print("Saved to %s" % 'statistics.pkl')

    """

import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from experience_replay import ReplayBuffer
from option_critic import to_tensor

from torch.utils.tensorboard import SummaryWriter



class OCAgent:

    def __init__(self, env, option_critic, num_episodes, max_history, max_steps_total, max_steps_ep, num_options, seed, learning_rate, batch_size, update_frequency, freeze_interval):
        self.env = env
        self.option_critic = option_critic
        self.option_critic_target = deepcopy(option_critic)
        self.num_episodes = num_episodes
        self.max_history = max_history
        self.max_steps_total = max_steps_total
        self.max_steps_ep = max_steps_ep
        self.num_options = num_options
        self.seed = seed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.freeze_interval = freeze_interval



    def train(self, args):
        
        # Writer for Tensorboard
        tb = SummaryWriter()
    
        optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=self.learning_rate)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        #env.seed(args.seed)

        buffer = ReplayBuffer(capacity=self.max_history, seed=self.seed)
        #logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

        steps = 0 ;
        #if args.switch_goal: print(f"Current goal {env.goal}")

        # The total loss over the training episodes
        total_loss = np.array([])
        total_reward = np.array([])
        total_steps = 0

        #while steps < self.max_steps_total:
        for i_episode in range(self.num_episodes):

            rewards = 0 ; option_lengths = {opt:[] for opt in range(self.num_options)}

            obs, reward, game_status = self.env.start_state()
            state = self.option_critic.get_state(to_tensor(obs))
            greedy_option  = self.option_critic.greedy_option(state)
            current_option = 0

            print(obs.shape)
            # The loss and for the episode
            i_loss = np.array([])
            i_reward = np.array([])


            # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
            # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
            # should be finedtuned (this is what we would hope).

            # Comment out for now as I want to verify that the algorithm works first
            # Later I can initialise the game with a new level after 1000 iterations
            """
            if args.switch_goal and logger.n_eps == 1000:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            'models/option_critic_{args.seed}_1k')
                env.switch_goal()
                print(f"New goal {env.goal}")

            if args.switch_goal and logger.n_eps > 2000:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            'models/option_critic_{args.seed}_2k')
                break
            """
            done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
            while not done: #and ep_steps < self.max_steps_ep:
                epsilon = self.option_critic.epsilon

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(self.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0
        
                action, logp, entropy = self.option_critic.get_action(state, current_option)

                # must convert int action to a tensor
                tensor_action = torch.IntTensor([[action]])
                next_obs, reward, game_status = self.env.step(tensor_action)
                done = not self.env.status_to_bool(game_status)

                i_reward = np.append(i_reward, reward)
                tb.add_scalar("Reward per timestep", reward, total_steps)

                if done:
                    total_loss = np.append(total_loss, i_loss.sum())
                    total_reward = np.append(total_reward, i_reward.sum())

                    tb.add_scalar("Cumulative Loss per episode", i_loss.sum(), i_episode)
                    tb.add_scalar("Cumulative Rewward per episode", i_reward.sum(), i_episode)
                    break

                buffer.push(obs, current_option, reward, next_obs, done)

                old_state = state
                state = self.option_critic.get_state(to_tensor(next_obs))

                option_termination, greedy_option = self.option_critic.predict_option_termination(state, current_option)
                rewards += reward

                


                actor_loss, critic_loss = None, None
                if len(buffer) > self.batch_size:
                    actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                        reward, done, next_obs, self.option_critic, self.option_critic_target, args)
                    loss = actor_loss

                    if steps % self.update_frequency == 0:
                        data_batch = buffer.sample(self.batch_size)
                        critic_loss = critic_loss_fn(self.option_critic, self.option_critic_target, data_batch, args)
                        loss += critic_loss
                    
                    tb.add_scalar("Loss per timestep", loss.item(), total_steps)
                    i_loss = np.append(i_loss, loss.item())
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if steps % self.freeze_interval == 0:
                        self.option_critic_target.load_state_dict(self.option_critic.state_dict())

                # update global steps etc
                steps += 1
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs
                total_steps += 1

                #logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

            #logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        policy_path = "./models/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "policy.pt"
        target_path = "./models/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "target.pt"
        self.env.save_model(self.option_critic.state_dict(), policy_path)
        self.env.save_model(self.option_critic_target.state_dict(), target_path)


    def play(self, model, args):
        
        self.option_critic.load_state_dict(model)
    
        

        buffer = ReplayBuffer(capacity=self.max_history, seed=self.seed)
        #logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

        steps = 0 ;
        #if args.switch_goal: print(f"Current goal {env.goal}")


        for i_episode in range(self.num_episodes):

            rewards = 0 ; option_lengths = {opt:[] for opt in range(self.num_options)}

            obs, reward, game_status = self.env.start_state()
            state = self.option_critic.get_state(to_tensor(obs))
            greedy_option  = self.option_critic.greedy_option(state)
            current_option = 0

            
            # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
            # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
            # should be finedtuned (this is what we would hope).

            # Comment out for now as I want to verify that the algorithm works first
            # Later I can initialise the game with a new level after 1000 iterations
            """
            if args.switch_goal and logger.n_eps == 1000:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            'models/option_critic_{args.seed}_1k')
                env.switch_goal()
                print(f"New goal {env.goal}")

            if args.switch_goal and logger.n_eps > 2000:
                torch.save({'model_params': option_critic.state_dict(),
                            'goal_state': env.goal},
                            'models/option_critic_{args.seed}_2k')
                break
            """
            done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
            while not done and ep_steps < self.max_steps_ep:
                epsilon = self.option_critic.epsilon

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(self.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0
        
                action, logp, entropy = self.option_critic.get_action(state, current_option)

                # must convert int action to a tensor
                tensor_action = torch.IntTensor([[action]])
                next_obs, reward, game_status = self.env.step(tensor_action)
                done = not self.env.status_to_bool(game_status)

                if done:
                    break

                buffer.push(obs, current_option, reward, next_obs, done)

                old_state = state
                state = self.option_critic.get_state(to_tensor(next_obs))

                option_termination, greedy_option = self.option_critic.predict_option_termination(state, current_option)
                rewards += reward


                # update global steps etc
                steps += 1
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs
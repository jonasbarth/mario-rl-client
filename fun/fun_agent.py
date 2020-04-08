import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.autograd import Variable

from fun import FeudalNet

from itertools import count

from torch.utils.tensorboard import SummaryWriter
import numpy as np


class FunAgent:

    def __init__(self, 
            env, 
            shared_model, 
            optimizer, 
            seed, 
            learning_rate, 
            num_steps, 
            gamma, 
            gamma_worker, 
            gamma_manager, 
            alpha, 
            tau_worker, 
            entropy_coef,
            value_manager_loss_coef,
            value_worker_loss_coef,
            max_grad_norm,
            num_episodes,
            max_steps
            ):
        self.env = env
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.seed = seed
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.gamma = gamma
        self.gamma_worker = gamma_worker
        self.gamma_manager = gamma_manager
        self.alpha = alpha
        self.tau_worker = tau_worker
        self.entropy_coef = entropy_coef
        self.value_manager_loss_coef = value_manager_loss_coef
        self.value_worker_loss_coef = value_worker_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_episodes = num_episodes
        self.model_name = "FuN"
        self.max_steps = max_steps

        

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                    shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad


    def train(self):
        """
        comment = f'_model={self.model_name} \
        gamma={self.gamma} gamma_worker={self.gamma_worker} gamma_manager={self.gamma_manager} \
        alpha={self.alpha} tau_worker={self.tau_worker} entropy_coef={self.entropy_coef} \
        value_manager_loss_coef={self.value_manager_loss_coef} value_worker_loss_coef={self.value_worker_loss_coef} \
        max_grad_norm={self.max_grad_norm} num_episodes={self.num_episodes} level={self.env.level_name()}'"""
        comment="_model=" + self.model_name + "_gamma=" + str(self.gamma) + "_gamma_worker=" + str(self.gamma_worker) \
            + "gamma_manager=" + str(self.gamma_manager) + "_alpha=" + str(self.alpha) + "_tau_worker=" + str(self.tau_worker) \
            + "_entropy_coef=" + str(self.entropy_coef) + "_value_manager_loss_coef=" + str(self.value_manager_loss_coef) \
          #  + "_value_worker_loss_coef=" + str(self.value_worker_loss_coef) + "_max_grad_norm=" + str(self.max_grad_norm) \
            + "_num_episodes=" + str(self.num_episodes) + "_level=" + self.env.level_name() + "_ego=" + str(self.env.egocentric) \
            + "_frame_skip=" + str(self.env.frame_skip) + "_max_steps=" + self.max_steps
        #seed = self.seed + self.rank
        torch.manual_seed(self.seed)

        tb = SummaryWriter(comment=comment)
        total_loss = np.array([])
        total_reward = np.array([])
        total_steps = 0

        ###observation space is numpy array of pixels
        ###action space is the numpy array of actions
        model = FeudalNet(self.env.observation_space, self.env.action_space, channel_first=True)

        if self.optimizer is None:
            print("no shared optimizer")
            self.optimizer = optim.Adam(self.shared_model.parameters(), lr=self.learning_rate)

        #writer = SummaryWriter(log_dir=log_dir)

        model.train()

        print(type(model))
        #obs = torch.from_numpy(obs)
        done = True

        episode_length = 0
        #for epoch in count():
        for i_episode in range(self.num_episodes):
            print("Episode", i_episode)
            obs, reward, game_status = self.env.start_state()
            # Sync with the shared model
            model.load_state_dict(self.shared_model.state_dict())

            if done:
                states = model.init_state(1)
            else:
                states = model.reset_states_grad(states)

            values_worker, values_manager = [], []
            log_probs = []
            rewards, intrinsic_rewards = [], []
            entropies = []  # regularisation
            manager_partial_loss = []

             # The loss and for the episode
            i_loss = np.array([])
            i_reward = np.array([])

            for t in count():
                #print("\tStep", t)
                episode_length += 1
                total_steps += 1
                value_worker, value_manager, action_probs, goal, nabla_dcos, states = model(obs.unsqueeze(0), states)
                m = Categorical(probs=action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                entropies.append(entropy)
                manager_partial_loss.append(nabla_dcos)

                obs, reward, game_status = self.env.step(torch.IntTensor([[action.item()]]))
                done = not self.env.status_to_bool(game_status)

                i_reward = np.append(i_reward, reward)
                tb.add_scalar("Reward per timestep", reward, total_steps)

                # I'm not using the max_episode length as Mario has a timeout
                #done = done or episode_length >= args.max_episode_length
                #reward = max(min(reward, 1), -1)
                intrinsic_reward = model._intrinsic_reward(states)
                intrinsic_reward = float(intrinsic_reward)  # TODO batch

                #plt_reward.add_value(None, intrinsic_reward, "Intrinsic reward")
                #plt_reward.add_value(None, reward, "Reward")
                #plt_reward.draw()

                #with self.lock:
                    #counter.value += 1

                if done:
                    episode_length = 0
                    break
                   # obs, reward, game_status = self.env.start_state()

                #obs = torch.from_numpy(obs)
                values_manager.append(value_manager)
                values_worker.append(value_worker)
                log_probs.append(log_prob)
                rewards.append(reward)
                intrinsic_rewards.append(intrinsic_reward)

                if done:
                    break

            tb.add_scalar("Cumulative Reward per episode", i_reward.sum(), i_episode)
            tb.add_scalar("Average reward per episode", i_reward.mean(), i_episode)

            R_worker = torch.zeros(1, 1)
            R_manager = torch.zeros(1, 1)
            if not done:
                value_worker, value_manager, _, _, _, _ = model(obs.unsqueeze(0), states)
                R_worker = value_worker.data
                R_manager = value_manager.data

            values_worker.append(Variable(R_worker))
            values_manager.append(Variable(R_manager))
            policy_loss = 0
            manager_loss = 0
            value_manager_loss = 0
            value_worker_loss = 0
            gae_worker = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                print("\t\tCalculating worker and manager loss")
                R_worker = self.gamma_worker * R_worker + rewards[i] + self.alpha * intrinsic_rewards[i]
                R_manager = self.gamma_manager * R_manager + rewards[i]
                advantage_worker = R_worker - values_worker[i]
                advantage_manager = R_manager - values_manager[i]
                value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
                value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

                # Generalized Advantage Estimation
                delta_t_worker = \
                    rewards[i] \
                    + self.alpha * intrinsic_rewards[i]\
                    + self.gamma_worker * values_worker[i + 1].data \
                    - values_worker[i].data
                gae_worker = gae_worker * self.gamma_worker * self.tau_worker + delta_t_worker

                policy_loss = policy_loss \
                    - log_probs[i] * gae_worker - self.entropy_coef * entropies[i]

                if (i + model.c) < len(rewards):
                    # TODO try padding the manager_partial_loss with end values (or zeros)
                    manager_loss = manager_loss \
                        - advantage_manager * manager_partial_loss[i + model.c]

            self.optimizer.zero_grad()

            total_loss = policy_loss \
                + manager_loss \
                + self.value_manager_loss_coef * value_manager_loss \
                + self.value_worker_loss_coef * value_worker_loss

            print("\tCalculating total loss")
            tb.add_scalar("Total loss per episode", total_loss, i_episode)
            total_loss.backward()
            """
            with lock:
                writer.add_scalars(
                    'data/loss' + str(rank),
                    {
                        'manager': float(manager_loss),
                        'worker': float(policy_loss),
                        'total': float(total_loss),
                    },
                    epoch
                )
                writer.add_scalars(
                    'data/value_loss' + str(rank),
                    {
                        'value_manager': float(value_manager_loss),
                        'value_worker': float(value_worker_loss),
                    },
                    epoch
                )
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.ensure_shared_grads(model, self.shared_model)
            self.optimizer.step()
            

        self.save()


    def train_steps(self):
        """
        comment = f'_model={self.model_name} \
        gamma={self.gamma} gamma_worker={self.gamma_worker} gamma_manager={self.gamma_manager} \
        alpha={self.alpha} tau_worker={self.tau_worker} entropy_coef={self.entropy_coef} \
        value_manager_loss_coef={self.value_manager_loss_coef} value_worker_loss_coef={self.value_worker_loss_coef} \
        max_grad_norm={self.max_grad_norm} num_episodes={self.num_episodes} level={self.env.level_name()}'"""
        comment="_model=" + self.model_name + "_gamma=" + str(self.gamma) + "_gamma_worker=" + str(self.gamma_worker) \
            + "gamma_manager=" + str(self.gamma_manager) + "_alpha=" + str(self.alpha) + "_tau_worker=" + str(self.tau_worker) \
            + "_entropy_coef=" + str(self.entropy_coef) + "_value_manager_loss_coef=" + str(self.value_manager_loss_coef) \
          #  + "_value_worker_loss_coef=" + str(self.value_worker_loss_coef) + "_max_grad_norm=" + str(self.max_grad_norm) \
            + "_num_episodes=" + str(self.num_episodes) + "_level=" + self.env.level_name() + "_ego=" + str(self.env.egocentric) \
            + "_frame_skip=" + str(self.env.frame_skip) + "_max_steps=" + self.max_steps
        #seed = self.seed + self.rank
        torch.manual_seed(self.seed)

        tb = SummaryWriter(comment=comment)
        total_loss = np.array([])
        total_reward = np.array([])
        total_steps = 0

        ###observation space is numpy array of pixels
        ###action space is the numpy array of actions
        model = FeudalNet(self.env.observation_space, self.env.action_space, channel_first=True)

        if self.optimizer is None:
            print("no shared optimizer")
            self.optimizer = optim.Adam(self.shared_model.parameters(), lr=self.learning_rate)

        #writer = SummaryWriter(log_dir=log_dir)

        model.train()

        print(type(model))
        #obs = torch.from_numpy(obs)
        done = True

        episode_length = 0
        #for epoch in count():
        for i_episode in range(self.num_episodes):
            print("Episode", i_episode)
            obs, reward, game_status = self.env.start_state()
            # Sync with the shared model
            model.load_state_dict(self.shared_model.state_dict())

            if done:
                states = model.init_state(1)
            else:
                states = model.reset_states_grad(states)

            values_worker, values_manager = [], []
            log_probs = []
            rewards, intrinsic_rewards = [], []
            entropies = []  # regularisation
            manager_partial_loss = []

             # The loss and for the episode
            i_loss = np.array([])
            i_reward = np.array([])

            for t in count():
                #print("\tStep", t)
                episode_length += 1
                total_steps += 1
                value_worker, value_manager, action_probs, goal, nabla_dcos, states = model(obs.unsqueeze(0), states)
                m = Categorical(probs=action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                entropies.append(entropy)
                manager_partial_loss.append(nabla_dcos)

                obs, reward, game_status = self.env.step(torch.IntTensor([[action.item()]]))
                done = not self.env.status_to_bool(game_status)

                i_reward = np.append(i_reward, reward)
                tb.add_scalar("Reward per timestep", reward, total_steps)

                # I'm not using the max_episode length as Mario has a timeout
                #done = done or episode_length >= args.max_episode_length
                #reward = max(min(reward, 1), -1)
                intrinsic_reward = model._intrinsic_reward(states)
                intrinsic_reward = float(intrinsic_reward)  # TODO batch

                #plt_reward.add_value(None, intrinsic_reward, "Intrinsic reward")
                #plt_reward.add_value(None, reward, "Reward")
                #plt_reward.draw()

                #with self.lock:
                    #counter.value += 1

                if done:
                    episode_length = 0
                    break
                   # obs, reward, game_status = self.env.start_state()

                #obs = torch.from_numpy(obs)
                values_manager.append(value_manager)
                values_worker.append(value_worker)
                log_probs.append(log_prob)
                rewards.append(reward)
                intrinsic_rewards.append(intrinsic_reward)

                if done:
                    break

            tb.add_scalar("Cumulative Reward per episode", i_reward.sum(), i_episode)
            tb.add_scalar("Average reward per episode", i_reward.mean(), i_episode)

            R_worker = torch.zeros(1, 1)
            R_manager = torch.zeros(1, 1)
            if not done:
                value_worker, value_manager, _, _, _, _ = model(obs.unsqueeze(0), states)
                R_worker = value_worker.data
                R_manager = value_manager.data

            values_worker.append(Variable(R_worker))
            values_manager.append(Variable(R_manager))
            policy_loss = 0
            manager_loss = 0
            value_manager_loss = 0
            value_worker_loss = 0
            gae_worker = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                print("\t\tCalculating worker and manager loss")
                R_worker = self.gamma_worker * R_worker + rewards[i] + self.alpha * intrinsic_rewards[i]
                R_manager = self.gamma_manager * R_manager + rewards[i]
                advantage_worker = R_worker - values_worker[i]
                advantage_manager = R_manager - values_manager[i]
                value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
                value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

                # Generalized Advantage Estimation
                delta_t_worker = \
                    rewards[i] \
                    + self.alpha * intrinsic_rewards[i]\
                    + self.gamma_worker * values_worker[i + 1].data \
                    - values_worker[i].data
                gae_worker = gae_worker * self.gamma_worker * self.tau_worker + delta_t_worker

                policy_loss = policy_loss \
                    - log_probs[i] * gae_worker - self.entropy_coef * entropies[i]

                if (i + model.c) < len(rewards):
                    # TODO try padding the manager_partial_loss with end values (or zeros)
                    manager_loss = manager_loss \
                        - advantage_manager * manager_partial_loss[i + model.c]

            self.optimizer.zero_grad()

            total_loss = policy_loss \
                + manager_loss \
                + self.value_manager_loss_coef * value_manager_loss \
                + self.value_worker_loss_coef * value_worker_loss

            print("\tCalculating total loss")
            tb.add_scalar("Total loss per episode", total_loss, i_episode)
            total_loss.backward()
            """
            with lock:
                writer.add_scalars(
                    'data/loss' + str(rank),
                    {
                        'manager': float(manager_loss),
                        'worker': float(policy_loss),
                        'total': float(total_loss),
                    },
                    epoch
                )
                writer.add_scalars(
                    'data/value_loss' + str(rank),
                    {
                        'value_manager': float(value_manager_loss),
                        'value_worker': float(value_worker_loss),
                    },
                    epoch
                )
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.ensure_shared_grads(model, self.shared_model)
            self.optimizer.step()
            

        self.save()


    def play(self):


        if self.optimizer is None:
            print("no shared optimizer")
            self.optimizer = optim.Adam(self.shared_model.parameters(), lr=self.learning_rate)


      
        #obs = torch.from_numpy(obs)
        done = True

        #for epoch in count():
        for i_episode in range(self.num_episodes):
            print("Episode", i_episode)
            last_obs, reward, game_status = self.env.start_state()
            # Sync with the shared model
            #model.load_state_dict(self.shared_model.state_dict())
            
            if done:
                states = self.shared_model.init_state(1)
            else:
                states = self.shared_model.reset_states_grad(states)
            
          
            for t in count():
               
                value_worker, value_manager, action_probs, goal, nabla_dcos, states = self.shared_model(last_obs.unsqueeze(0), states)
                m = Categorical(probs=action_probs)
                action = m.sample()


                obs, reward, game_status = self.env.step(torch.IntTensor([[action.item()]]))
                done = not self.env.status_to_bool(game_status)
                last_obs = obs

                if done:
                   
                    break
              
           

          
           



    def save(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.env.directory("./models/" + dt)
        self.env.save_parameters("./models/" + dt + "/hyperparameters.json")

        fun_path = "./models/" + dt + "/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "fun.pt"
        worker_path = "./models/" + dt + "/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "worker.pt"
        manager_path = "./models/" + dt + "/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "manager.pt"
        perception_path = "./models/" + dt + "/epochs_" + str(self.num_episodes) + "_" + self.env.level_path[15:-4] + "_" + "perception.pt"

        self.env.save_model(self.shared_model, fun_path)
        self.env.save_model(self.shared_model.worker, worker_path)
        self.env.save_model(self.shared_model.manager, manager_path)
        self.env.save_model(self.shared_model.perception, perception_path)
import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.autograd import Variable

from fun import FeudalNet

from itertools import count

from torch.utils.tensorboard import SummaryWriter


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
            max_grad_norm
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

        

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                    shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad


    def train(self):
        #seed = self.seed + self.rank
        torch.manual_seed(self.seed)

       
        ###observation space is numpy array of pixels
        ###action space is the numpy array of actions
        model = FeudalNet(self.env.observation_space, self.env.action_space, channel_first=True)

        if self.optimizer is None:
            print("no shared optimizer")
            self.optimizer = optim.Adam(self.shared_model.parameters(), lr=self.learning_rate)

        #writer = SummaryWriter(log_dir=log_dir)

        model.train()

        obs, reward, game_status = self.env.start_state()
        #obs = torch.from_numpy(obs)
        done = True

        episode_length = 0
        for epoch in count():
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

            for step in range(self.num_steps):
                episode_length += 1
                value_worker, value_manager, action_probs, goal, nabla_dcos, states = model(obs.unsqueeze(0), states)
                m = Categorical(probs=action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                entropies.append(entropy)
                manager_partial_loss.append(nabla_dcos)

                obs, reward, game_status = self.env.step(torch.IntTensor([[action.item()]]))
                done = not self.env.status_to_bool(game_status)

                # I'm not using the max_episode length as Mario has a timeout
                #done = done or episode_length >= args.max_episode_length
                reward = max(min(reward, 1), -1)
                intrinsic_reward = model._intrinsic_reward(states)
                intrinsic_reward = float(intrinsic_reward)  # TODO batch

                #plt_reward.add_value(None, intrinsic_reward, "Intrinsic reward")
                #plt_reward.add_value(None, reward, "Reward")
                #plt_reward.draw()

                #with self.lock:
                    #counter.value += 1

                if done:
                    episode_length = 0
                    obs, reward, game_status = self.env.start_state()

                #obs = torch.from_numpy(obs)
                values_manager.append(value_manager)
                values_worker.append(value_worker)
                log_probs.append(log_prob)
                rewards.append(reward)
                intrinsic_rewards.append(intrinsic_reward)

                if done:
                    break

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

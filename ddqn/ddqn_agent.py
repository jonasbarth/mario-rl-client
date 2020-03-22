import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import count
from PIL import Image
import os, errno



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        actions = self.head(x.view(x.size(0), -1))
        #print(actions)
        return actions
    
    













Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
#TODO: provide the screen height and width
#TODO: provide number of actions from action space

    
        





#print('Complete')
#env.render()
#env.close()
plt.ioff()
plt.show()


class DQNAgent:

    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, num_episodes, n_channels, screen_height, screen_width, game):
        self.n_actions = 8
        self.n_channels = n_channels
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.num_episodes = num_episodes
        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.steps_done = 0
        self.buffer = torch.randn((4,self.n_channels,screen_height,screen_width))
        self.game = game
        self.i_loss = np.array([])
       
        
        
    def select_action(self, state):
        #global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print("Best action")
                #print(state.shape)
                #print(self.policy_net(state))
                #print(self.policy_net(state).max(1))
                #print(self.policy_net(state).max(1)[1])
                #print(self.policy_net(state).max(1)[0].max(), self.policy_net(state).max(1)[1].max() )
                #print(self.policy_net(state).max(1)[1].view(1,4))
                return self.policy_net(state).max(1)[1].max().view(1, 1)
        else:
            print("Random action")
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def train(self):

        total_loss = np.array([])

        for i_episode in range(self.num_episodes):
            print("Training episode:", i_episode)
            # Initialize the environment and state
            # Send POST init to the server and get the state
            #env.reset()
            #last_screen = get_screen()
            #current_screen = get_screen()
            #state = current_screen - last_screen
            state, reward, game_status = self.game.start_state()

            # the loss for the episode
            i_loss = np.array([])
          
            
            print("Initialising start state")
            for t in count():
                # Select and perform an action
                #print("Selecting action")
               
                action = self.select_action(state)
                #print("Action selected")
                #print("Taking action")
                next_state, reward, status = self.game.step(action)
                #print("Took action", action, "and got reward", reward)
                print("Game Status", status)
                game_status = status
                
                #_, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                #last_screen = current_screen
                #current_screen = get_screen()
                #if not done:
                    #next_state = current_screen - last_screen
                #else:
                    #next_state = None

                # Store the transition in memory
                #print("Pushing transition to memory")
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                
                #print("Optimising model")
                # Perform one step of the optimization (on the target network)
                # Save the loss
                self.optimize_model()
                

                print(total_loss, i_loss)

                if game_status != "RUNNING":
                    self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            total_loss = np.append(total_loss, self.i_loss.mean())
            self.i_loss = np.array([])
            print("Episode", i_episode, "finished")

            
        print("Training finished")
        self.save_model("./ddqn/models/epochs_" + str(self.num_episodes) + "_" + self.game.level_path[15:-4] + "_")
        self.save_loss("./ddqn/loss/epochs_" + str(self.num_episodes) + "_" + self.game.level_path[15:-4] + "_")
        
        

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    
    def optimize_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
       
        
        #print(batch.state)
        #print(batch.action)
        #print(batch.reward)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool).repeat(4, 1)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).repeat(4, 1)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
       
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device).repeat(4, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        print(state_action_values.shape, expected_state_action_values.shape)

        # Reshape the state_action_values
        state_action_values = torch.reshape(state_action_values, (expected_state_action_values.shape[0], 1, expected_state_action_values.shape[1]))

        print(state_action_values.shape, expected_state_action_values.shape)
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
      
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.i_loss = np.append(self.i_loss, loss.item())



    def normalise_pixels(self, state):
        return

    
    
    def add_to_buffer(self, frames):
        
        #get the first of the frames
        current_frame = frames[self.n_frames-1]
        next_frame = None
        
        for frame in range(0, self.n_frames):
            next_frame = self.buffer[frame]
            self.buffer[frame] = current_frame
            current_frame = next_frame
            

    def save_model(self, path):
        policy_path = path + "policy.pt"
        target_path = path + "target.pt"
        self.delete(policy_path)
        self.delete(target_path)
        torch.save(self.policy_net.state_dict(), policy_path)
        print("Saved the policy net")
        torch.save(self.target_net.state_dict(), target_path)
        print("Saved the target net")
        
        parameter_path = path + "model_parameters.txt"
        self.delete(parameter_path)
        f = open(parameter_path, "w+")
        f.write("n_actions = " + str(self.n_actions))
        f.write("n_channels = " + str(self.n_channels))
        f.write("batch_size = " + str(self.BATCH_SIZE))
        f.write("gamma = " + str(self.GAMMA))
        f.write("eps_start = " + str(self.EPS_START))
        f.write("eps_end = " + str(self.EPS_END))
        f.write("eps_decay = " + str(self.EPS_DECAY))
        f.close()
        
        
    def load_model(self, path):
        policy_path = path + "policy.pt"
        target_path = path + "target.pt"
        
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.policy_net.eval()
        print("Loaded the policy net")
        
        self.target_net.load_state_dict(torch.load(target_path))
        self.target_net.eval()
        print("Loaded the target net")
        
        
    def delete(self, path):
        try:
            os.remove(path)
        except OSError as e: # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:
                pass
            
            
    def play(self, num_epochs):
        cum_rewards = {}
        for i_episode in range(num_epochs):
            print("Training episode:", i_episode)
            cum_rewards[i_episode] = 0
            # Initialize the environment and state
            # Send POST init to the server and get the state
            #env.reset()
            #last_screen = get_screen()
            #current_screen = get_screen()
            #state = current_screen - last_screen
            frames = self.client.init_env()[0]
           
            state = self.resize_frames(frames)
            
            print("Initialising start state")
            for t in count():
                # Select and perform an action
                
                action = self.policy_net(state).max(1)[1].max().view(1, 1)
                #print("Action selected")
                #print("Taking action")
                frames, reward, game_status = self.client.step(action)
                cum_rewards[i_episode] += reward
                #print("Took action", action, "and got reward", reward)
                print("Game Status", game_status)
                next_state = self.resize_frames(frames)
                
                #_, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                #last_screen = current_screen
                #current_screen = get_screen()
                #if not done:
                    #next_state = current_screen - last_screen
                #else:
                    #next_state = None

             

                # Move to the next state
                state = next_state
                
                if game_status != "RUNNING":
                    self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    break

            print("Episode", i_episode, "finished")
            
        print("Training finished")
        return cum_rewards


    def save_loss(self, loss):
        """
        Saves the loss of the entire training duration as a pandas dataframe.
        Loss is a numpy array of dimensions 1.
        """
        print("Saving loss")
        # add another dimension for episode values
        loss = np.expand_dims(loss, axis=1)

        # create a column with episode numbers
        episodes = np.arange(1, self.num_episodes + 1)
        episodes = np.expand_dims(episodes, axis=1)

        eps_loss = np.append(episodes, loss, axis=1)

        df = pd.DataFrame(data=eps_loss, columns=["episode", "huber_loss"])
        df.to_csv("loss.csv")
        print("Loss saved")



       



    




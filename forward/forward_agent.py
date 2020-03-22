import torch
import pandas as pd
import numpy as np
from itertools import count

class ForwardAgent:

    def __init__(self, game, num_episodes):
        self.game = game
        self.num_episodes = num_episodes


    def run(self):
       

        for i_episode in range(self.num_episodes):
            print("Training episode:", i_episode)
          
            state, reward, game_status = self.game.start_state()

         
            print("Initialising start state")
            for t in count():
                # Select and perform an action
                #print("Selecting action")
               
                action = torch.Tensor([[3]])
                #print("Action selected")
                #print("Taking action")
                next_state, reward, status = self.game.step(action)
                #print("Took action", action, "and got reward", reward)
                print("Game Status", status)
                print("Reward", reward)
                game_status = status
                


                # Move to the next state
                state = next_state

                if status != "RUNNING":
                    break
                
               
            print("Episode", i_episode, "finished")

     
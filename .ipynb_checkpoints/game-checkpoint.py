from py4j.java_gateway import JavaGateway, GatewayParameters
from py4j.java_collections import SetConverter, MapConverter, ListConverter
import numpy as np
import torch
import os, errno
import pandas as pd

class Game:
    
    def __init__(self, visible, scale, mario_state, timer, fps, level_path, preprocessor):
        self.visible = visible
        self.scale = scale
        self.mario_state = mario_state
        self.timer = timer
        self.fps = fps
        self.level_path = level_path
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25335))
        self.preprocessor = preprocessor
        self.right =  [False,True,False,False,False]
        self.right_run = [False,True,False,True,False]
        self.right_jump = [False,True,False,False,True]
        self.right_jump_run = [False,True,False,True,True]
        self.left = [True,False,False,False,False]
        self.left_run = [True,False,False,True,False]
        self.left_jump = [True,False,False,False,True]
        self.left_jump_run = [True,False,False,True,True]
        self.all_actions = [self.right, self.right_run, self.right_jump, self.right_jump_run, self.left, self.left_run, self.left_jump, self.left_jump_run]
        self.tensor_action_map =  {0: self.right, 1:self.right_run, 2:self.right_jump, 3:self.right_jump_run, 4:self.left, 5:self.left_run, 6:self.left_jump, 7:self.left_jump_run}


        
    def start_state(self):
        obs = self.gateway.entry_point.initGameEnv(self.visible, self.scale, self.mario_state, self.timer, self.fps, self.level_path, self.preprocessor.scaled_width, self.preprocessor.scaled_height)
        
        reward = obs.getValue()
        frames = obs.getByteArray()
        game_status = obs.getGameStatus()
     
        
        processed_frames = self.preprocessor.np_array(frames)
        
        return (self.to_tensor(processed_frames), reward, game_status)
        
    
    def step(self, action):
        action = self.tensor_to_action(action)
        java_list = ListConverter().convert(action, self.gateway._gateway_client)
        obs = self.gateway.entry_point.executeAction(java_list)
        
        reward = obs.getValue()
        game_status = obs.getGameStatus()

        if game_status == "RUNNING":
            frames = obs.getByteArray()
            processed_frames = self.preprocessor.np_array(frames) 
            return (self.to_tensor(processed_frames), reward, game_status)

        else:
            return (None, reward, game_status)
        
    def reset(self, level_path):
        self.level_path = level_path
        return self.start_state()

    def tensor_to_action(self, action):
        list_action = action.tolist()
        return self.tensor_action_map[list_action[0][0]]

    def to_tensor(self, np_frames):
        return torch.FloatTensor(np_frames)


    def save_model(self, model, path):
        self.delete(path)
        print("Saving model")
        torch.save(model.state_dict(), path)
        print("Model saved")


    def save_parameters(self, path, level, n_actions, n_channels, batch_size, gamma):
        self.delete(path)
        f = open(path, "w+")
        f.write("Level = " + self.game.level_path[15:-4])
        
        f.close()
        
        
    def load_model(self, path):
       
        print("Loading model")
        return torch.load(path)
       
        
        
    def delete(self, path):
        try:
            os.remove(path)
        except OSError as e: # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:
                pass


    def save_loss(self, path, loss):
        """
        Saves the loss of the entire training duration as a pandas dataframe.
        Loss is a numpy array of dimensions 1.
        """
        print("Saving loss")


        df = pd.DataFrame(data=loss, columns=["loss"])
        df.to_csv(path + "loss.csv")
        print("Loss saved")
        
        
        
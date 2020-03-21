from py4j.java_gateway import JavaGateway
from py4j.java_collections import SetConverter, MapConverter, ListConverter
import numpy as np
import torch

class Game:
    
    def __init__(self, visible, scale, mario_state, timer, fps, level_path, preprocessor):
        self.visible = visible
        self.scale = scale
        self.mario_state = mario_state
        self.timer = timer
        self.fps = fps
        self.level_path = level_path
        self.gateway = JavaGateway()
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
        frames = obs.getByteArray()
        game_status = obs.getGameStatus()

        processed_frames = self.preprocessor.np_array(frames)
        
        
        return (self.to_tensor(processed_frames), reward, game_status)

    def tensor_to_action(self, action):
        list_action = action.tolist()
        return self.tensor_action_map[list_action[0][0]]

    def to_tensor(self, np_frames):
        return torch.FloatTensor(np_frames)
        
        
        
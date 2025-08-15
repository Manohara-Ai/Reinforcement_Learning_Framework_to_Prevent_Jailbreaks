import torch
import random
import os
import numpy as np
from collections import deque
from .dqn_network import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(3076, 256, 3)

        model_path = r'model\Classic Model.pth'

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True),)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Initializing with default weights.")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, input_filter, action=None):
        state = input_filter.get_state(action)
        return np.array(state)
    
    def get_reward(self, output_filter, response):
        return 0

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

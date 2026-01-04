import torch
import torch.nn as nn

from Game import Game
from PUCTNet import PUCTNet

class TicTacToeNet(PUCTNet):
    def __init__(self, max_num_actions, num_players, device = "cpu"):
        super().__init__(max_num_actions, num_players) # must be executed first
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, padding = 1),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Flatten(), # (B, C * W * H)
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, max_num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.device = device
        self.to(device)

    def predict(self, players, states, possible_actions_list):
        # State: Bx3x3 (batch * height * width)
        # States from game are already in the perspective of current player, no need to do conversion here
        x = torch.tensor(states, dtype = torch.float32, device = self.device).unsqueeze(dim = 1) # transform state into valid input of shape BxCxHxW
        policy_logits, value_logits = self(x)

        # Scores provided to both players at current state in a zero-sum game
        players = torch.tensor(players).to(self.device)
        value = torch.tanh(value_logits)
        value = value.repeat(1, 2)
        value[players == 1, 0] = -value[players == 1, 0]
        value[players == 0, 1] = -value[players == 0, 1]
        
        return policy_logits, value # return raw policy logits; value must still be transformed

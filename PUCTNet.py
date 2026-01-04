import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import ReplayBuffer, ListStorage

from typehints import *

class PUCTNet(nn.Module):
    def __init__(self, max_num_actions: int, num_players: int):
        """
        Compulsory to implement.
        super().__init__(max_num_actions, num_players) must be called before defining feature_extractor, policy_head and value_head.
        """
        super().__init__()
        self.feature_extractor = nn.Identity()
        self.policy_head = nn.LazyLinear(max_num_actions) # dummy layers
        self.value_head = nn.LazyLinear(num_players) # dummy layers
    
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optional to implement.
        Input tensor has shape (B, ...), where B = batch_size

        RETURNS:
            Raw policy and value logits obtained from policy and value heads respectively.
        """
        x = self.feature_extractor(x)
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        return policy_logits, value_logits

    def predict(self, players: list[Player], states: list[State], possible_actions_list: list[Actions]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compulsory to implement. Instance attributes feature_extractor, policy_head and value_head must be defined.
        All inputs are batched.
        
        Value transformation function:
            tanh activation is recommended for 1-2 players. In this case, scores in Game.final_scores must be bounded by [-1, 1].
            sigmoid activation is recommended for more than 2 players. In this case, scores in Game.final_scores must be bounded by [0, 1].

        RETURNS:
            Raw policy logits and transformed value.
        """
        x = torch.tensor(states, dtype = torch.float32)
        policy_logits, value_logits = self(x)
        value = torch.tanh(value_logits)
        return policy_logits, value

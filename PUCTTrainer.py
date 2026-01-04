import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import ReplayBuffer, ListStorage
from copy import deepcopy
from tqdm.auto import tqdm

from typehints import *
from Game import Game
from PUCT import PUCT
from PUCTNet import PUCTNet

class PUCTTrainer:
    def __init__(self,
                 game: Game,
                 predictor: PUCTNet,
                 optimizer: optim.Optimizer,
                 policy_loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                 value_loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 c: float = 2):
        self.game = game # game at initial state
        self.predictor = predictor
        self.optimizer = optimizer
        self.c = c

        self.policy_loss_fn = policy_loss_fn
        self.value_loss_fn = value_loss_fn
        
        self.replay_buffer = ReplayBuffer(storage = ListStorage(buffer_size), batch_size = batch_size, collate_fn = self.collate_fn)
        self.batch_size = batch_size

    def collate_fn(self, batch: list[tuple[Player, State, Actions, torch.Tensor, Scores]]) -> tuple[list[Player], list[State], list[Actions], list[torch.Tensor], list[Scores]]:
        players, states, possible_actions_list, policies, scores_list = [], [], [], [], []
        for player, state, possible_actions, policy, scores in batch:
            players.append(player)
            states.append(state)
            possible_actions_list.append(possible_actions)
            policies.append(policy)
            scores_list.append(scores)
        return players, states, possible_actions_list, policies, scores_list
    
    def train_predictor(self, iterations: int, playouts_per_node: int, episodes: int, gradient_updates: int, disable_progress_bar: bool = False) -> None:
        """
        Update weights of predictor through self-play. PUCT is reset after each episode.
        
        PARAMETERS:
            iterations: number of iterations
            playouts_per_node: number of Monte Carlo playouts per node traversed in an episode
            episodes: number of episodes per iteration
            gradient_updates: number of mini-batches used to train predictor per iteration
        """
        with tqdm(range(iterations), desc = "Training Predictor", disable = disable_progress_bar) as progress_bar:
            for iteration in progress_bar:
                for _ in tqdm(range(episodes), desc = "Self-Play", disable = disable_progress_bar, leave = False):
                    self.predictor.eval()
                    tree = PUCT(self.game, self.predictor, c = self.c)
                    experiences, scores = tree.self_play(playouts_per_node = playouts_per_node, training = True, render = False)
                    for player, state, possible_actions, (action, visit_count) in experiences:
                        assert len(possible_actions) > 0, "No actions available but game is not terminated yet! Please provide a do-nothing action if this is intended!"
                        true_policy = torch.zeros(tree.game.max_num_actions(), dtype = torch.float32)
                        true_policy[action] = torch.tensor(visit_count, dtype = torch.float32)
                        total_visit_count = true_policy.sum()
                        if total_visit_count > 0:
                            true_policy /= total_visit_count
                        
                        experience = (player, state, possible_actions, true_policy, scores)
                        self.replay_buffer.add(experience)

                avg_policy_loss, avg_value_loss, avg_loss = 0, 0, 0
                for _ in tqdm(range(gradient_updates), desc = "Updating Weights", disable = disable_progress_bar, leave = False):
                    self.predictor.train()
                    self.optimizer.zero_grad()
                    players, states, possible_actions_list, true_policies, true_values = self.replay_buffer.sample()
                    pred_policies_logits, pred_values = self.predictor.predict(players, states, deepcopy(possible_actions_list)) # raw logits needed for policy for CE loss, which is used in most cases
                    true_policies = torch.stack(true_policies).to(pred_policies_logits.device)
                    true_values = torch.tensor(true_values, dtype = torch.float32, device = pred_values.device)
        
                    masked_policy_logits = torch.full_like(pred_policies_logits, -1e9) # mask invalid actions with -1e9 (to prevent NaN/inf from appearing in policy loss)
                    for batch, actions in enumerate(possible_actions_list):
                        masked_policy_logits[batch, actions] = pred_policies_logits[batch, actions]
        
                    policy_loss = self.policy_loss_fn(masked_policy_logits, true_policies)
                    value_loss = self.value_loss_fn(pred_values, true_values)
                    loss = policy_loss + value_loss
                    loss.backward()
                    self.optimizer.step()

                    avg_policy_loss += policy_loss.item()
                    avg_value_loss += value_loss.item()
                    avg_loss += loss.item()

                avg_policy_loss, avg_value_loss, avg_loss = avg_policy_loss / gradient_updates, avg_value_loss / gradient_updates, avg_loss / gradient_updates
                print(f"[Iteration {iteration + 1}] Samples in Replay Buffer: {len(self.replay_buffer)} - Policy Loss: {avg_policy_loss:.3g} - Value Loss: {avg_value_loss:.3g} - Total Loss: {avg_loss:.3g}")

            progress_bar.close()

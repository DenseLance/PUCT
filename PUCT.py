from __future__ import annotations
import torch
import random
import logging
from tqdm.auto import tqdm
from typing import Optional

from typehints import *
from Game import Game
from PUCTNet import PUCTNet

class PUCTNode:
    def __init__(self, player: Player, state: State, parent_node: Optional[PUCTNode] = None, policy_prior: float = 1, c: float = 2):
        self.player = player
        self.state = state

        self.parent_node = parent_node
        self.children = dict()

        self.is_expanded = False
        self.has_outcome = False

        self.w = 0
        self.n = 0
        self.policy_prior = policy_prior # policy prior (predicted probability to bias exploration toward promising moves)
        self.value = None

        self.c = c

    def eval(self, exploration: bool = True) -> float:
        return self.Q() + self.U() if exploration else self.n

    def Q(self) -> float: # exploitation term
        return self.w / self.n if self.n > 0 else 0

    def U(self) -> float: # exploration term
        return self.c * self.policy_prior * pow(self.parent_node.n, 0.5) / (1 + self.n)

    def V(self) -> float: # value (outcome predicted by predictor if state is non-terminal/actual outcome if state is terminal)
        return self.value

    def add_child(self, next_player: Player, next_state: State, action: Action, policy_prior: float) -> None:
        if action not in self.children:
            self.children[action] = PUCTNode(next_player, next_state, parent_node = self, policy_prior = policy_prior, c = self.c)

    def choose_best_action(self, exploration: bool = True) -> Action:
        return max(self.children, key = lambda action: self.children[action].eval(exploration = exploration))

    def sample_action(self) -> Action:
        return random.choices(list(self.children.keys()), weights = [self.children[action].eval(exploration = False) for action in self.children], k = 1)[0] # sample using distribution of visit counts

class PUCT:
    def __init__(self, game: Game, predictor: PUCTNet, root: Optional[PUCTNode] = None, c: float = 2, epsilon: float = 0.25, alpha: float = 0.03):
        assert c >= 0, "Exploration parameter must be greater than or equal to 0!"
        assert 0 <= epsilon <= 1, "Scaling factor for dirichlet noise, epsilon, must be between 0 and 1 inclusive!"
        assert alpha > 0, "alpha parameter for dirichlet noise must be greater than 0!"
        self.game = game # game at current state
        self.predictor = predictor
        self.c = c # c = 2 is recommended if scores are bounded by [-1, 1]; otherwise c = sqrt(2) is recommended if scores are bounded by [0, 1]
        self.epsilon = epsilon
        self.alpha = alpha
        self.root = PUCTNode(self.game.current_player(), self.game.get_state(), parent_node = None, policy_prior = 1, c = self.c) if root is None else root
        
    def selection(self, node: PUCTNode) -> tuple[Game, list[PUCTNode]]:
        path = [node]
        ongoing_game = self.game.copy()
        while path[-1].is_expanded is True and ongoing_game.has_outcome() is False: # loop if node exists and does not refer to terminal state
            action = path[-1].choose_best_action(exploration = True)
            path.append(path[-1].children[action])
            ongoing_game.take_action(action)
            assert path[-1].state == ongoing_game.get_state(), f"State transition is not deterministic! Check your game environment! {path[-1].state} != {ongoing_game.get_state()}"
        return ongoing_game, path

    def expansion(self, ongoing_game: Game, path: list[PUCTNode]) -> tuple[Game, list[PUCTNode]]:
        if ongoing_game.has_outcome() is True:
            path[-1].has_outcome = True
            return ongoing_game, path

        assert not path[-1].is_expanded, "Node is already expanded!" # if this happens, please submit an issue on Github as this is an implementation bug

        with torch.no_grad():
            possible_actions = ongoing_game.possible_actions()
            policy_logits, value = self.predictor.predict([ongoing_game.current_player()],
                                                          [ongoing_game.get_state()],
                                                          [possible_actions])
            policy_logits, value = policy_logits.squeeze(dim = 0), value.squeeze(dim = 0) # unwrap batch-wise
            
            masked_policy_logits = torch.full_like(policy_logits, -torch.inf)
            masked_policy_logits[possible_actions] = policy_logits[possible_actions] # mask invalid actions with -inf
            masked_policy = torch.softmax(masked_policy_logits, dim = 0)
            dirichlet_noise = torch.distributions.Dirichlet(torch.full_like(masked_policy[possible_actions], self.alpha)).sample()
            masked_policy[possible_actions] = (1 - self.epsilon) * masked_policy[possible_actions] + self.epsilon * dirichlet_noise

        for action in ongoing_game.possible_actions():
            expanded_game = ongoing_game.copy()
            expanded_game.take_action(action)
            path[-1].add_child(expanded_game.current_player(), expanded_game.get_state(), action, masked_policy[action].item())

        assert len(path[-1].children) > 0, "No actions available but node is not terminal state! Please provide a do-nothing action if this is intended!"

        path[-1].is_expanded = True
        path[-1].value = value.tolist()
        
        return ongoing_game, path

    def backpropagation(self, ongoing_game: Game, path: list[PUCTNode]) -> None:
        if ongoing_game.has_outcome() is True:
            scores = ongoing_game.final_scores() # use true utility for each player if state is terminal
            path[-1].has_outcome = True
        else:
            scores = path[-1].value # use estimated/predicted utility for each player if state is non-terminal

        for i in range(1, len(path)):
            path[i].w += scores[path[i - 1].player]
            path[i].n += 1

    def explore(self, iterations: int = 1) -> None:
        for _ in range(iterations):
            ongoing_game, path = self.selection(self.root)
            ongoing_game, path = self.expansion(ongoing_game, path)
            self.backpropagation(ongoing_game, path)

    def exploit(self, training: bool = False) -> tuple[Game, list[PUCTNode]]:
        game = self.game.copy()
        if training:
            action = self.root.sample_action()
        else:
            action = self.root.choose_best_action(exploration = False)
        child = self.root.children[action]
        game.take_action(action)
        return game, child

    def self_play(self, playouts_per_node: int = 1, training: bool = False, render: bool = False) -> tuple[list[list[Player, State, Actions, tuple[Actions, list[int]]]], Scores]:
        self.predictor.eval()
        tree, game = self, self.game.copy()
        experiences = []
        if render:
            game.render()
        while game.has_outcome() is False:
            tree.explore(iterations = playouts_per_node)
            experiences.append([
                game.current_player(),
                game.get_state(),
                game.possible_actions(),
                (list(tree.root.children.keys()), [tree.root.children[action].n for action in tree.root.children])
            ]) # player, state, possible_actions, (action, visit_count)
            game, child = tree.exploit(training = training)
            tree = PUCT(game, tree.predictor, root = child, c = tree.c) # re-use subtree and its statistics; this works since predictor weights are not updated
            if render:
                game.render()
        return experiences, game.final_scores()
    
    def human_play(self, human_player: Player, playouts_per_node: int = 1, training: bool = False, render: bool = False) -> tuple[list[list[Player, State, Actions, tuple[Actions, list[int]]]], Scores]:
        self.predictor.eval()
        tree, game = self, self.game.copy()
        experiences = []
        if render:
            game.render()
        while game.has_outcome() is False:
            tree.explore(iterations = playouts_per_node)
            experiences.append([
                game.current_player(),
                game.get_state(),
                game.possible_actions(),
                (list(tree.root.children.keys()), [tree.root.children[action].n for action in tree.root.children])
            ]) # player, state, possible_actions, (action, visit_count)
            if game.current_player() == human_player:
                actions = game.possible_actions()
                print("Possible actions:", actions)
                action = int(input("> "))
                assert action in actions, "You have chosen an action that is not permissible!"
                
                child = tree.root.children[action] if action in tree.root.children else None # create new tree if child node for the action is not found 
                game.take_action(action)
            else:
                game, child = tree.exploit(training = training)

            tree = PUCT(game, tree.predictor, root = child, c = self.c) # re-use subtree and its statistics; this works since predictor weights are not updated
            if render:
                game.render()
        return experiences, game.final_scores()

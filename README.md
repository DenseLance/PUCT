# Predictor + UCT (PUCT)

## Installation

There are plans to port this project over to `mcts-simple` later this year. Currently, you may use git clone or fork this repository.

## Dependencies

* `tqdm`

* `torch`

* `torchrl`

## Usage

The code found in this library is simplified and made easy to understand for research purposes.

#### Typehints

Our library introduces additional typehints for state, action, player and scores for better readability.

```python
State = tuple[int | float | tuple[State]]
Action = int
Actions = tuple[Action] | list[Action]
Player = int
Score = float
Scores = tuple[Score] | list[Score]
```

#### Game Environment

A game environment that inherits from the `Game` class that must be defined for all MCTS variants.

| Method                                | Description                                                                                                                                                                                                                          | Mandatory |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| `copy() -> Game`                      | Returns a deep copy of the current game, including its internal state. If RNG is used, use the `random.Random` instead of querying the random module directly so that environment remains deterministic.                             | ❌         |
| `render() -> None`                    | Displays the visual representation of the current state of the game.                                                                                                                                                                 | ❌         |
| `get_state() -> State`                | Returns the current state of the game, viewed from the **perspective of the current player**. State provided should be immutable and hashable if possible.                                                                           | ✔️        |
| `num_players() -> int`                | Returns the number of players, which must remain constant over an episode.                                                                                                                                                           | ✔️        |
| `current_player() -> Player`          | Returns the player that is taking an action for the current state. Players are labelled from 0 to `num_players` - 1.                                                                                                                 | ✔️        |
| `max_num_actions() -> int`            | Returns the maximum number of actions, which must remain constant over an episode.                                                                                                                                                   | ✔️        |
| `possible_actions() -> Actions`       | Returns a list of all possible actions that can be taken by current player. Actions are labelled from 0 to `max_num_actions` - 1. List of possible actions should be made immutable where possible, but does not need to be ordered. | ✔️        |
| `take_action(action: Action) -> None` | Make the current player take action in the current state.                                                                                                                                                                            | ✔️        |
| `has_outcome() -> bool`               | Returns `True` if state is terminal, else `False`.                                                                                                                                                                                   | ✔️        |
| `final_scores() -> Scores`            | Returns a list of scores achieved by each player at terminal state. List of scores is indexed by the player's ID.                                                                                                                    | ✔️        |

#### Predictor (for PUCT)

A predictor that inherits from the `PUCTNet` class that must be defined for PUCT.

| Method                                                                                                                           | Description                                                                                                                                                      | Mandatory |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `__init__(max_num_actions: int, num_players: int)`                                                                               | Initializes the predictor. Attributes `feature_extractor`, `policy_head` and `value_head` must be defined.                                                       | ✔️        |
| `__call__(x: Tensor) -> tuple[Tensor, Tensor]`                                                                                   | Forward pass. Called by `PUCTNet.predict`. Returns policy and value logits.                                                                                      | ❌         |
| `predict(players: list[Player], states: list[State], possible_actions_list: list[Actions]) -> tuple[torch.Tensor, torch.Tensor]` | Predicts policy logits and values for each player, given the player, state and possible actions from that state. Player, state and possible actions are batched. | ✔️        |

#### MCTS/UCT

Performing MCTS/UCT playouts is simple and can be done in 3 lines. In this case, `YourGame()` is the game environment defined by you.

```python
game = YourGame()
tree = MCTS(game)
tree.self_play(playouts_per_node = 1000, render = True)
```

After `self_play` or `human_play`, node statistics (e.g. visit counts) will be preserved and can be accessed. However, statistics for the initial state will never be updated for our implementation.

#### PUCT

PUCT requires a predictor to be trained first. This can be done via self-play as shown below.

```python
game = YourGame()
predictor = GameNet(game.max_num_actions(), game.num_players())
optimizer = optim.Adam(predictor.parameters(), lr = 1e-4, weight_decay = 1e-4)
trainer = PUCTTrainer(game, predictor, optimizer, buffer_size = 100000, batch_size = 32)
trainer.train_predictor(iterations = 10, playouts_per_node = 1000, episodes = 100, gradient_updates = 200)
```

After training , we can view its performance via `self_play`or  `human_play`. It is recommended to use the same `playouts_per_node` for both training and evaluation.

```python
tree = PUCT(game)
tree.self_play(playouts_per_node = 1000, training = False, render = True)
```

You may refer to the examples provided at <a href = "https://github.com/DenseLance/PUCT">DenseLance/PUCT</a> on how to create a game environment and predictor (if PUCT is used). 

## Predictor + UCT (PUCT)

PUCT is a variant of UCT which introduces a probability prior predicted by a neural network into the equation. Our implementation closely follows the formulation proposed by AlphaZero, with some tweaks. Specifically, our implementation works for any game with <u>any number of players</u>.

#### PUCT Stages

1. Selection
   - **Exploration:** Traverse through the search tree from the current node to a leaf node that has not gone through the expansion stage yet. Intermediate nodes are sampled from the distribution of predicted probability priors. Dirichlet noise is added to these probability priors to promote greater exploration.
   - **Exploitation:** At current node, choose the child node with the highest visit count.
2. Expansion
   - Unlike MCTS and UCT, PUCT does not perform the simulation stage. Instead, if the current state is non terminal, the predictor's value head is used to predict its value.
   - If state is non-terminal, at least one child node is created for the leaf node. The predictor's policy head provides a probability prior for each child node that is used in the modified UCB1 formula.
3. Backpropagation
   - True scores obtained at terminal state or predicted scores obtained at non-terminal state are used to update Q-values for traversed nodes.

## Citations

[1]    Auer, P. (2002). Using confidence bounds for exploitation-exploration trade-offs. *Journal of machine learning research*, *3*(Nov), 397-422.

[2]    Rosin, C. D. (2011). Multi-armed bandits with episode context. *Annals of Mathematics and Artificial Intelligence*, *61*(3), 203-230.

[3]    Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den 
Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go 
with deep neural networks and tree search. *nature*, *529*(7587), 484-489.

[4]    Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., 
Guez, A., ... & Hassabis, D. (2017). Mastering chess and shogi by 
self-play with a general reinforcement learning algorithm. *arXiv preprint arXiv:1712.01815*.

[5]    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., 
Schmitt, S., ... & Silver, D. (2020). Mastering atari, go, chess and
 shogi by planning with a learned model. *Nature*, *588*(7839), 604-609.

## To Do

- [ ] PUCT for non-deterministic games.

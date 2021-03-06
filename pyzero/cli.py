#!/usr/bin/env python

# global imports

import argparse
import functools
import glob
import logging
import os
import tensorflow as tf
import sys

from datetime import date, datetime
from typing import List


# local imports

from action import Action
from action_history import ActionHistory
from environment import Environment
from game import Game
from helpers import MinMaxStats
from muzero_config import MuZeroConfig
from network import Network
from network_output import NetworkOutput
from node import Node
from player import Player
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage


def make_board_game_config(
    action_space_size: int, max_moves: int,
    dirichlet_alpha: float,
    lr_init: float) -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):

        if num_moves < 30:
          return 1.0
        else:
          return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1)
    )


def make_go_config() -> MuZeroConfig:

    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01
    )


def make_chess_config() -> MuZeroConfig:

    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1
    )


def make_shogi_config() -> MuZeroConfig:

    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1
    )


def make_atari_config() -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):

        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature
    )


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):

    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()



##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: MuZeroConfig,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer):

    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(
    config: MuZeroConfig,
    network: Network) -> Game:

    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)

        current_observation = game.make_image(-1)

        expand_node(
            root, game.to_play(), game.legal_actions(),
            network.initial_inference(current_observation))

        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: MuZeroConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network):


    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                    history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount, min_max_stats)


def select_action(
    config: MuZeroConfig,
    num_moves: int,
    node: Node,
    network: Network):

    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())

    _, action = softmax_sample(visit_counts, t)

    return action


# Select the child with the highest UCB score.
def select_child(
    config: MuZeroConfig,
    node: Node,
    min_max_stats: MinMaxStats):

    _, action, child = max((ucb_score(config, node, child, min_max_stats),action,child) for action, child in node.children.items())

    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: MuZeroConfig,
    parent: Node, child: Node,
    min_max_stats: MinMaxStats) -> float:


    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        value_score = child.reward + config.discount * min_max_stats.normalize(child.value())
    else:
        value_score = 0

    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
    node: Node,
    to_play: Player,
    actions: List[Action],
    network_output: NetworkOutput):

    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())

    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node],
    value: float,
    to_play: Player,
    discount: float, min_max_stats: MinMaxStats):

    for node in reversed(search_path):

        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################


##################################
####### Part 2: Training #########


def train_network(
    config: MuZeroConfig,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer):

    network = Network()
    learning_rate = config.lr_init * config.lr_decay_rate**(tf.train.get_global_step() / config.lr_decay_steps)
    optimizer = tf.train.MomentumOptimizer(learning_rate, config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)

    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(
    optimizer: tf.compat.v1.train.Optimizer,
    network: Network,
    batch,
    weight_decay: float):

    loss = 0

    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
          value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
          predictions.append((1.0 / len(actions), value, reward, policy_logits))

          hidden_state = scale_gradient(hidden_state, 0.5)

        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            l = (
              scalar_loss(value, target_value) +
              scalar_loss(reward, target_reward) +
              tf.nn.softmax_cross_entropy_with_logits(
                  logits=policy_logits, labels=target_policy))

            loss += scale_gradient(l, gradient_scale)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return -1

######### End Training ###########
##################################

# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()




def train_cli(args):
  logging.info('Starting up')
  logging.info(f'Config: {make_atari_config()}')
  logging.info('Finished executing')



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def noop(args=None):
    logging.error('Not implemented function is called')


def args_switch(args):
    fn = switcher.get(args.func, noop)
    logging.info('fn: %s', fn)
    return fn(args)


switcher = {
    'train': train_cli,
}


def main():

    try:

        exe_path = os.path.dirname(os.path.realpath(sys.argv[0]))

        today = str(date.today())

        log_handlers = []
        log_handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-4s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=log_handlers)

        parser = argparse.ArgumentParser(prog='imgen')

        subparsers = parser.add_subparsers()

        train = subparsers.add_parser('train')

        train.add_argument_group('train', '')
        train.set_defaults(func='train')
        train.add_argument('--image-input-folder', action='store', required=True)
        train.add_argument('--resolution', action='store', type=int, required=True)
        train.add_argument('--epochs', action='store', type=int, required=True)
        train.add_argument('--batch-size', action='store', type=int, required=True)
        train.add_argument('--gpu', action='store', type=str2bool, default=False, required=True)
        train.add_argument('--nets-input-folder', action='store', required=False, default=None)

        args = parser.parse_args()

        logging.info('ARGS: %s', args)

        if not any(vars(args).values()):
            logging.error("No parameter were passed")
            parser.print_help()
            exit(1)
        else:
            args_switch(args)

    except KeyboardInterrupt:
        logging.info("Ctrl+c was pressed, exiting...")
        exit(0)
    except Exception as e:
        logging.error('Exception caught in main')
        logging.exception('Exception caught: %s', e)
        exit(1)
    finally:
        logging.info("Quitting...")

if __name__ == '__main__':
    exit(main())

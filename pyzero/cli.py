#!/usr/bin/env python

# global imports

import argparse
import functools
import glob
import logging
import os
import sys

from datetime import date, datetime


# local imports

from action import Action
from action_history import ActionHistory
from environment import Environment
from game import Game
from helpers import MinMaxStats
from muzero_config import MuZeroConfig
from node import Node
from player import Player
from replay_buffer import ReplayBuffer


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

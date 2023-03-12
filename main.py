import os

from src.networks.simple_network import SimpleNetwork
from src.user_interface.ui import GameUI
from src.games.space_invaders import SpaceInvaders
from src.algorithms.PPO import PPO


if __name__ == '__main__':
    # ppo = PPO()
    # ppo.load('test_model')
    # ui = GameUI(SpaceInvaders, model = ppo)
    # ui.run()
    print('Starting Training')
    ppo = PPO()
    print('Training PPO')
    print('######################')
    ppo.train(SpaceInvaders, SimpleNetwork, save_location = f'{os.environ["PBS_O_WORKDIR"]}/ppo_2')
    ppo.train(SpaceInvaders, SimpleNetwork, save_location = 'ppo_1')

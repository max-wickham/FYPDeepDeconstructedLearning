import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from src.networks.simple_network import SimpleActorNetwork, SimpleCriticNetwork
from src.user_interface.ui import GameUI
from src.games.space_invaders import SpaceInvaders
from src.algorithms.PPO import PPO


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # ppo = PPO()
    # ppo.load('ppo_3')
    # ui = GameUI(SpaceInvaders, model = ppo)
    # ui.run()
    print('Starting Training')
    ppo = PPO()
    print('Training PPO')
    print('######################')
    ppo.train(SpaceInvaders, SimpleActorNetwork, SimpleCriticNetwork,
    save_location = f'{os.environ["PBS_O_WORKDIR"]}/ppo_9',
    stats_location= f'{os.environ["PBS_O_WORKDIR"]}/ppo_9_stats')
    # ppo.train(SpaceInvaders, SimpleNetwork, save_location = 'ppo_5')

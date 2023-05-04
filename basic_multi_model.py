import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from src.networks.simple_network import SimpleActorNetwork, SimpleCriticNetwork, SimpleDDQNNetwork
from src.user_interface.ui import GameUI
from src.games.space_invaders_large import SpaceInvadersLarge
from src.algorithms.PPO import PPO
from src.algorithms.DDQN import DDQN
from src.algorithms.MultiModelPPO import MultiModelPPO



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ############ PPO Train
    print('Starting Training')
    ppo = MultiModelPPO()
    print('Training PPO')
    print('######################')
    ppo.train(SpaceInvadersLarge, SimpleActorNetwork, SimpleCriticNetwork,
        save_location = f'{os.environ["PBS_O_WORKDIR"]}/models/ppo_multi_simple',
        stats_location= f'{os.environ["PBS_O_WORKDIR"]}/models/ppo_multi_simple_stats')

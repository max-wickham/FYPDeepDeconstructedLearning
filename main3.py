import os
import multiprocessing

from src.algorithms.MultiModelPPO2_2 import MultiModelPPO2
from src.algorithms.MultiModelPPO3 import MultiModelPPO3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.algorithms.MultiModelDDQN import MultiModelDDQN
from src.networks.large_network import LargeSimpleActorNetwork, LargeSimpleCriticNetwork
from src.user_interface.ui import GameUI
from src.games.space_invaders import SpaceInvaders
from src.games.space_invaders_large import SpaceInvadersLarge
from src.algorithms.PPO import PPO
from src.algorithms.DDQN import DDQN
from src.algorithms.MultiModelPPO import MultiModelPPO



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    ########### PPO Play
    # ppo = PPO()
    # ppo.load('models/ppo_8')
    # ui = GameUI(SpaceInvaders, model = ppo, record=False)
    # # ui = GameUI(SpaceInvaders, record=False)
    # ui.run()
    # # # ui.playback()


    ############ PPO Train
    # print('Starting Training')
    # ppo = PPO()
    # print('Training PPO')
    # print('######################')
    # ppo.train(SpaceInvaders, SimpleActorNetwork, SimpleCriticNetwork,
    #     save_location = f'{os.environ["PBS_O_WORKDIR"]}/models/ppo_large_simple',
    #     stats_location= f'{os.environ["PBS_O_WORKDIR"]}/models/ppo_large_simple_stats')
    # ppo.train(SpaceInvaders, SimpleActorNetwork,SimpleCriticNetwork, save_location = 'ppo_3')

    ############ DDQN Train
    # ddqn = DDQN()
    # ddqn.train(SpaceInvaders, SimpleDDQNNetwork, save_location = 'ddqn_1')

    ############ DDQN Run
    # ddqn = DDQN()
    # ddqn.load('ddqn_1')
    # ui = GameUI(SpaceInvaders, model = ddqn, record=False)
    # ui.run()
    # # ui.playback()


    ########### MultiModelPPO Train
    print('Starting Training')
    MultiModelPPO2.train(SpaceInvadersLarge, LargeSimpleActorNetwork, LargeSimpleCriticNetwork,
        save_location = f'{os.environ["PBS_O_WORKDIR"]}/models/large_ppo2',
        stats_location= f'{os.environ["PBS_O_WORKDIR"]}/models/large_ppo2_stats')

    # # # ########### MultiModelPPO Play
    # multi_model_ppo = MultiModelPPO()
    # multi_model_ppo.load('multi_model_ppo_1')
    # # ui = GameUI(SpaceInvadersLarge, model = multi_model_ppo, record=False)
    # ui = GameUI(SpaceInvaders, model = multi_model_ppo, record=False)
    # ui.run()
    # # # ui.playback()


    ############## MultiModelDDQN Train
    # MultiModelDDQN.train(SpaceInvadersLarge, SimpleDDQNNetwork, SimpleCriticNetwork,
    # save_location = f'{os.environ["PBS_O_WORKDIR"]}/models/multi_ddqn_large_single_agent',
    # stats_location = f'{os.environ["PBS_O_WORKDIR"]}/models/multi_ddqn_large_single_agent_stats'
    # )

    # MultiModelDDQN.train(SpaceInvaders, SimpleDDQNNetwork, SimpleCriticNetwork,
    # save_location = 'models/multi_ddqn_simple')

    ############## MultiModelDDQN Play
    # ui = GameUI(SpaceInvaders, model = MultiModelDDQN('models/multi_ddqn_simple'), record=False)
    # ui.run()
    # # # ui.playback()

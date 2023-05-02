import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from src.networks.simple_network import SimpleActorNetwork, SimpleCriticNetwork, SimpleDDQNNetwork
from src.user_interface.ui import GameUI
from src.games.space_invaders import SpaceInvaders
from src.games.space_invaders_large import SpaceInvadersLarge
from src.algorithms.PPO import PPO
from src.algorithms.DDQN import DDQN
from src.algorithms.MultiModelPPO import MultiModelPPO



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    ########### PPO Play
    ppo = PPO()
    ppo.load('models/ppo_9')
    ui = GameUI(SpaceInvadersLarge, model = ppo, record=False)
    # ui = GameUI(SpaceInvaders, record=False)
    ui.run()
    # # ui.playback()


    ############ PPO Train
    # print('Starting Training')
    # ppo = PPO()
    # print('Training PPO')
    # print('######################')
    # # ppo.train(SpaceInvaders, SimpleActorNetwork, SimpleCriticNetwork, save_location = f'{os.environ["PBS_O_WORKDIR"]}/ppo_6')
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
    # print('Starting Training')
    # multi_model_ppo = MultiModelPPO()
    # multi_model_ppo.train(SpaceInvaders, SimpleActorNetwork, SimpleCriticNetwork, save_location = 'multi_model_ppo_1')

    # # # ########### MultiModelPPO Play
    # multi_model_ppo = MultiModelPPO()
    # multi_model_ppo.load('multi_model_ppo_1')
    # # ui = GameUI(SpaceInvadersLarge, model = multi_model_ppo, record=False)
    # ui = GameUI(SpaceInvaders, model = multi_model_ppo, record=False)
    # ui.run()
    # # # ui.playback()

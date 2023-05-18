'''Multi Agent'''
from src.interfaces.network import Network
from src.interfaces.game import Game
import gc
import time
import dataclasses
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import os
from typing import NamedTuple, Type
import json

import numpy as np
import numpy.typing as npt
# Importing tensorflow modules this way makes type hinting work because reasons
import tensorflow as tf
import tensorflow_probability as tfp
import tensor_annotations.tensorflow as ttf

from src.networks.empty_network import EmptyNetwork
keras = tf.keras
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses

@dataclasses.dataclass
class Trajectories:
    '''Trajectory stored from one frame of the game'''
    observations: npt.NDArray
    done_flags: list[bool]
    actions: npt.NDArray
    rewards: npt.NDArray
    discount_cumulative_rewards: npt.NDArray
    scores: list[float]

def action_to_action_array(action: int) -> list[float]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]

class WorkerArgs(NamedTuple):
    '''Args for input to a worker'''
    num_observations: int
    input_queue: multiprocessing.Queue
    output_queue: multiprocessing.Queue
    stop_queue: multiprocessing.Queue
    game_type: Type[Game]
    observation_dims: int
    action_dims: int
    num_actors: int
    actor_network: type[Network]
    critic_network: type[Network]
    load_location: str
    discount_factor: float
    eps: float
    eps_min: float
    eps_decay: float


def create_trajectories_process(
    args: WorkerArgs
):
    '''Can be run as a separate process to collect training data'''
    eps = args.eps
    while True:
        # stop if received stop message
        if not args.stop_queue.empty():
            break

        # continue unless asked to collect data
        if args.input_queue.empty():
            continue

        args.input_queue.get()
        print('Started Process')
        observation_count = 0
        network_controller = MultiModelDDQN.NetworkController(
            args.actor_network,
            args.critic_network,
            args.num_actors,
            args.observation_dims,
            args.action_dims,
            load=True,
            load_location=args.load_location
        )

        observations: list[npt.NDArray] = []
        actions: list[int] = []
        rewards: list[float] = []
        discount_cumulative_rewards: list[float] = []
        sub_agent_rewards: list[npt.NDArray] = []
        scores = []
        done_flags = []

        while observation_count < args.num_observations:
            game = args.game_type()
            observation = game.get_model_input()
            done = False
            new_observations = []
            new_actions = []
            new_rewards = []
            score = 0
            while not done:
                score += 1
                # step the game
                action = network_controller.epsilon_greedy(
                    args.eps, observation)
                done, reward = game.step(action_to_action_array(action))
                # store the frame state
                new_observations.append(observation)
                new_actions.append(action)
                new_rewards.append(reward)
                # update variables
                observation = game.get_model_input()
                observation_count += 1
                if eps > args.eps_min:
                    eps -= args.eps_decay

            # calculate discount cumulative rewards
            new_discount_cumulative_rewards = []
            discount_cumulative_reward = 0
            for reward in reversed(new_rewards):
                discount_cumulative_reward = reward + \
                    discount_cumulative_reward * args.discount_factor
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))

            observations += new_observations

        trajectories = Trajectories(
            np.array(observations),
            done_flags,
            np.array(actions),
            np.array(rewards),
            np.array(discount_cumulative_rewards),
            scores
        )
        print('Completed Process')
        args.output_queue.put(trajectories)




class MultiModel:

    class NetworkController:
        '''Neural nets used for function approximation'''

        LEARNING_RATE = 0.005

        def __init__(self,
                     actor_network: type[Network],
                     critic_network: type[Network],
                     num_actors: int,
                     observation_dims: int,
                     action_dims: int,
                     load: bool = False,
                     load_location: str = '') -> None:
            self._actor_network = actor_network
            self._critic_network = critic_network
            self._num_actors = num_actors
            self._observation_dims = observation_dims
            self._action_dims = action_dims
            self._load = load
            self._load_location = load_location
            # Actors implement policy to maximise their own sub value
            self._actors = [
                self._actor_network(self._observation_dims,
                                    pow(self._action_dims, 2))
                for _ in range(self._num_actors)
            ]
            # Critic estimates Q function for each sub agent using global reward
            self._critic = self._critic_network(
                self._observation_dims, self._num_actors)
            self._target_critic = self._critic_network(
                self._observation_dims, self._num_actors)
            self.update_target_critic()
            if load:
                self.load(self._load_location)
            else:
                self.critic_optimiser = Adam(lr=self.LEARNING_RATE)
                self._actor_optimisers = [
                    Adam(lr=self.LEARNING_RATE)
                ]

        def save(self, save_location: str):
            '''Save the current models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            with open(f'{save_location}/model_configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps(
                    {
                        'num_actors': self._num_actors,
                        'observation_dims': self._observation_dims,
                        'action_dims': self._action_dims,
                    }
                ))
            for i, actor in enumerate(self._actors):
                actor.model.save(f'{save_location}/actor_{i}')
            self._critic.model.save(f'{save_location}/critic')

        @classmethod
        def load_saved_model(cls, load_location) -> 'MultiModel.NetworkController':
            '''Load a saved model without passing any configs'''
            network_controller = cls.__new__(cls)
            network_controller.load(load_location)
            return network_controller

        def load(self, load_location: str):
            '''Load the saved models'''
            with open(f'{load_location}/model_configs.json', 'r', encoding='utf8') as file:
                model_configs = json.loads(file.read())
            self._num_actors = model_configs['num_actors']
            self._observation_dims = model_configs['observation_dims']
            self._action_dims = model_configs['action_dims']
            self._actors = [Network(
                model_configs['observation_dims'],
                model_configs['action_dims']
            ) for i in range(model_configs['num_actors'])]
            for i, actor in enumerate(self._actors):
                actor.load(f'{load_location}/actor_{i}')
            self._critic = Network(
                model_configs['observation_dims'],
                model_configs['num_actors']
            )
            self._critic.load(f'{load_location}/critic')

        def update_target_critic(self):
            self._target_critic.model.set_weights(self._critic.model.get_weights())

        @property
        def critic(self):
            '''Getter for the critic'''
            return self._critic

        @property
        def actors(self):
            '''Getter for the critic'''
            return self._actors


    class Trainer:
        '''
        Train the models for a given game
        '''

        NUM_ACTORS = 3
        NUM_WORKERS = 7
        EPS_MIN = 0.05
        EPS_DECAY = 0.0001
        DISCOUNT_FACTOR = 0.95
        NUM_FRAMES = 40000
        TARGET_C = 20000
        BATCH_SIZE = 5000

        def __init__(self,
                     actor_network: type[Network],
                     critic_network: type[Network],
                     game_type: type[Game],
                     save_location: str,
                     stats_location: str) -> None:

        self.actor_network = actor_network
        self.critic_network = critic_network

        self.game_type = game_type
        self.input_shape = game_type.get_input_shape()
        self.action_shape = game_type.get_action_shape()

        self.save_location = save_location
        self.stats_location = stats_location
        if not os.path.exists(f'{save_location}'):
            os.mkdir(f'{save_location}')
        if not os.path.exists(f'{stats_location}'):
            os.mkdir(f'{stats_location}')


        if os.environ['first_run'] == 'True':
                self.network_controller = MultiModel.NetworkController(
                    actor_network,
                    critic_network,
                    self.NUM_ACTORS,
                    self.input_shape,
                    self.action_shape,
                    load=False)

        else:
            self.network_controller = MultiModel.NetworkController(
                actor_network,
                critic_network,
                self.NUM_ACTORS,
                self.input_shape,
                self.action_shape,
                load=True)
            self.network_controller.load(save_location)

        # queues for messaging between workers
        self.task_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.stop_queue = multiprocessing.Queue()

        self.workers: list[multiprocessing.Process] = []

        def stop_workers(self):
            '''Send a message to each worker to trigger a stop'''
            self.stop_queue.put(True)

        def create_workers(self, eps: float, eps_min: float, eps_decay: float, num_observations):
            '''Create Workers to collect training data'''
            BaseManager.register('Trajectories', Trajectories)
            manager = BaseManager()
            manager.start()
            self.workers = []
            for _ in range(self.NUM_WORKERS):
                process = Process(target=create_trajectories_process, args=[
                    WorkerArgs(
                        int(num_observations / self.NUM_WORKERS),
                        self.task_queue,
                        self.response_queue,
                        self.stop_queue,
                        self.game_type,
                        self.input_shape,
                        self.action_shape,
                        self.NUM_ACTORS,
                        self.actor_network,
                        self.critic_network,
                        self.save_location,
                        self.DISCOUNT_FACTOR,
                        eps,
                        eps_min,
                        eps_decay
                    )
                ])
                self.workers.append(process)
                process.start()

        def collect_trajectories_parallel(self,
                                          num_observations,
                                          eps: float,
                                          eps_min: float,
                                          eps_decay: float):
            '''Use the workers to collect a set of training data'''
            # start the workers
            for _ in self.workers:
                self.task_queue.put(True)

            # get the trajectories from each of the workers
            trajectories_list: list[Trajectories] = []
            while len(trajectories_list) < len(self.workers):
                if not self.response_queue.empty():
                    print('Received response')
                    trajectories = self.response_queue.get()
                    trajectories_list.append(trajectories)

            # combine the trajectories from each worker
            trajectories = Trajectories(
                # list of observations
                np.concatenate(
                    tuple(trajectories.observations for trajectories in trajectories_list)),
                # flags for completed game frames
                [done_flag for trajectories in
                    trajectories_list for done_flag in trajectories.done_flags],
                # list of actions
                np.array([action for trajectories in trajectories_list for action in list(
                    trajectories.actions)]),
                # list of rewards
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.rewards)]),
                # list of discount cumulative rewards
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.discount_cumulative_rewards)]),
                # list of scores
                [score for trajectories in trajectories_list for score in trajectories.scores],
            )

            return trajectories


        def critic_loss(self, trajectories: Trajectories):
            '''
            Calculate the MSE of the critic q functions
            May need to add KL divergence term
            '''

        def actor_advantage_estimates(self, actor_index: int, trajectories: Trajectories):
            '''Calculate the advantage estimate for an actor'''


        def experience_replay(self, trajectories: Trajectories):
            '''
            Train the critic on the discount cumulative rewards
            and the actors on their sub rewards
            '''

            # Critic needs to be updated

            # Advantage functions can be calculated for the actors

            # Actors can be updated

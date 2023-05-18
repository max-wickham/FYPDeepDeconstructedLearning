'''Multi Agent Q Learning'''
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
    sub_agent_rewards: npt.NDArray
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

            # TODO try this
            # https://proceedings.neurips.cc/paper/2019/file/97108695bd93b6be52fa0334874c8722-Paper.pdf#cite.grimm2019learning
            critic_values: ttf.Tensor1 = network_controller.value(
                np.array(new_observations))
            values_shifted: ttf.Tensor1 = tf.identity(
                critic_values)  # type: ignore
            values_shifted: ttf.Tensor1 = tf.slice(
                values_shifted, [1, 0], [-1, -1])
            values_shifted = tf.math.scalar_mul(args.discount_factor, values_shifted) # type: ignore
            # remove the last value from values
            values: ttf.Tensor1 = tf.slice(
                critic_values, [0, 0], [int(critic_values.shape[0])-1, -1])  # type: ignore
            new_sub_agent_rewards: npt.NDArray = tf.subtract(
                values, values_shifted).numpy()

            observations += new_observations[:-1]
            actions += new_actions[:-1]
            rewards += new_rewards[:-1]
            sub_agent_rewards = np.concatenate((sub_agent_rewards, new_sub_agent_rewards))  \
                if len(sub_agent_rewards) > 0 else new_sub_agent_rewards
            discount_cumulative_rewards += list(
                reversed(new_discount_cumulative_rewards))[:-1]
            scores.append(score)
            done_flags += ([False] * (len(new_observations) - 1)) + [True]

        trajectories = Trajectories(
            np.array(observations),
            done_flags,
            np.array(actions),
            np.array(rewards),
            np.array(discount_cumulative_rewards),
            np.array(sub_agent_rewards),
            scores
        )
        print('Completed Process')
        args.output_queue.put(trajectories)



class MultiModelDDQN:
    '''
    Multiple DDQN agents that are each trained to try and maximise
    one node in the penultimate layer of a critic. The critic is
    trained to estimate the discount cumulative reward of the current
    state.

    The player chooses an action that maximises the summed q value
    given by each agent for that action.
    '''

    class NetworkController:
        '''
        Contains N models to estimate Q functions and one critic model
        to estimate the discount cumulative rewards of a given state.

        Each of the N Q models has a second model that is updated periodically
        for stability.
        '''
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
            self._actors = [
                self._actor_network(self._observation_dims,
                                    pow(self._action_dims, 2))
                for _ in range(self._num_actors)
            ]
            self._target_actors = []
            self.update_target_networks()
            self._critic = self._critic_network(
                self._observation_dims, self._num_actors)
            if load:
                self.load(self._load_location)
                self.update_target_networks()
            else:
                self.critic_optimiser = Adam(lr=self.LEARNING_RATE)

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
        def load_saved_model(cls, load_location) -> 'MultiModelDDQN.NetworkController':
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

        def epsilon_greedy(self, eps: float, observation: npt.NDArray) -> int:
            '''Select an action from the current sub agents using the epsilon greedy strategy'''
            if np.random.random() < eps:
                # exploration
                return self.get_random_action()
            else:
                # exploitation
                q_vals = np.array(
                    [
                        self.get_action_q_vals(i, observation)
                        for i in range(self._num_actors)
                    ]
                )
                q_vals = np.sum(q_vals, axis=0)
                return int(np.argmax(q_vals))

        def value(self, observations: npt.NDArray) -> ttf.Tensor1:
            '''Get the critic values for a given set of observations'''
            values = self._critic(observations, multi_dim=True)
            return values

        def get_random_action(self) -> int:
            '''Generate a random action within the action space'''
            return np.random.randint(0, pow(self._action_dims, 2))

        def update_target_networks(self):
            '''Update the actor targets to match the actors'''
            self._target_actors = [
                self._actor_network(self._observation_dims,
                                    pow(self._action_dims, 2))
                for _ in range(self._num_actors)
            ]
            for target_actor, actor in zip(self._target_actors, self._actors):
                target_actor.model.set_weights(actor.model.get_weights())

        def get_action_q_vals(self, actor_index, observation) -> npt.NDArray:
            '''Get the q val for an observation'''
            return self._actors[actor_index](observation).numpy()[0]

        def target_q_vals(self, observation) -> npt.NDArray:
            return np.array([
                self._target_actors[actor_index](observation).numpy()[0]
                for actor_index in range(self._num_actors)
            ])

        def target_reward_for_action(self,
                                     actor_index: int,
                                     observation: npt.NDArray,
                                     action: int) -> float:
            '''Get the target reward for a given action'''
            return self._target_actors[actor_index](observation).numpy()[0][action]

        def fit_actor(self, actor_index, states, targets):
            '''Train the actor on the new targets for a set of states'''
            self._actors[actor_index].model.fit(
                np.array(
                    [tf.convert_to_tensor(np.array(
                        state).reshape(-1).reshape(1, self._observation_dims)) for state in states]
                ).reshape(len(states), self._observation_dims),
                np.array(targets).reshape(len(targets), len(targets[0])),
                epochs=3,
            )

        @property
        def critic(self):
            '''Getter for the critic'''
            return self._critic

    class Trainer:
        '''
        Train the N Q-Function models for a given game
        '''

        NUM_ACTORS = 3
        INIT_MEMORY = 1000
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
                self.network_controller = MultiModelDDQN.NetworkController(
                    actor_network,
                    critic_network,
                    self.NUM_ACTORS,
                    self.input_shape,
                    self.action_shape,
                    load=False)
            else:
                self.network_controller = MultiModelDDQN.NetworkController(
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
                # list of discount cumulative rewards
                np.array([rewards for trajectories in trajectories_list for rewards in list(
                    trajectories.sub_agent_rewards)]),
                # list of scores
                [score for trajectories in trajectories_list for score in trajectories.scores],
            )

            return trajectories

        def experience_replay(self, trajectories: Trajectories):
            '''
            Train the critic on the discount cumulative rewards
            and the actors on their sub rewards
            '''
            # TODO find a way to do this without for loops
            state_q_vals = [
                self.network_controller.target_q_vals(observation)
                for observation in trajectories.observations
            ]
            # best_actions = [
            #     int(
            #         np.argmax(
            #             np.sum(next_state_q_vals, axis=0)
            #         )
            #     ) for next_state_q_vals in state_q_vals
            # ]

            for actor_index in range(self.NUM_ACTORS):
                print('Training actor', actor_index)
                states = []
                targets = []
                for index in range(len(trajectories.observations)-1):
                    # if game not over, target q val includes discounted future utility
                    done = trajectories.done_flags[index]
                    # set the target to the reward received from this state for this agent
                    target = trajectories.sub_agent_rewards[index][actor_index]
                    action = trajectories.actions[index]
                    if not done:
                        new_observation = trajectories.observations[index + 1]

                        ###### next action chosen by the current agent ######
                        # best_action = int(
                        #     np.argmax(
                        #         self.network_controller.get_action_q_vals(
                        #             actor_index, new_observation)
                        #     )
                        # )
                        # target = target + self.DISCOUNT_FACTOR * \
                        #     self.network_controller.target_reward_for_action(
                        #         actor_index, new_observation, best_action)

                        ###### next action chosen by combined agents ######
                        # This method is better acording to the paper
                        # https://people.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf
                        # Q_i(s,a) = r_i + Q_i(s+1,pi*(s+1)), where a+1 is chosen by the group of agents
                        # get the next q vals using the target agents for stability
                        next_state_q_vals = state_q_vals[index + 1]
                        # get the next actiont that was taken by the global policy
                        best_action = trajectories.actions[index + 1]

                        target = target + self.DISCOUNT_FACTOR * next_state_q_vals[actor_index][best_action]

                    observation = trajectories.observations[index]
                    current_q_vals = self.network_controller.get_action_q_vals(
                        actor_index, observation)
                    # update the current q_val with the discount future reward for that action using the global policy
                    current_q_vals[action] = target
                    states.append(observation)
                    targets.append(current_q_vals)

                self.network_controller.fit_actor(actor_index, states, targets)

            # train the critic
            with tf.GradientTape() as critic_tape:
                critic_tape.watch(
                    self.network_controller.critic.model.trainable_variables)
                critic_values = self.network_controller.value(
                    trajectories.observations)
                values = tf.reduce_sum(critic_values, axis=1)
                discount_cumulative_rewards = tf.reshape(
                    trajectories.discount_cumulative_rewards,
                    (len(trajectories.discount_cumulative_rewards),)
                )
                critic_loss = 0.5 * \
                    kls.mean_squared_error(discount_cumulative_rewards, values)
            critic_gradients = critic_tape.gradient(
                critic_loss, self.network_controller.critic.model.trainable_variables)
            self.network_controller.critic_optimiser.apply_gradients(
                zip(critic_gradients, self.network_controller.critic.model.trainable_variables))

        def run_frames(self):
            '''Run through the game to train the sub agents'''
            total_frames = 0
            eps = 1
            num_updates_since_target_update = 0
            num_batches = 0
            while total_frames < self.NUM_FRAMES:
                print(f'Running Batch {num_batches}')
                num_batches += 1
                # collect a batch size amount of trajectories
                self.network_controller.save(self.save_location)
                trajectories = self.collect_trajectories_parallel(
                    self.BATCH_SIZE,
                    eps,
                    self.EPS_MIN,
                    self.EPS_DECAY
                )
                with open(f'{self.stats_location}/scores.csv',
                          'a',
                          encoding='utf8') as file:
                    for score in trajectories.scores:
                        file.write(f'{score}\n')
                eps = max(self.EPS_MIN, eps - self.EPS_DECAY *
                          len(trajectories.observations))
                self.experience_replay(
                    trajectories
                )
                num_updates_since_target_update += len(
                    trajectories.observations)
                if num_updates_since_target_update > self.TARGET_C:
                    num_updates_since_target_update = 0
                    self.network_controller.update_target_networks()
            self.network_controller.save(self.save_location)

        def train(self):
            '''Train the models'''
            eps = 1 if os.environ['first_run'] == 'True' else self.EPS_MIN
            self.create_workers(eps, self.EPS_MIN, self.EPS_DECAY, int(
                self.BATCH_SIZE / self.NUM_WORKERS))
            self.run_frames()
            self.stop_workers()

    @classmethod
    def train(cls,
              game_type: type[Game],
              actor_network: type[Network],
              critic_network: type[Network],
              save_location='multi_ddqn',
              stats_location='multi_ddqn_stats'):
        '''Use the algorithm to train models on the given game type'''
        trainer = cls.Trainer(
            actor_network,
            critic_network,
            game_type,
            save_location,
            stats_location
        )
        trainer.train()

    def __init__(self, load_location: str) -> None:
        self.load_location = load_location
        self.network_controller = self.NetworkController.load_saved_model(
            load_location)

    def compute_action(self, game: Game):
        '''Return an action array for the current game'''
        return action_to_action_array(
            self.network_controller.epsilon_greedy(0, game.get_model_input()))

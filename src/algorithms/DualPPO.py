'''
PPO Algorithm
Algorithm Pseudo Code https://spinningup.openai.com/en/latest/algorithms/ppo.html
'''
import gc
from src.interfaces.game import Game
from src.interfaces.network import Network
import tensorflow_probability as tfp
from tensor_annotations import axes
import tensor_annotations.tensorflow as ttf
import dataclasses
from dataclasses import field
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing.shared_memory import SharedMemory
import os
from typing import Callable, NamedTuple, Any, Type
import json
from multiprocessing import Pool, Process, TimeoutError
import importlib


import numpy as np
import numpy.typing as npt
# Importing tensorflow modules this way makes type hinting work because reasons
import tensorflow as tf
keras = tf.keras
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.losses as kls


@dataclasses.dataclass
class Trajectories:
    '''Trajectory stored from one frame of the game'''
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    probabilities: npt.NDArray
    discount_cumulative_rewards: npt.NDArray
    destroyed_enemy_rewards: npt.NDArray
    scores : list[float]

    def get_observations(self):
        '''Returns observations'''
        return self.observations


def create_trajectories_process(
    num_observations: int,
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    stop_queue: multiprocessing.Queue,
    game_type: Type[Game],
    observation_dims: int,
    action_dims: int,
    network_type: type[Network],
    load_location: str
):
    '''Create a set of trajectories on an individual thread'''

    while True:
        if not stop_queue.empty():
            break
        if input_queue.empty():
            continue
        input_queue.get()
        # importlib.reload(tf)
        # keras = tf.keras;
        # Adam = tf.keras.optimizers.Adam
        # kls = tf.keras.losses
        observation_count = 0
        actions = []
        observations = []
        rewards = []
        probs = []
        scores = []
        discount_cumulative_rewards = []
        enemies_destroyed = []
        network_controller = DualPPO.NetworkController(
            network_type, network_type, observation_dims, action_dims, load=True)
        network_controller.load(load_location)
        while observation_count < num_observations:
            game = game_type()
            num_enemies = 0
            done = False
            new_actions = []
            new_observations = []
            new_rewards = []
            new_probs = []
            prev_action = 0
            prev_prob = None
            action_repetitions = 0
            new_enemies_destroyed = []
            print('New Game')
            frame = 0
            while not done and observation_count < num_observations:
                frame += 1
                observation = game.get_model_input()
                observation = tf.reshape(
                    observation, shape=(1, len(observation)))
                if action_repetitions == 0:
                    observation_count += 1
                    # TODO create network controller from model file
                    prob, action = network_controller.get_prob_action(
                        observation)
                    prev_prob = prob
                    prev_action = action
                done, reward = game.step(
                    action_to_action_array(prev_action))
                if done or action_repetitions == 0:
                    new_observations.append(observation)
                    new_actions.append(prev_action)
                    new_rewards.append(reward)
                    new_probs.append(prev_prob)
                    new_enemies_destroyed.append(1 if len(game._game_state.enemies) < num_enemies else 0)  # type: ignore
                num_enemies = len(game._game_state.enemies) < num_enemies  # type: ignore
                action_repetitions = (action_repetitions + 1) % 4
            scores.append(frame)
            new_discount_cumulative_rewards = []
            discount_cumulative_reward = 0
            for reward in reversed(new_rewards):
                discount_cumulative_reward = reward + discount_cumulative_reward * 0.99
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))
            discount_cumulative_reward = 0
            for reward in reversed(new_enemies_destroyed):
                discount_cumulative_reward = reward + discount_cumulative_reward * 0.9
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))


            observations += new_observations
            actions += new_actions
            rewards += new_rewards
            probs += new_probs
            enemies_destroyed += new_enemies_destroyed
            discount_cumulative_rewards += reversed(
                new_discount_cumulative_rewards)

        # print(discount_cumulative_rewards)
        print('Completed Collection')
        trajectories = Trajectories(
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(probs),
            np.array(discount_cumulative_rewards),
            np.array(enemies_destroyed),
            scores
        )
        print('Putting trajectories on queue')
        output_queue.put(trajectories)

        continue


def action_to_action_array(action: int) -> list[float]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


class DualPPO:
    '''Model implementing the PPO training algorithm'''

    class NetworkController:
        '''Actor and critic models'''
        actor: Network
        critic: Network

        def __init__(self,
                     actor_network: type[Network],
                     critic_network:  type[Network],
                     input_dims: int,
                     output_dims: int,
                     load=False,
                     custom_reward_function : None | Callable = None) -> None:
            if load:
                return
            self.learning_rate = 0.005

            # takes an observation and outputs a set of actions
            # square action dims to get output space
            self.actor = actor_network((input_dims), output_dims * output_dims)
            # takes an observation and outputs a value
            self.critic = critic_network((input_dims), 5)
            self.actor_optimiser = Adam(lr=self.learning_rate)
            self.critic_optimiser = Adam(lr=self.learning_rate)

        def load(self, load_location: str, load_critic=False):
            '''Load saved models'''
            print('Load')
            self.actor = keras.models.load_model(
                f'{load_location}/actor',  custom_objects={'tf': tf})  # type: ignore
            print('Actor Loaded')
            if load_critic:
                self.critic = keras.models.load_model(
                    f'{load_location}/critic',  custom_objects={'tf': tf})  # type: ignore

        def save(self, save_location: str):
            '''Save current models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            self.actor.model.save(f'{save_location}/actor')
            self.critic.model.save(f'{save_location}/critic')

        def get_prob_action(self, observation: npt.NDArray) -> tuple[npt.NDArray, int]:
            '''Returns the probability of
            each action combination and an action sampled from this distribution'''

            prob = self.actor(observation)
            prob = prob.numpy()[0]
            dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            action = dist.sample(1)
            action = int(action.numpy()[0])
            if action >= 4:
                action = 3
            return prob, action

        def get_values(self, observations: npt.NDArray) -> npt.NDArray:
            '''Run the critic on a set of observations'''
            values = self.critic(observations, multi_dim=True).numpy()
            values = np.reshape(values, (len(values),5))
            return values

    class Trainer:
        '''Train a PPO model'''

        class SubModel:

            # test this 
            

        def __init__(self,
                     game_type: type[Game],
                     actor_network: type[Network],
                     critic_network: type[Network],
                     save_location: str,
                     stats_location='ppo_stats',
                     ) -> None:
            self.stats_location = stats_location
            try:
                os.rmdir(stats_location)
            except Exception:
                ...
            os.mkdir(stats_location)
            # with open(f'{self.stats_location}/game_stats.csv', 'a', encoding='utf8') as file:
            #     file.write('value,advantage,action\n')
            with open(f'{self.stats_location}/loss_stats.csv', 'a', encoding='utf8') as file:
                file.write('actor_loss,critic_loss\n')

            self.save_location = save_location
            self.action_dims = game_type.get_action_shape()
            self.observation_dims = game_type.get_input_shape()
            self.actor_network_type = actor_network
            self.critic_network_type = critic_network
            # self.total_time_steps = 10000000
            # self.total_time_steps = 60000000
            # self.observations_per_batch = 20000
            self.num_workers = 2
            self.total_time_steps = 50000
            self.observations_per_batch = 500

            self.updates_per_iteration = 2
            self.game_type = game_type
            self.gamma = 0.95
            self.clip = 0.2
            self.repeat_action_num = 3

            self.network_controller = PPO.NetworkController(
                actor_network, critic_network, self.observation_dims, self.action_dims)
            self.shoot_network_controller = PPO.NetworkController(
                actor_network, critic_network, self.observation_dims, self.action_dims)

            self.task_queue = multiprocessing.Queue()
            self.response_queue = multiprocessing.Queue()
            self.stop_queue = multiprocessing.Queue()
            self.workers: list[multiprocessing.Process] = []

        def create_trajectories_parallel(self) -> Trajectories:
            '''Create a set of trajectories using parallel workers'''
            self.save()
            # Start the workers
            for _ in self.workers:
                self.task_queue.put(True)
            # Wait until received trajectories from every worker
            trajectories_list: list[Trajectories] = []
            while len(trajectories_list) < len(self.workers):
                if not self.response_queue.empty():
                    trajectories = self.response_queue.get()
                    trajectories_list.append(trajectories)

            trajectories = Trajectories(
                np.concatenate(
                    tuple(trajectories.observations for trajectories in trajectories_list)),
                np.array([action for trajectories in trajectories_list for action in list(
                    trajectories.actions)]),
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.rewards)]),
                np.concatenate(
                    tuple(
                        np.reshape(trajectories.probabilities,
                                   (trajectories.probabilities.shape[0],
                                    1, trajectories.probabilities.shape[1])
                                   )
                        for trajectories in trajectories_list)),
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.discount_cumulative_rewards)]),
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.destroyed_enemy_rewards)]),
                [score for trajectories in trajectories_list for score in trajectories.scores]
            )
            return trajectories

        def create_trajectories(self) -> Trajectories:
            '''
            Collect a set of trajectories by running policy on the environment
            '''
            observation_count = 0
            actions = []
            observations = []
            rewards = []
            probs = []
            discount_cumulative_rewards = []

            while observation_count < self.observations_per_batch:
                print('New Game')
                game = self.game_type()
                done = False
                new_actions = []
                new_observations = []
                new_rewards = []
                new_probs = []
                prev_action = 0
                prev_prob = None
                action_repetitions = 0
                while not done and observation_count < self.observations_per_batch:
                    observation = game.get_model_input()
                    if action_repetitions == 0:
                        observation_count += 1
                        prob, action = self.network_controller.get_prob_action(
                            observation)
                        prev_prob = prob
                        prev_action = action
                    done, reward = game.step(
                        action_to_action_array(prev_action))
                    if done or action_repetitions == 0:
                        new_observations.append(observation)
                        new_actions.append(prev_action)
                        new_rewards.append(reward)
                        new_probs.append(prev_prob)
                    action_repetitions = (action_repetitions + 1) % 4

                new_discount_cumulative_rewards = []
                discount_cumulative_reward = 0
                for reward in reversed(new_rewards):
                    discount_cumulative_reward = reward + discount_cumulative_reward * self.gamma
                    new_discount_cumulative_rewards.append(
                        discount_cumulative_reward)

                observations += new_observations
                actions += new_actions
                rewards += new_rewards
                probs += new_probs
                discount_cumulative_rewards += new_discount_cumulative_rewards
                print(sum(new_discount_cumulative_rewards) /
                      len(new_discount_cumulative_rewards))
            return Trajectories(
                np.array(observations),
                np.array(actions),
                np.array(rewards),
                np.array(probs),
                np.array(discount_cumulative_rewards)
            )

        def compute_value_advantage_estimates(self, trajectories: Trajectories
                                              ) -> tuple[npt.NDArray, npt.NDArray]:
            '''For a given set of trajectories calculate the set of advantage estimates'''
            # compute the value function for the set of trajectories using the critic
            values: np.ndarray = self.network_controller.get_values(
                trajectories.observations)
            values = np.sum(values, axis=1)
            # compute the log estimates using the actor
            # log_probs = self.model.get_log_probs(observations, actions)
            advantages = np.subtract(
                trajectories.discount_cumulative_rewards, values)
            # advantages = trajectories.discount_cumulative_rewards - values
            advantages = (advantages - np.mean(advantages)) / \
                (np.std(advantages) + 1e-10)
            # TODO normalize advantages ?
            return values, advantages

        def actor_loss(self, new_probs: ttf.Tensor1,
                       current_probs: npt.NDArray,
                       actions: npt.NDArray,
                       advantages: npt.NDArray,
                       critic_loss: ttf.Tensor1):
            '''Calculate the actor loss'''
            entropy = tf.reduce_mean(tf.math.negative(
                tf.math.multiply(new_probs, tf.math.log(new_probs))))
            # print(new_probs)
            # print(entropy)
            surrogate_1 = []
            surrogate_2 = []

            advantages_tensor = tf.convert_to_tensor(
                advantages, dtype=tf.float32)
            # print(advantages_tensor)
            # new_probs_indexed = tf.convert_to_tensor(
            #     np.array([prob[action] for prob, action in zip(new_probs, actions)]), dtype=tf.float32)
            current_probs = np.reshape(
                current_probs, (current_probs.shape[0], current_probs.shape[2]))
            current_probs_indexed = tf.convert_to_tensor(
                np.array([prob[action]
                          for prob, action in zip(current_probs, actions)]), dtype=tf.float32)
            new_probs_indexed = tf.gather_nd(
                new_probs,
                indices=tf.constant([[index, action] for index, action in enumerate(actions)]))
            # current_probs_indexed = tf.gather_nd(
            #     current_probs,
            #     indices = tf.constant([[index, action] for index, action in enumerate(actions)]))
            ratios = tf.math.divide(new_probs_indexed, current_probs_indexed)
            surrogate_1 = tf.math.multiply(ratios, advantages_tensor)
            surrogate_2 = tf.math.multiply(tf.clip_by_value(
                ratios, 1.0 - self.clip, 1.0 + self.clip), advantages)
            sr1 = tf.stack(surrogate_1)
            sr2 = tf.stack(surrogate_2)
            loss = tf.math.negative(tf.reduce_mean(
                tf.math.minimum(sr1, sr2)) - critic_loss + 0.001 * entropy)
            return loss

        def update_policy(self, trajectories: Trajectories, advantage_estimates: np.ndarray):
            '''
            Update the policy using the trajectories and advantage estimates
            Use stochastic gradient descent using ADAM
            '''
            discount_cumulative_rewards = tf.reshape(
                trajectories.discount_cumulative_rewards,
                (len(trajectories.discount_cumulative_rewards),)
            )

            # advantages = tf.reshape(
            #     advantage_estimates, (len(advantage_estimates),))
            # current_probs = tf.reshape(trajectories.probabilities,
            #                            (len(trajectories.probabilities), self.action_dims*self.action_dims))
            # print()

            with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
                actor_tape.watch(
                    self.network_controller.actor.model.trainable_variables)
                critic_tape.watch(
                    self.network_controller.critic.model.trainable_variables)
                probabilities = self.network_controller.actor(
                    trajectories.observations, training=True, multi_dim=True)
                # actor_tape.watch(self.network_controller.actor.model.trainable_variables)
                values = self.network_controller.critic(
                    trajectories.observations, training=True, multi_dim=True)
                # TODO compute value sum and value standard distribution
                breakpoint()
                correlation = tf.stats.corelation(values)
                value_sd = tf.math.reduce_std(values, axis=1)
                values = tf.math.reduce_sum(values, axis=1)
                breakpoint()
                values = tf.reshape(values, (len(values),))
                critic_loss = 0.5 * \
                    kls.mean_squared_error(discount_cumulative_rewards, values)
                print('Computing Loss')
                actor_loss = self.actor_loss(
                    probabilities,
                    trajectories.probabilities,
                    trajectories.actions,
                    advantage_estimates,
                    critic_loss
                )
                # critic_loss = tf.subtract(critic_loss, 0.01*tf.math.reduce_mean(value_sd))
                breakpoint()

            # calculate and apply gradients
            actor_gradients = actor_tape.gradient(
                actor_loss, self.network_controller.actor.model.trainable_variables)
            critic_gradients = critic_tape.gradient(
                critic_loss, self.network_controller.critic.model.trainable_variables)

            self.network_controller.actor_optimiser.apply_gradients(
                zip(actor_gradients, self.network_controller.actor.model.trainable_variables))
            self.network_controller.critic_optimiser.apply_gradients(
                zip(critic_gradients, self.network_controller.critic.model.trainable_variables))
            return actor_loss, critic_loss

        def save(self):
            '''Save the trained models'''
            if not os.path.exists(f'{self.save_location}'):
                os.mkdir(f'{self.save_location}')
            configs = {
                'game_type': str(type(self.game_type)),
                'network_type': str(type(self.actor_network_type))
            }
            with open(f'{self.save_location}/configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps(configs))
            self.network_controller.save(self.save_location)
            # self.network_controller.actor.model.save(f'{self.save_location}/actor')
            # self.network_controller.critic.model.save(f'{self.save_location}/critic')

        def create_workers(self):
            '''Create Workers to collect training data'''
            BaseManager.register('Trajectories', Trajectories)
            manager = BaseManager()
            manager.start()
            self.workers = []
            for _ in range(self.num_workers):
                process = Process(target=create_trajectories_process, args=[
                    int(self.observations_per_batch / self.num_workers),
                    self.task_queue,
                    self.response_queue,
                    self.stop_queue,
                    self.game_type,
                    self.observation_dims,
                    self.action_dims,
                    self.actor_network_type,
                    self.save_location])
                self.workers.append(process)
                process.start()

        def stop_workers(self):
            '''Send a message to each worker to trigger a stop'''
            self.stop_queue.put(True)

        def train(self):
            '''Train the model by running games'''
            # should also save the game type, network type and the training parameters
            time_steps = 0
            batches = 0
            print('training')
            self.create_workers()
            trajectories: Trajectories | None = None
            advantages: npt.NDArray | None = None
            while time_steps < self.total_time_steps:
                print('Steps', time_steps)
                batches += 1
                # trajectories = self.create_trajectories()
                trajectories = self.create_trajectories_parallel()
                max_cumulative_discount_reward = max(
                    trajectories.discount_cumulative_rewards)
                with open(f'{self.stats_location}/scores.csv',
                          'a',
                          encoding='utf8') as file:
                    for score in trajectories.scores:
                        file.write(f'{score}\n')

                print('Created Trajectories')
                time_steps += len(trajectories.observations)

                values, advantages = self.compute_value_advantage_estimates(
                    trajectories)
                print('Collected Data')
                # Write the game data to file
                # with open(f'{self.stats_location}/game_stats.csv', 'a', encoding='utf8') as file:
                #     for value, advantage, action in zip(values, advantages, trajectories.actions):
                #         file.write(f'{value},{advantage},{action}\n')
                for _ in range(self.updates_per_iteration):
                    print('Updating')
                    actor_loss, critic_loss = self.update_policy(
                        trajectories, advantages)
                    with open(f'{self.stats_location}/loss_stats.csv',
                              'a',
                              encoding='utf8') as file:
                        file.write(f'{actor_loss},{critic_loss}\n')
                    self.save()

                del trajectories
                gc.collect()

            self.stop_workers()

    def load(self, load_location: str):
        '''Load a pre-trained model'''
        self.actor = keras.models.load_model(
            f'{load_location}/actor',  custom_objects={'tf': tf})  # type: ignore
        self.critic = keras.models.load_model(
            f'{load_location}/critic',  custom_objects={'tf': tf})  # type: ignore
        self.count = 0

    def compute_action(self, game: Game):
        '''Compute the actions of a current game state of the loaded model'''
        input_vector = game.get_model_input()
        input_vector = tf.convert_to_tensor(
            np.array(input_vector).reshape(-1).reshape(
                1, len(input_vector)
            ))
        prob: ttf.Tensor1 = self.actor(input_vector)  # type: ignore
        prob = prob.numpy()[0]
        # print(prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample(1)

        action_num = action.numpy() < 2
        # print(action_num)
        # if self.count % 10  == 0:

        #     print(self.critic(input_vector).numpy()[0][0])
        self.count += 1
        return (action_to_action_array(action), self.critic(input_vector).numpy()[0][0])

    def train(self,
              game: type[Game],
              actor_network: type[Network],
              critic_network: type[Network],
              save_location='model',
              stats_location='ppo_stats'):
        '''Train the model on a specific game and network'''
        trainer = PPO.Trainer(game,
                              actor_network,
                              critic_network,
                              save_location,
                              stats_location=stats_location)
        trainer.train()

    actor: keras.Model
    critic: keras.Model

    def __init__(self) -> None:
        ...

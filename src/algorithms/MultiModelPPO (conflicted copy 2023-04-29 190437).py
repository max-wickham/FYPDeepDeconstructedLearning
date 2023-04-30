'''
One PPO per critic penultimate layer length
'''

import gc
import dataclasses
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import os
from typing import Type
import json

import numpy as np
import numpy.typing as npt
# Importing tensorflow modules this way makes type hinting work because reasons
import tensorflow as tf
import tensorflow_probability as tfp
import tensor_annotations.tensorflow as ttf
keras = tf.keras
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses

from src.interfaces.network import Network
from src.interfaces.game import Game


@dataclasses.dataclass
class Trajectories:
    '''Trajectory stored from one frame of the game'''
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    # probabilities outputted by each actor network
    probabilities: list[npt.NDArray]
    discount_cumulative_rewards: npt.NDArray
    scores: list[float]

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
    num_actors: int,
    load_location: str
):
    '''Create a set of trajectories on an individual thread'''
    ACTION_REPETITIONS = 2
    while True:
        if not stop_queue.empty():
            break
        if input_queue.empty():
            continue
        input_queue.get()

        # initialize variables
        observation_count = 0
        actions = []
        observations = []
        rewards = []
        probs: list[list[npt.NDArray]] = [[] for _ in range(num_actors)]
        scores = []
        discount_cumulative_rewards = []
        network_controller = MultiModelPPO.NetworkController(
            network_type,
            network_type,
            observation_dims,
            action_dims,
            num_actors,
            load=True
        )

        # load the saved network
        network_controller.load(load_location)

        while observation_count < num_observations:
            # create a new game
            game = game_type()
            done = False
            new_actions = []
            new_observations = []
            new_rewards = []
            new_probs = [[] for _ in range(num_actors)]
            prev_action = 0
            prev_prob: list[npt.NDArray] = []
            action_repetitions = 0
            frame = 0
            # run the game until it is done or we have enough observations
            while not done and observation_count < num_observations:
                frame += 1
                observation = game.get_model_input()
                observation = tf.reshape(
                    observation, shape=(1, len(observation)))
                if action_repetitions == 0:
                    observation_count += 1
                    prob, action = network_controller.get_prob_action(
                        observation)
                    prev_prob = prob
                    prev_action = action
                done, reward = game.step(
                    action_to_action_array(prev_action))
                # repeat the same action for ACTION_REPETITIONS frames
                if done or action_repetitions == 0 and not len(prev_prob) == 0:
                    new_observations.append(observation)
                    new_actions.append(prev_action)
                    new_rewards.append(reward)
                    for prob, prob_list in zip(prev_prob, new_probs):
                        prob_list.append(prob)
                action_repetitions = (
                    action_repetitions + 1) % ACTION_REPETITIONS
            scores.append(frame)
            new_discount_cumulative_rewards = []
            discount_cumulative_reward = 0
            # reverse the rewards and calculate the discount cumulative reward
            for reward in reversed(new_rewards):
                discount_cumulative_reward = reward + discount_cumulative_reward * 0.95
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))

            observations += new_observations
            actions += new_actions
            rewards += new_rewards
            for probs_list, new_probs_list in zip(probs, new_probs):
                probs_list += new_probs_list
            # reverse the discount cumulative rewards to correct order
            discount_cumulative_rewards += reversed(
                new_discount_cumulative_rewards)

        trajectories = Trajectories(
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            [np.array(prob) for prob in probs],
            np.array(discount_cumulative_rewards),
            scores
        )
        output_queue.put(trajectories)

        continue


def action_to_action_array(action: int) -> list[float]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


class MultiModelPPO:
    '''Model implementing the PPO training algorithm'''

    class NetworkController:
        '''Actor and critic models'''
        actors: list[Network]
        critic: Network

        def __init__(self,
                     actor_network: type[Network],
                     critic_network:  type[Network],
                     input_dims: int,
                     output_dims: int,
                     num_actors: int,
                     load=False) -> None:
            self.num_actors = num_actors
            if load:
                return
            self.learning_rate = 0.005

            # takes an observation and outputs a set of actions
            # square action dims to get output space

            self.actors: list[actor_network] = [actor_network(
                (input_dims), output_dims * output_dims) for _ in range(num_actors)]
            # takes an observation and outputs a value
            self.critic = critic_network((input_dims), num_actors)
            self.actor_optimiser = Adam(lr=self.learning_rate)
            self.critic_optimiser = Adam(lr=self.learning_rate)

        def load(self, load_location: str, load_critic=False):
            '''Load saved models'''
            print('Load')

            self.actors = [  # type: ignore
                keras.models.load_model(
                    f'{load_location}/actor_{i}',  custom_objects={'tf': tf}
                )  # type: ignore
                for i in range(self.num_actors)
            ]
            print('Actor Loaded')
            if load_critic:
                self.critic = keras.models.load_model(
                    f'{load_location}/critic',  custom_objects={'tf': tf})  # type: ignore

        def save(self, save_location: str):
            '''Save current models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            for i, actor in enumerate(self.actors):
                actor.model.save(f'{save_location}/actor_{i}')
            self.critic.model.save(f'{save_location}/critic')

        def get_prob_action(self, observation: npt.NDArray) -> tuple[list[npt.NDArray], int]:
            '''Returns the probability of
            each action combination and an action sampled from this distribution'''
            # Get the probability of each action from each sub agent
            probs = [actor(observation).numpy()[0] for actor in self.actors]
            # Multiply the probabilities together
            prob = tf.reduce_prod(probs, axis=0)
            # Normalise the probabilities
            prob = prob / (np.sum(prob) + 0.001)
            # Sample an action from the distribution
            dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            action = dist.sample(1)
            action = int(action.numpy()[0])
            if action >= 4:
                action = 3
            return probs, action

        def get_values(self, observations: npt.NDArray) -> ttf.Tensor1:
            '''Run the critic on a set of observations'''
            values = self.critic(observations, multi_dim=True)
            return values

    class Trainer:
        '''Train a PPO model'''

        def __init__(self,
                     game_type: type[Game],
                     actor_network: type[Network],
                     critic_network: type[Network],
                     save_location: str,
                     stats_location='ppo_stats1',
                     ) -> None:
            self.stats_location = stats_location
            # remove old stats if they exist
            if os.path.exists(stats_location):
                os.rmdir(stats_location)
            os.mkdir(stats_location)
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
            self.num_actors = 3

            self.updates_per_iteration = 2
            self.game_type = game_type
            self.gamma = 0.95
            self.clip = 0.2
            self.repeat_action_num = 3

            with open(f'{self.stats_location}/loss_stats.csv', 'a', encoding='utf8') as file:
                file.write(
                    f'{"".join([f"actor_loss{i}," for i in range(self.num_actors)])}critic_loss\n'
                )

            self.network_controller = MultiModelPPO.NetworkController(
                actor_network,
                critic_network,
                self.observation_dims,
                self.action_dims,
                self.num_actors,
            )

            # queues for messaging between workers
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

            # combine the trajectories from each worker
            trajectories = Trajectories(
                # list of observations
                np.concatenate(
                    tuple(trajectories.observations for trajectories in trajectories_list)),
                # list of actions
                np.array([action for trajectories in trajectories_list for action in list(
                    trajectories.actions)]),
                # list of rewalen(self.actors)rds
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.rewards)]),
                # list of lists of probabilities
                [
                    np.concatenate(
                        tuple(
                            np.reshape(trajectories.probabilities[i],
                                       (trajectories.probabilities[i].shape[0],
                                        1, trajectories.probabilities[i].shape[1])
                                       )
                            for trajectories in trajectories_list))
                    for i in range(self.num_actors)
                ],
                # list of cumulative rewards
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.discount_cumulative_rewards)]),
                # list of scores
                [score for trajectories in trajectories_list for score in trajectories.scores]
            )
            return trajectories

        def compute_value_advantage_estimates(self, trajectories: Trajectories
                                              ) -> tuple[ttf.Tensor1, ttf.Tensor1, ttf.Tensor1]:
            '''For a given set of trajectories calculate the set of advantage estimates'''
            # compute the value function for the set of trajectories using the critic
            # values are of dimension num_observations x num_actors
            values = self.network_controller.get_values(
                trajectories.observations)
            sum_values: ttf.Tensor1 = tf.reduce_sum(values, axis=1)

            values_shifted: ttf.Tensor1 = tf.identity(values)  # type: ignore
            # remove the first value from values_shifted
            values_shifted: ttf.Tensor1 = tf.slice(
                values_shifted, [1, 0], [-1, -1])
            # remove the last value from values
            values: ttf.Tensor1 = tf.slice(
                values, [0, 0], [int(values.shape[0])-1, -1])  # type: ignore
            advantages: ttf.Tensor1 = tf.subtract(values_shifted, values)
            # normalise the advantages, which is a num_observations x num_actors tensor
            advantages = tf.subtract(advantages, tf.reduce_mean(advantages, axis=0)) / \
                (tf.math.reduce_std(advantages, axis=0) + 1e-10)
            return values, sum_values, advantages

        def actor_loss(self,
                       new_probs: ttf.Tensor1,
                       current_probs: npt.NDArray,
                       actions: npt.NDArray,
                       advantages: ttf.Tensor1):
            '''Calculate the actor loss'''
            entropy = tf.reduce_mean(tf.math.negative(
                tf.math.multiply(new_probs, tf.math.log(new_probs))))
            surrogate_1 = []
            surrogate_2 = []
            current_probs = np.reshape(
                current_probs, (current_probs.shape[0], current_probs.shape[2]))
            current_probs_indexed = tf.convert_to_tensor(
                np.array([prob[action]
                          for prob, action in zip(current_probs, actions)]), dtype=tf.float32)
            new_probs_indexed = tf.gather_nd(
                new_probs,
                indices=tf.constant([[index, action] for index, action in enumerate(actions)]))
            ratios = tf.math.divide(new_probs_indexed, current_probs_indexed)
            ratios = tf.slice(ratios, [0], [len(ratios)-1])  # type: ignore
            surrogate_1 = tf.math.multiply(ratios, advantages)
            surrogate_2 = tf.math.multiply(tf.clip_by_value(
                ratios, 1.0 - self.clip, 1.0 + self.clip), advantages)
            surrogate_1 = tf.stack(surrogate_1)
            surrogate_2 = tf.stack(surrogate_2)
            loss = tf.math.negative(tf.reduce_mean(
                tf.math.minimum(surrogate_1, surrogate_2)) + 0.001 * entropy)
            return loss

        def update_policy(self, trajectories: Trajectories, advantage_estimates: ttf.Tensor1):
            '''
            Update the policy using the trajectories and advantage estimates
            Use stochastic gradient descent using ADAM
            '''
            discount_cumulative_rewards = tf.reshape(
                trajectories.discount_cumulative_rewards,
                (len(trajectories.discount_cumulative_rewards),)
            )
            with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
                for actor in self.network_controller.actors:
                    actor_tape.watch(actor.model.trainable_variables)
                critic_tape.watch(
                    self.network_controller.critic.model.trainable_variables)
                sub_probabilities: list[ttf.Tensor1] = [
                    actor(trajectories.observations,
                                training=True, multi_dim=True)
                    for actor in self.network_controller.actors
                ]  # type: ignore
                values = self.network_controller.critic(
                    trajectories.observations, training=True, multi_dim=True)
                values_summed = tf.math.reduce_sum(values, axis=1)
                values_summed = tf.reshape(values_summed, (len(values),))
                critic_loss = kls.mean_squared_error(
                    discount_cumulative_rewards, values_summed)
                actor_losses = [
                    self.actor_loss(
                        probabilities,
                        old_probabilities,
                        trajectories.actions,
                        advantage_estimates[:, i]
                    )
                    for probabilities, old_probabilities, i
                    in zip(sub_probabilities, trajectories.probabilities, range(self.num_actors))
                ]

            # calculate and apply gradients
            sub_actor_gradients = [
                actor_tape.gradient(
                    actor_loss, self.network_controller.actors[i].model.trainable_variables)
                for actor_loss, i in zip(actor_losses, range(self.num_actors))
            ]
            critic_gradients = critic_tape.gradient(
                critic_loss, self.network_controller.critic.model.trainable_variables)

            for actor, actor_gradients in zip(self.network_controller.actors, sub_actor_gradients):
                self.network_controller.actor_optimiser.apply_gradients(
                    zip(actor_gradients, actor.model.trainable_variables))
            self.network_controller.critic_optimiser.apply_gradients(
                zip(critic_gradients, self.network_controller.critic.model.trainable_variables))
            return actor_losses, critic_loss

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
                    self.num_actors,
                    self.save_location
                    ])
                self.workers.append(process)
                process.start()

        def stop_workers(self):
            '''Send a message to each worker to trigger a stop'''
            self.stop_queue.put(True)

        def train(self):
            '''Train the model by running games'''
            # should also save the game type, network type and the training parameters
            self.create_workers()

            # initialise variables
            time_steps = 0
            batches = 0
            trajectories: Trajectories | None = None
            advantages: ttf.Tensor1 | None = None

            # run for the specified number of time steps
            while time_steps < self.total_time_steps:
                batches += 1
                # create the trajectories using the parallel workers
                trajectories = self.create_trajectories_parallel()

                # save the scores, TODO move this to the trajectories class
                with open(f'{self.stats_location}/scores.csv',
                          'a',
                          encoding='utf8') as file:
                    for score in trajectories.scores:
                        file.write(f'{score}\n')

                time_steps += len(trajectories.observations)

                _, _, advantages = self.compute_value_advantage_estimates(
                    trajectories)
                # Write the game data to file
                # with open(f'{self.stats_location}/game_stats.csv', 'a', encoding='utf8') as file:
                #     for value, advantage, action in zip(values, advantages, trajectories.actions):
                #         file.write(f'{value},{advantage},{action}\n')
                for _ in range(self.updates_per_iteration):
                    actor_losses, critic_loss = self.update_policy(
                        trajectories, advantages)
                    # save the losses
                    with open(f'{self.stats_location}/loss_stats.csv',
                              'a',
                              encoding='utf8') as file:
                        file.write(
                            f'{"".join([f"{actor_loss}," for actor_loss in actor_losses])}\
                                {critic_loss}\n'
                        )
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
              stats_location='ppo_stats1'):
        '''Train the model on a specific game and network'''
        trainer = self.Trainer(game,
                              actor_network,
                              critic_network,
                              save_location,
                              stats_location=stats_location)
        trainer.train()

    actor: keras.Model
    critic: keras.Model

    def __init__(self) -> None:
        ...

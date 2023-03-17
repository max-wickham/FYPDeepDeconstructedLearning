'''
PPO Algorithm
Algorithm Pseudo Code https://spinningup.openai.com/en/latest/algorithms/ppo.html
'''
import os
from typing import NamedTuple, Any
import json
from multiprocessing import Pool, TimeoutError


import numpy as np
import numpy.typing as npt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes
import tensorflow.keras.losses as kls
import tensorflow_probability as tfp

from src.interfaces.network import Network
from src.interfaces.game import Game


class Trajectories(NamedTuple):
    '''Trajectory stored from one frame of the game'''
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    probabilities: npt.NDArray
    discount_cumulative_rewards: npt.NDArray


def action_to_action_array(action: int) -> list[int]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


class PPO:
    '''Model implementing the PPO training algorithm'''

    class NetworkController:
        '''Actor and critic models'''

        def __init__(self, network: type[Network], input_dims: int, output_dims: int) -> None:
            self.learning_rate = 0.005

            # takes an observation and outputs a set of actions
            # square action dims to get output space
            self.actor = network((input_dims), output_dims * output_dims)
            # takes an observation and outputs a value
            self.critic = network((input_dims), 1)
            self.actor_optimiser = Adam(lr=self.learning_rate)
            self.critic_optimiser = Adam(lr=self.learning_rate)

        def get_prob_action(self, observation: npt.NDArray[float]) -> tuple[npt.NDArray[float], int]:
            '''Returns the probability of
            each action combination and an action sampled from this distribution'''
            prob = self.actor(observation)
            prob = prob.numpy()[0]
            dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            action = dist.sample(1)
            return prob, int(action.numpy()[0])

        def get_values(self, observations: npt.NDArray[float]) -> npt.NDArray[float]:
            '''Run the critic on a set of observations'''
            values = self.critic(observations, multi_dim=True).numpy()
            values = np.reshape(values, (len(values,)))
            return values

    class Trainer:
        '''Train a PPO model'''

        def __init__(self, game_type: type[Game], network: type[Network]) -> None:

            self.action_dims = game_type.get_action_shape()
            self.observation_dims = game_type.get_input_shape()
            # self.total_time_steps = 10000000
            self.total_time_steps = 30000000
            self.observations_per_batch = 5000
            self.updates_per_iteration = 10
            self.game_type = game_type
            self.network_type = network
            self.gamma = 0.95
            self.clip = 0.2
            self.repeat_action_num = 6

            self.network_controller = PPO.NetworkController(
                network, self.observation_dims, self.action_dims)

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
                prev_action = None
                prev_prob = None
                action_repetitions = 0
                while not done and observation_count < self.observations_per_batch:
                    observation = game.get_model_input()
                    if action_repetitions == 0:
                        observation_count += 1
                        prob, action = self.network_controller.get_prob_action(observation)
                        prev_prob = prob
                        prev_action = action
                    done, reward = game.step(
                        action_to_action_array(action))
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
            ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
            '''For a given set of trajectories calculate the set of advantage estimates'''
            # compute the value function for the set of trajectories using the critic
            values: np.ndarray = self.network_controller.get_values(
                trajectories.observations)
            # compute the log estimates using the actor
            # log_probs = self.model.get_log_probs(observations, actions)
            advantages = trajectories.discount_cumulative_rewards - values
            advantages = (advantages - np.mean(advantages)) / \
                (np.std(advantages) + 1e-10)
            # TODO normalize advantages ?
            return values, advantages

        def actor_loss(self, new_probs: npt.NDArray,
                       current_probs: npt.NDArray,
                       actions: npt.NDArray,
                       advantages: npt.NDArray,
                       critic_loss: npt.NDArray):
            '''Calculate the actor loss'''
            entropy = tf.reduce_mean(tf.math.negative(
                tf.math.multiply(new_probs, tf.math.log(new_probs))))
            surrogate_1 = []
            surrogate_2 = []

            advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
            new_probs_indexed = tf.convert_to_tensor(
                np.array([prob[action] for prob, action in zip(new_probs, actions)]), dtype=tf.float32)
            current_probs_indexed = tf.convert_to_tensor(
                np.array([prob[action] for prob, action in zip(current_probs, actions)]), dtype=tf.float32)
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
                trajectories.discount_cumulative_rewards, (len(trajectories.discount_cumulative_rewards),))
            advantages = tf.reshape(
                advantage_estimates, (len(advantage_estimates),))
            current_probs = tf.reshape(trajectories.probabilities,
                                       (len(trajectories.probabilities), self.action_dims*self.action_dims))

            with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
                actor_tape.watch(self.network_controller.actor.model.trainable_variables)
                critic_tape.watch(self.network_controller.critic.model.trainable_variables)
                probabilities = self.network_controller.actor(
                    trajectories.observations, training=True, multi_dim=True)
                actor_tape.watch(self.network_controller.actor.model.trainable_variables)
                values = self.network_controller.critic(
                    trajectories.observations, training=True, multi_dim=True)
                values = tf.reshape(values, (len(values),))
                critic_loss = 0.5 * \
                    kls.mean_squared_error(discount_cumulative_rewards, values)
                actor_loss = self.actor_loss(
                    probabilities, trajectories.probabilities, trajectories.actions, advantage_estimates, critic_loss)

            actor_gradients = actor_tape.gradient(
                actor_loss, self.network_controller.actor.model.trainable_variables)
            critic_gradients = critic_tape.gradient(
                critic_loss, self.network_controller.critic.model.trainable_variables)
            self.network_controller.actor_optimiser.apply_gradients(
                zip(actor_gradients, self.network_controller.actor.model.trainable_variables))
            self.network_controller.critic_optimiser.apply_gradients(
                zip(critic_gradients, self.network_controller.critic.model.trainable_variables))
            return actor_loss, critic_loss

        def save(self, save_location: str):
            '''Save the trained models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            configs = {
                'game_type': str(type(self.game_type)),
                'network_type': str(type(self.network_type))
            }
            with open(f'{save_location}/configs.json','w', encoding='utf8') as file:
                file.write(json.dumps(configs))
            self.network_controller.actor.model.save(f'{save_location}/actor')
            self.network_controller.critic.model.save(f'{save_location}/critic')

        def train(self, save_location: str):
            '''Train the model by running games'''
            # should also save the game type, network type and the training parameters
            time_steps = 0
            batches = 0
            print('training')
            while time_steps < self.total_time_steps:
                batches += 1
                trajectories = self.create_trajectories()
                print('Created Trajectories')
                time_steps += len(trajectories.observations)

                values, advantages = self.compute_value_advantage_estimates(
                    trajectories)

                for _ in range(self.updates_per_iteration):
                    actor_loss, critic_loss = self.update_policy(
                        trajectories, advantages)

                self.save(save_location)

    def load(self, load_location: str):
        '''Load a pre-trained model'''
        self.actor = keras.models.load_model(
            f'{load_location}/actor',  custom_objects={'tf': tf})

    def compute_action(self, game: Game):
        '''Compute the actions of a current game state of the loaded model'''
        input_vector = game.get_model_input()
        input_vector = tf.convert_to_tensor(
                np.array(input_vector).reshape(-1).reshape(
                    1, len(input_vector)
                ))
        prob = self.actor(input_vector)
        prob = prob.numpy()[0]
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample(1)
        return action_to_action_array(action)

    def train(self, game: type[Game], network: type[Network], save_location='model'):
        '''Train the model on a specific game and network'''
        trainer = PPO.Trainer(game, network)
        trainer.train(save_location)

    def __init__(self) -> None:
        # actor for use when loading a model
        self.actor = None

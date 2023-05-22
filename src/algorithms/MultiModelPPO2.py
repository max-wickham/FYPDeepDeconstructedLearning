'''
Uses N PPO agents. The critic outputs N values that are summed to give the value function.
Each of the agents is trained using the corresponding value function node,
with the advantage function calculated by giving the a weight proportional
to the node value over the summed value of the discount cumulative reward.
A_n = V_n / V_t * DCR
The probability of each action outputted by the agents is then multiplied

This version is improved to contain better load and saving of the state
'''
from src.interfaces.game import Game
from src.interfaces.network import Network
from src.util.configs import DataclassSaveMixin
from contextlib import ExitStack
import gc
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
keras = tf.keras
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses


@dataclasses.dataclass
class Trajectories:
    '''Trajectory stored from one frame of the game'''
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    probabilities: list[npt.NDArray]
    discount_cumulative_rewards: npt.NDArray
    scores: list[float]
    advantages: npt.NDArray

    def get_observations(self):
        '''Returns observations'''
        return self.observations


def compute_advantage_estimates(observations: npt.NDArray,
                                network_controller: 'MultiModelPPO2.NetworkController',
                                discount_cumulative_rewards: npt.NDArray):
    '''Calculate the advantage estimates for a single game'''
    # Compute the critic value for each frame
    # Weight the discount cumulative reward by the critic value
    # Compute the difference between the two
    critic_values = network_controller.get_values(observations).numpy() + 0.01
    val_sum = np.sum(critic_values, axis=1)
    scaled_dcr = discount_cumulative_rewards / (val_sum)
    proportional_dcr = critic_values * scaled_dcr[:, np.newaxis]
    advantages = np.subtract(proportional_dcr, critic_values)
    normalised_advantages = (
        advantages - np.mean(advantages, axis=0)) / np.std(advantages, axis=0)
    normalised_advantages[np.isnan(normalised_advantages)] = 0
    return normalised_advantages


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

def create_trajectories_process(
    args: WorkerArgs
):
    '''Create a set of trajectories on an individual thread'''
    ACTION_REPETITIONS = 2
    while True:
        if not args.stop_queue.empty():
            break
        if args.input_queue.empty():
            continue
        args.input_queue.get()

        # initialize variables
        observation_count = 0
        actions = []
        observations = []
        rewards = []
        probs: list[list[npt.NDArray]] = [[] for _ in range(args.num_actors)]
        scores = []
        discount_cumulative_rewards = []
        advantages: None | npt.NDArray = None
        network_controller = MultiModelPPO2.NetworkController(
            args.actor_network,
            args.critic_network,
            args.observation_dims,
            args.action_dims,
            args.num_actors,
            load=True
        )

        # load the saved network
        network_controller.load(args.load_location)

        while observation_count < args.num_observations:
            # create a new game
            game = args.game_type()
            done = False
            new_actions = []
            new_observations = []
            new_rewards = []
            new_probs = [[] for _ in range(args.num_actors)]
            prev_action = 0
            prev_prob: list[npt.NDArray] = []
            action_repetitions = 0
            frame = 0
            # run the game until it is done or we have enough observations
            while not done and observation_count < args.num_observations:
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

            # calculate the values of each observation
            # calculate the difference in values of each observation
            # remove the last action, observation, reward and dcr
            observations += new_observations
            actions += new_actions
            rewards += new_rewards
            for probs_list, new_probs_list in zip(probs, new_probs):
                probs_list += new_probs_list
            # reverse the discount cumulative rewards to correct order
            discount_cumulative_rewards += reversed(
                new_discount_cumulative_rewards)
            # print(advantages)
            # print(new_advantages)
            new_advantages = compute_advantage_estimates(
                np.array(new_observations),
                network_controller,
                np.array(list(reversed(
                    new_discount_cumulative_rewards))))
            advantages = new_advantages if advantages is None else np.concatenate(
                (advantages, new_advantages))
        if advantages is not None:
            trajectories = Trajectories(
                np.array(observations),
                np.array(actions),
                np.array(rewards),
                [np.array(prob) for prob in probs],
                np.array(discount_cumulative_rewards),
                scores,
                advantages
            )
            args.output_queue.put(trajectories)

        continue


def action_to_action_array(action: int) -> list[float]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


class MultiModelPPO2:
    '''Model implementing the PPO training algorithm'''
    INFO = 'MultiModelPPO2 with product of sub agents probabilities and cyclic training'

    class NetworkController:
        '''Actor and critic models'''
        _actors: list[Network]
        _critic: Network
        LEARNING_RATE = 0.005

        def __init__(self,
                     actor_network: type[Network],
                     critic_network:  type[Network],
                     input_dims: int,
                     output_dims: int,
                     num_actors: int,
                     load=False) -> None:
            self._num_actors = num_actors
            if load:
                return
            self.input_dims = input_dims
            self.output_dims = output_dims
            # takes an observation and outputs a set of actions
            # square action dims to get output space
            self._actors: list[actor_network] = [actor_network(
                (input_dims), output_dims * output_dims) for _ in range(num_actors)]
            # takes an observation and outputs a value
            self._critic = critic_network((input_dims), num_actors)
            self.actor_optimisers = [
                Adam(lr=self.LEARNING_RATE) for _ in range(self._num_actors)]
            self.critic_optimiser = Adam(lr=self.LEARNING_RATE)

        @classmethod
        def load_saved_model(cls, load_location) -> 'MultiModelPPO2.NetworkController':
            '''Load a saved model without passing any configs'''
            network_controller = cls.__new__(cls)
            network_controller.load(load_location)
            network_controller.actor_optimisers = [
                Adam(lr=network_controller.LEARNING_RATE) for _ in range(network_controller._num_actors)]
            network_controller.critic_optimiser = Adam(lr=network_controller.LEARNING_RATE)
            return network_controller

        def load(self, load_location: str):
            '''Load saved models'''
            with open(f'{load_location}/model_configs.json', 'r', encoding='utf8') as file:
                model_configs = json.loads(file.read())
            self._num_actors = model_configs['num_actors']
            self.input_dims = model_configs['input_dims']
            self.output_dims = model_configs['output_dims']
            self._actors = [Network(
                model_configs['input_dims'],
                model_configs['output_dims']
            ) for i in range(model_configs['num_actors'])]
            for i, actor in enumerate(self._actors):
                actor.load(f'{load_location}/actor_{i}')
            self._critic = Network(
                model_configs['input_dims'],
                model_configs['output_dims']
            )
            self._critic.load(f'{load_location}/critic')

        def save(self, save_location: str):
            '''Save the current models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            with open(f'{save_location}/model_configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps(
                    {
                        'num_actors': self._num_actors,
                        'input_dims': self.input_dims,
                        'output_dims': self.output_dims,
                    }
                ))
            for i, actor in enumerate(self._actors):
                actor.model.save(f'{save_location}/actor_{i}')
            self._critic.model.save(f'{save_location}/critic')

        def get_prob_action(self, observation: npt.NDArray) -> tuple[list[npt.NDArray], int]:
            '''Returns the probability of
            each action combination and an action sampled from this distribution'''
            # Get the probability of each action from each sub agent
            probs = [actor(observation).numpy()[0] for actor in self._actors]
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
            values = self._critic(observations, multi_dim=True)
            # values = self._critic(
            #     tf.reshape(observations,
            #                (observations.shape[0], observations.shape[2])))
            return values

        @property
        def actors(self):
            '''Actors Getter'''
            return self._actors

        @property
        def critic(self):
            '''Critic Getter'''
            return self._critic

    class Trainer:
        '''Train a PPO model'''
        class Configs(DataclassSaveMixin):
            '''Training Parameters'''
            NUM_WORKERS: int = 7
            TOTAL_TIME_STEPS: int = 300000
            OBSERVATIONS_PER_BATCH: int = 3000
            NUM_AGENTS: int = 3
            UPDATES_PER_ITERATION: int = 2
            GAMMA: float = 0.99
            CLIP: float = 0.2
            REPEAT_ACTION_NUM: int = 3
            GAMES_TO_TRAIN_OVER: int = 2000

        class TrainingState(DataclassSaveMixin):
            '''Current Training State'''
            games_played: int = 0
            time_steps: int = 0

        def __init__(self,
                     game_type: type[Game],
                     actor_network: type[Network],
                     critic_network: type[Network],
                     save_location: str,
                     stats_location='ppo_stats1',
                     ) -> None:
            self.stats_location = stats_location
            self.save_location = save_location
            self.action_dims = game_type.get_action_shape()
            self.observation_dims = game_type.get_input_shape()
            self.actor_network_type = actor_network
            self.critic_network_type = critic_network
            self.game_type = game_type

            if not 'LOAD' in os.environ or os.environ['LOAD'] != 'true':
                self.configs = self.Configs()
                self.training_state = self.TrainingState()
                self.network_controller = MultiModelPPO2.NetworkController(
                    actor_network,
                    critic_network,
                    self.observation_dims,
                    self.action_dims,
                    self.configs.NUM_AGENTS,
                )
                if not os.path.exists(stats_location):
                    os.mkdir(stats_location)
                with open(f'{self.stats_location}/loss_stats.csv', 'w', encoding='utf8') as file:
                    file.write(
                        f'{"".join([f"actor_loss{i},"for i in range(self.configs.NUM_AGENTS)])}critic_loss\n'
                    )
                with open(f'{self.stats_location}/scores.csv', 'w', encoding='utf8') as file:
                    file.write(
                        ''
                    )
            else:
                self.configs = self.Configs.load(
                    f'{self.save_location}/training_configs.json')
                self.training_state = self.TrainingState.load(
                    f'{self.save_location}/training_state.json')
                self.network_controller = MultiModelPPO2.NetworkController.load_saved_model(
                    self.save_location)
                # reset time steps
                self.training_state.time_steps = 0

            # queues for messaging between workers
            self.task_queue = multiprocessing.Queue()
            self.response_queue = multiprocessing.Queue()
            self.stop_queue = multiprocessing.Queue()

            self.workers: list[multiprocessing.Process] = []

        def create_trajectories_parallel(self) -> Trajectories:
            '''Create a set of trajectories using parallel workers'''
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
                # list of rewards
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
                    for i in range(self.configs.NUM_AGENTS)
                ],
                # list of cumulative rewards
                np.array([reward for trajectories in trajectories_list for reward in list(
                    trajectories.discount_cumulative_rewards)]),
                # list of scores
                [score for trajectories in trajectories_list for score in trajectories.scores],
                # advantages 2d numpy array
                np.concatenate(
                    [trajectories.advantages for trajectories in trajectories_list])
            )
            return trajectories

        def actor_loss(self,
                       new_probs: ttf.Tensor1,
                       current_probs: npt.NDArray,
                       actions: npt.NDArray,
                       advantages: ttf.Tensor1):
            '''Calculate the actor loss'''
            # the entropy of the probabilities is the log of a probability times the probability
            entropy = tf.reduce_mean(tf.math.negative(
                tf.math.multiply(new_probs, tf.math.log(new_probs))))
            surrogate_1 = []
            surrogate_2 = []
            current_probs = np.reshape(
                current_probs, (current_probs.shape[0], current_probs.shape[2]))
            # index each probability with the action that was taken at that observation
            current_probs_indexed = tf.convert_to_tensor(
                np.array([prob[action]
                          for prob, action in zip(current_probs, actions)]), dtype=tf.float32)
            new_probs_indexed = tf.gather_nd(
                new_probs,
                indices=tf.constant([[index, action] for index, action in enumerate(actions)]))
            ratios = tf.math.divide(new_probs_indexed, current_probs_indexed)
            # # reduce length of ratios by one to match number of advantages
            # ratios = tf.slice(
            #     ratios, [0], [len(ratios)-1])  # type: ignore
            surrogate_1 = tf.math.multiply(ratios, advantages)
            surrogate_2 = tf.math.multiply(tf.clip_by_value(
                ratios, 1.0 - self.configs.CLIP, 1.0 + self.configs.CLIP), advantages)
            surrogate_1 = tf.stack(surrogate_1)
            surrogate_2 = tf.stack(surrogate_2)
            loss = tf.math.negative(tf.reduce_mean(
                tf.math.minimum(surrogate_1, surrogate_2)) + 0.001 * entropy)
            return loss

        def update_policy(self, trajectories: Trajectories):
            '''
            Update the policy using the trajectories and advantage estimates
            Use stochastic gradient descent using ADAM
            '''
            discount_cumulative_rewards = tf.reshape(
                trajectories.discount_cumulative_rewards,
                (len(trajectories.discount_cumulative_rewards),)
            )

            advantage_estimates: ttf.Tensor1 = tf.convert_to_tensor(
                trajectories.advantages, dtype=tf.float32)  # type: ignore
            actor_losses = []
            actor_tapes = []
            critic_loss = 0
            mod_games_played = self.training_state.games_played % self.configs.GAMES_TO_TRAIN_OVER
            index = int(
                mod_games_played / (self.configs.GAMES_TO_TRAIN_OVER / self.configs.NUM_AGENTS))
            actor = self.network_controller.actors[index]
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                actor_tape.watch(actor.model.trainable_variables)
                # watch the trainable variables of each actor
                for actor, actor_tape in zip(self.network_controller.actors, actor_tapes):
                    actor_tape.watch(actor.model.trainable_variables)
                # watch the trainable critic variables
                critic_tape.watch(
                    self.network_controller.critic.model.trainable_variables)
                # get a list of probabilities for each observation for each actor
                sub_probabilities: list[ttf.Tensor1] = [
                    actor(trajectories.observations,
                          training=True, multi_dim=True)
                    for actor in self.network_controller.actors
                ]  # type: ignore
                # get a set of state values using the critic for each actor
                values = self.network_controller.critic(
                    trajectories.observations, training=True, multi_dim=True)
                # calculate the sum of the sub values
                values_summed = tf.math.reduce_sum(values, axis=1)
                values_summed = tf.reshape(values_summed, (len(values),))
                # calculate the critic loss with respect to the discount cumulative rewards
                critic_loss = kls.mean_squared_error(
                    discount_cumulative_rewards, values_summed)

                actor_loss = self.actor_loss(
                    sub_probabilities[index],
                    trajectories.probabilities[index],
                    trajectories.actions,
                    advantage_estimates[:, index])

            # calculate and apply gradients
            actor_gradient = actor_tape.gradient(
                actor_loss, actor.model.trainable_variables)
            critic_gradients = critic_tape.gradient(
                critic_loss, self.network_controller.critic.model.trainable_variables)

            if not tf.math.is_nan(actor_loss).numpy() and not tf.math.is_nan(critic_loss).numpy():
                self.network_controller.actor_optimisers[index].apply_gradients(
                    zip(actor_gradient, actor.model.trainable_variables)
                )
                self.network_controller.critic_optimiser.apply_gradients(
                    zip(critic_gradients, self.network_controller.critic.model.trainable_variables))
            return actor_losses, critic_loss

        def save(self):
            '''Save the trained models'''
            if not os.path.exists(f'{self.save_location}'):
                os.mkdir(f'{self.save_location}')
            self.network_controller.save(self.save_location)
            with open(f'{self.save_location}/configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps({
                    'game_type': str(self.game_type),
                    'network_type': str(self.actor_network_type),
                    'algo_info' : MultiModelPPO2.INFO
                }))
            self.configs.save(f'{self.save_location}/training_configs.json')
            self.training_state.save(
                f'{self.save_location}/training_state.json')
            print(self.training_state)
            self.network_controller.save(self.save_location)

        def create_workers(self):
            '''Create Workers to collect training data'''
            BaseManager.register('Trajectories', Trajectories)
            manager = BaseManager()
            manager.start()
            self.workers = []

            for _ in range(self.configs.NUM_WORKERS):
                process = Process(target=create_trajectories_process, args=[
                    WorkerArgs(
                        int(self.configs.OBSERVATIONS_PER_BATCH /
                        self.configs.NUM_WORKERS),
                        self.task_queue,
                        self.response_queue,
                        self.stop_queue,
                        self.game_type,
                        self.observation_dims,
                        self.action_dims,
                        self.configs.NUM_AGENTS,
                        self.actor_network_type,
                        self.critic_network_type,
                        self.save_location,
                        self.configs.GAMMA
                    )
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
            trajectories: Trajectories | None = None
            # run for the specified number of time steps
            while self.training_state.time_steps < self.configs.TOTAL_TIME_STEPS:
                # create the trajectories using the parallel workers
                self.save()
                trajectories = self.create_trajectories_parallel()
                # save the scores
                with open(f'{self.stats_location}/scores.csv',
                          'a',
                          encoding='utf8') as file:
                    for score in trajectories.scores:
                        file.write(f'{score}\n')

                self.training_state.time_steps += len(
                    trajectories.observations)
                self.training_state.games_played += len(trajectories.scores)
                for _ in range(self.configs.UPDATES_PER_ITERATION):
                    actor_losses, critic_loss = self.update_policy(
                        trajectories)
                    # save the losses
                    with open(f'{self.stats_location}/loss_stats.csv',
                              'a',
                              encoding='utf8') as file:
                        file.write(
                            f'{"".join([f"{actor_loss}," for actor_loss in actor_losses])}\
                                {critic_loss}\n'
                        )
                del trajectories
                gc.collect()

            self.stop_workers()

    def get_values(self, game):
        '''Get critic values for a game state'''
        model_input = game.get_model_input()
        return self.network_controller.critic(
            tf.convert_to_tensor(
                np.array(model_input).reshape(-1).reshape(
                    1, model_input.shape[0]
                ))
        ).numpy()[0]

    # def get_values(self, game):
    #     model_input = game.get_model_input()
    #     return self.network_controller.get_values(model_input)

    def compute_action(self, game: Game):
        '''Compute the actions of a current game state of the loaded model'''
        input_vector = game.get_model_input()
        return self.network_controller.get_prob_action(input_vector)[0]

    @classmethod
    def train(cls,
              game: type[Game],
              actor_network: type[Network],
              critic_network: type[Network],
              save_location='model',
              stats_location='ppo_stats1'):
        '''Train the model on a specific game and network'''
        trainer = cls.Trainer(game,
                              actor_network,
                              critic_network,
                              save_location,
                              stats_location=stats_location)
        trainer.train()

    def __init__(self, load_location) -> None:
        self.network_controller = self.NetworkController.load_saved_model(
            load_location)

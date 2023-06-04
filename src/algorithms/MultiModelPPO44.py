'''
Uses N PPO agents. The critic outputs N values that are summed to give the value function.
Each of the agents is trained using the corresponding value function node,
with the advantage function calculated by giving the a weight proportional
to the node value over the summed value of the discount cumulative reward.
DCR_n = V_n / V_t * DCR

Uses another agents to decide on the best actions to be taken, as well as feeding the decisions
of agents in the previous frame as inputs.
'''
from src.interfaces.game import Game
from src.interfaces.network import Network
from src.networks.simple_network import MultiplicationLayer
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
    # num observations * num actions
    actions: npt.NDArray
    # num observations
    rewards: npt.NDArray
    # num observations * num actions * 4
    distributions: npt.NDArray
    votes: npt.NDArray
    # num observations
    discount_cumulative_rewards: npt.NDArray
    # num games
    scores: list[float]
    # num observations * num agents
    advantages: npt.NDArray

    def get_observations(self):
        '''Returns observations'''
        return self.observations


def compute_advantage_estimates(observations: npt.NDArray,
                                votes: npt.NDArray,
                                network_controller: 'MultiModelPPO4.NetworkController',
                                rewards,
                                discount_factor):
    '''Calculate the advantage estimates for a single game'''
    num_agents, num_actions = network_controller.num_actors, network_controller.action_dims
    # broadcast rewards to give reward for both each agent and each action agent combination
    rewards: npt.NDArray = np.array(rewards)
    rewards = np.reshape(rewards, (len(rewards),1))
    rewards = np.hstack([rewards] * (num_agents + num_actions * num_agents))
    # flatten the votes to be N by (num_actions * num_agents)
    votes: npt.NDArray = np.reshape(votes, (votes.shape[0], votes.shape[1] * votes.shape[2]))
    # remove the vote from a reward
    rewards[:, num_agents:] -= np.where(votes < 0.0, 0.0, votes)
    # calculate the discount cumulative rewards
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    discount_cumulative_rewards = np.zeros(rewards.shape[1])
    for i in reversed(range(len(rewards))):
        discount_cumulative_rewards = rewards[i] + discount_factor * discount_cumulative_rewards
        discounted_rewards[i] = discount_cumulative_rewards
    # N by num_agents
    critic_values = network_controller.get_values(observations).numpy() + 0.01
    # N by (num_agents + num_actions * num_agents)
    critic_values = np.hstack([critic_values] * (1 + num_actions ))
    dcr_sum = np.sum(discounted_rewards, axis=1)
    # dcr_sum = np.where(np.abs(dcr_sum) < 0.001, dcr_sum + 0.001 * np.sign(dcr_sum), dcr_sum)
    # scale the dcr by the critic values
    proportional_dcr = critic_values * discounted_rewards
    p_dcr_sum = np.sum(proportional_dcr, axis=1)
    filter_small_vals = lambda vals: \
        np.where(np.abs(vals) < 0.001, vals + 0.001 * np.sign(vals), vals)
    p_dcr_sum, dcr_sum = filter_small_vals(p_dcr_sum), filter_small_vals(dcr_sum)
    # scale the dcr such that the row sums is not affected by the scaling by critic values
    proportional_dcr = proportional_dcr / (p_dcr_sum[:, np.newaxis]) * (dcr_sum[:, np.newaxis])
    # compute the advantage estimates
    advantages = np.subtract(proportional_dcr, critic_values)
    # normalise the advantage estimates in each column
    normalised_advantages = (
        advantages - np.mean(advantages, axis=0)) / np.std(advantages, axis=0)
    # remove any erroneous nan values, should never occur
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

def create_trajectories(args: WorkerArgs) -> Trajectories | None:
        ACTION_REPETITIONS = 2
            # initialize variables
        observation_count = 0
        # num observations * num actions
        actions : list[list[int]] = []
        observations = []
        rewards = []
        # num observations * num actions * 4 ( mean var, vote mean var)
        distributions: list[list[npt.NDArray]] = []
        votes: list[list[float]] = []
        # distributions: list[list[npt.NDArray]] = [[] for _ in range(args.num_actors+1)]
        scores = []
        discount_cumulative_rewards = []
        advantages: None | npt.NDArray = None
        network_controller = MultiModelPPO4.NetworkController(
            args.actor_network,
            args.critic_network,
            args.observation_dims,
            args.action_dims,
            args.num_actors,
            load=True
        )

        # load the saved network
        network_controller.load(args.load_location)

        frame_actions = []
        frame_votes = []
        frame_dists = []
        while observation_count < args.num_observations:
            # create a new game
            game = args.game_type()
            done = False
            new_actions = []
            new_observations = []
            new_rewards = []
            new_distributions = []
            new_votes = []
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
                    frame_dists, frame_actions, frame_votes = network_controller.get_prob_action(
                        observation, with_full_votes=True)
                    # frame votes agent votes for each action appended
                done, reward = game.step(frame_actions)
                if done or action_repetitions == 0:
                    new_observations.append(observation)
                    new_actions.append(list(frame_actions))
                    new_rewards.append(reward)
                    new_distributions.append(frame_dists)
                    new_votes.append(frame_votes)
                action_repetitions = (
                    action_repetitions + 1) % ACTION_REPETITIONS
            scores.append(frame)
            new_discount_cumulative_rewards = []
            discount_cumulative_reward = 0
            # reverse the rewards and calculate the discount cumulative reward
            for reward in reversed(new_rewards):
                discount_cumulative_reward = reward + \
                    discount_cumulative_reward * args.discount_factor
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))

            # calculate the values of each observation
            # calculate the difference in values of each observation
            # remove the last action, observation, reward and dcr
            observations += new_observations
            actions += new_actions
            rewards += new_rewards
            distributions += new_distributions
            votes += new_votes
            # reverse the discount cumulative rewards to correct order
            discount_cumulative_rewards += reversed(
                new_discount_cumulative_rewards)
            new_advantages = compute_advantage_estimates(
                np.array(new_observations),
                np.array(new_votes),
                network_controller,
                new_rewards,
                args.discount_factor,
            )
            # num_agent * n * 5 -> num_agent * n * 1
            advantages = new_advantages if advantages is None else np.concatenate(
                (advantages, new_advantages))

        if advantages is not None:
            trajectories = Trajectories(
                np.array(observations),
                np.array(actions),
                np.array(rewards),
                np.array(distributions),
                np.array(votes),
                np.array(discount_cumulative_rewards),
                scores,
                advantages
            )
            return trajectories
        return None


def create_trajectories_process(
    args: WorkerArgs
):
    '''Create a set of trajectories on an individual thread'''
    while True:
        if not args.stop_queue.empty():
            break
        if args.input_queue.empty():
            continue
        args.input_queue.get()
        trajectories = create_trajectories(args)
        if trajectories is not None:
            args.output_queue.put(trajectories)
        continue


class MultiModelPPO4:
    '''Model implementing the PPO training algorithm'''
    INFO = '''
    MultiModelPPO43 with cyclic training,
    Uses a voting system to decide which agent to listen to,
    each agent gives an urgency score to its actions and the agent with the highest urgency score is used.
    The urgency score is subtracted from the reward for an agent however, encouraging agents to only take control when needed.

    Uses better prob calc for normal distributions
    '''

    class NetworkController:
        '''Actor and critic models'''
        _actors: list[Network]
        _critic: Network
        LEARNING_RATE = 0.005

        def __init__(self,
                     actor_network: type[Network],
                     critic_network:  type[Network],
                     input_dims: int,
                     action_dims: int,
                     num_actors: int,
                     load=False) -> None:
            self.num_actors = num_actors

            self.input_dims = input_dims
            self.action_dims = action_dims
            self.actor_network = actor_network
            self.critic_network = critic_network
            # For each action the agent should output mean and variance as well as probability of taking control
            self.actor_output_size = action_dims * 4
            self.actor_input_size = self.input_dims + self.actor_output_size * num_actors
            # feed the last actor decisions back into the networks
            self._actors: list[actor_network] = [actor_network(
                self.actor_input_size, self.actor_output_size) for _ in range(num_actors)]
            # takes an observation and outputs the value of the state
            self._critic = critic_network((input_dims), num_actors)
            self.actor_optimisers = [
                Adam(lr=self.LEARNING_RATE) for _ in range(self.num_actors)]
            self.critic_optimiser = Adam(lr=self.LEARNING_RATE)

            self.last_output = np.zeros(self.actor_output_size * num_actors)

        @classmethod
        def load_saved_model(cls, load_location) -> 'MultiModelPPO4.NetworkController':
            '''Load a saved model without passing any configs'''
            network_controller = cls.__new__(cls)
            network_controller.load(load_location)
            network_controller.last_output = np.zeros(
                network_controller.actor_output_size * network_controller.num_actors)
            network_controller.actor_optimisers = [
                Adam(lr=network_controller.LEARNING_RATE) for _ in range(network_controller.num_actors)]
            network_controller.critic_optimiser = Adam(
                lr=network_controller.LEARNING_RATE)
            return network_controller

        def load(self, load_location: str, use_current_model = False):
            '''Load saved models'''
            with open(f'{load_location}/model_configs.json', 'r', encoding='utf8') as file:
                model_configs = json.loads(file.read())
            self.num_actors = model_configs['num_actors']
            self.actor_output_size = model_configs['actor_output_size']
            self.actor_input_size = model_configs['actor_input_size']
            self.input_dims = model_configs['input_dims']
            self.action_dims = model_configs['action_dims']
            self.actor_output_size = self.action_dims * 4
            if use_current_model:
                self._actors = [self.actor_network(self.actor_input_size, self.actor_output_size) for _ in range(self.num_actors)]
                self._critic = self.critic_network(
                    self.input_dims,
                    self.num_actors
                )
            else:
                self._actors = [Network(
                    self.actor_input_size, self.actor_output_size)
                    for i in range(self.num_actors)]
                self._critic = Network(
                    self.input_dims,
                    self.num_actors
                )
            for i, actor in enumerate(self._actors):
                actor.load(f'{load_location}/actor_{i}')
            self._critic.load(f'{load_location}/critic')

        def save(self, save_location: str):
            '''Save the current models'''
            if not os.path.exists(f'{save_location}'):
                os.mkdir(f'{save_location}')
            with open(f'{save_location}/model_configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps(
                    {
                        'num_actors': self.num_actors,
                        'input_dims': self.input_dims,
                        'action_dims': self.action_dims,
                        'actor_output_size': self.actor_output_size,
                        'actor_input_size': self.actor_input_size,
                    }, indent=4))
            for i, actor in enumerate(self._actors):
                actor.model.save(f'{save_location}/actor_{i}')
            self._critic.model.save(f'{save_location}/critic')

        def get_prob_action(self, observation: npt.NDArray, with_full_votes = False) -> tuple[list[tuple[float, float]], list[int], list[tuple[float, int]]]:
            '''Returns the probability of
            each action combination and an action sampled from this distribution'''
            # Get the probability of each action from each sub agent
            if isinstance(observation, np.ndarray):
                observation = tf.convert_to_tensor(
                    observation, dtype=tf.float32) # type: ignore
                observation = tf.expand_dims(observation, 0)
            observation = np.concatenate(
                (observation.numpy()[0], self.last_output))
            # length is the number of actors
            actor_outputs = [actor(observation).numpy()[
                0] for actor in self._actors]
            # store the output to be fed back
            self.last_output = np.array(np.concatenate(actor_outputs))
            Vote, ActorIndex = float, int
            # length is the number of possible actions
            action_votes : list[tuple[Vote,ActorIndex]] = []
            # for each actor and each action whether it should take control
            actor_vote = []
            # actor_vote = np.array([0.0 for _ in range(self.num_actors)])
            for action_index in range(self.action_dims):
                votes = [
                    np.random.normal(actor_output[4*action_index] / 2, actor_output[4*action_index + 1] / 2)
                    for actor_output in actor_outputs
                ]
                actor_vote.append(votes)
                # actor_vote += np.array(votes)
                actor_index: ActorIndex = int(np.argmax(votes))
                vote: Vote = votes[actor_index]
                action_votes.append((vote, actor_index))
            actor_vote = np.array(actor_vote)
            # print(actor_votes)
            # take the prob distribution for the action for the actor index corresponding to each action
            # then sample the prob distribution to get an action of either 0 or 1
            action_distributions = [
                # select the actor corresponding to the correct actor index,
                # then select the distribution corresponding to the action index
                actor_outputs[action_vote[1]][action_index * 4+2: action_index * 4+4]
                for action_index, action_vote in enumerate(action_votes)
            ]
            actions = [
                tfp.distributions.Categorical(
                probs=distribution, dtype=tf.float32).sample(1)
                for distribution in action_distributions
            ]
            if not with_full_votes:
                return actor_outputs, actions, action_votes
            else:
                return actor_outputs, actions, actor_vote

        def get_values(self, observations: npt.NDArray) -> ttf.Tensor1:
            '''Run the critic on a set of observations'''
            values = self._critic(observations, multi_dim=True)
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
            # this will be multiplied by 1000
            TOTAL_TIME_STEPS: int = 600000
            OBSERVATIONS_PER_BATCH: int = 10000
            NUM_AGENTS: int = 3
            UPDATES_PER_ITERATION: int = 1
            GAMMA: float = 0.997
            CLIP: float = 0.2
            REPEAT_ACTION_NUM: int = 3
            GAMES_TO_TRAIN_OVER: int = 1000
            ENTROPY_SCALAR = 0.001
            VOTE_ENTROPY_SCALAR = 0.0005

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
                self.network_controller = MultiModelPPO4.NetworkController(
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
                # self.network_controller = MultiModelPPO4.NetworkController.load_saved_model(
                #     self.save_location)

                self.network_controller = MultiModelPPO4.NetworkController(
                    actor_network,
                    critic_network,
                    self.observation_dims,
                    self.action_dims,
                    self.configs.NUM_AGENTS,
                    load=True
                )
                # load the saved network
                self.network_controller.load(self.save_location, use_current_model=True)
                # reset time steps
                self.training_state.time_steps = 0

            # queues for messaging between workers
            self.task_queue = multiprocessing.Queue()
            self.response_queue = multiprocessing.Queue()
            self.stop_queue = multiprocessing.Queue()

            self.workers: list[multiprocessing.Process] = []

            self.worker_args =                         WorkerArgs(
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

            # tf.config.threading.set_intra_op_parallelism_threads(self.configs.NUM_WORKERS)
            # tf.config.threading.set_inter_op_parallelism_threads(self.configs.NUM_WORKERS)

        def create_trajectories_parallel(self) -> Trajectories:
            '''Create a set of trajectories using parallel workers'''

            if self.configs.NUM_WORKERS <= 1:
                return create_trajectories(
                    self.worker_args
                )
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
                # list of lists of distributions
                np.concatenate(
                    [
                        trajectories.distributions for trajectories in trajectories_list
                    ], axis = 0
                ),
                np.concatenate(
                    [
                        trajectories.votes for trajectories in trajectories_list
                    ], axis = 0
                ),
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
                       new_distributions: ttf.Tensor1,
                       current_distributions: npt.NDArray,
                       votes: npt.NDArray,
                       actions: npt.NDArray, # num observations by num actions
                       advantages: ttf.Tensor1):
            '''Calculate the actor loss'''
            # the entropy of the probabilities is the log of a probability times the probability
            losses = []
            for action_index in range(self.action_dims):
                action_new_dists = new_distributions[:,action_index*4+2:action_index*4+4]
                action_cur_dists = current_distributions[:,action_index*4+2:action_index*4+4]
                entropy = tf.reduce_mean(tf.math.negative(
                    tf.math.multiply(action_new_dists, tf.math.log(action_new_dists))))
                surrogate_1 = []
                surrogate_2 = []
                current_probs_indexed = tf.convert_to_tensor(
                    np.array([prob[int(action)]
                            for prob, action in zip(action_cur_dists, actions[:,action_index])]), dtype=tf.float32)
                new_probs_indexed = tf.gather_nd(
                    action_new_dists,
                    indices=tf.constant([[index, int(action)] for index, action in enumerate(actions[:,action_index])]))
                ratios = tf.math.divide(new_probs_indexed, current_probs_indexed)
                action_advantages = advantages[:,0]
                surrogate_1 = tf.math.multiply(ratios, action_advantages)
                surrogate_2 = tf.math.multiply(tf.clip_by_value(
                    ratios, 1.0 - self.configs.CLIP, 1.0 + self.configs.CLIP), action_advantages)
                surrogate_1 = tf.stack(surrogate_1)
                surrogate_2 = tf.stack(surrogate_2)
                loss = tf.math.negative(tf.reduce_mean(
                    tf.math.minimum(surrogate_1, surrogate_2)) + self.configs.ENTROPY_SCALAR * entropy)
                losses.append(loss)
            return sum(losses)

        def actor_voting_loss(self,
                       new_distributions: ttf.Tensor1,
                       current_distributions: npt.NDArray,
                       votes: npt.NDArray,
                       actions: npt.NDArray, # num observations by num actions
                       advantages: ttf.Tensor1
        ):
            losses = []
            new_distributions = tf.convert_to_tensor(
                    new_distributions, dtype=tf.float32) # type: ignore

            for action_index in range(self.action_dims):
                action_vote_new_dists = new_distributions[:,action_index*4:action_index*4+2]
                action_vote_cur_dists = current_distributions[:,action_index*4:action_index*4+2]
                entropy = tf.math.negative(
                    tf.math.log(tf.multiply(action_vote_cur_dists[:,1],action_vote_cur_dists[:,1])))
                entropy = tf.clip_by_value(entropy, 0, 1000)
                entropy = tf.reduce_mean(entropy)
                surrogate_1 = []
                surrogate_2 = []

                new_mean, new_std_dev = tf.split(action_vote_new_dists, num_or_size_splits=2, axis=1)
                new_mean, new_std_dev = tf.squeeze(new_mean, axis=-1) / tf.constant(2.0, dtype=tf.dtypes.float32), tf.squeeze(new_std_dev, axis=-1) / tf.constant(2.0, dtype=tf.dtypes.float32)
                curr_mean, curr_std_dev = tf.split(action_vote_cur_dists, num_or_size_splits=2, axis=1)
                curr_mean, curr_std_dev = tf.squeeze(curr_mean, axis=-1) / tf.constant(2.0, dtype=tf.dtypes.float32), tf.squeeze(curr_std_dev, axis=-1) / tf.constant(2.0, dtype=tf.dtypes.float32)
                # new_normal_dist = tfp.distributions.Normal(loc=new_mean, scale=new_std_dev + 0.0001)
                # curr_normal_dist = tfp.distributions.Normal(loc=curr_mean, scale=curr_std_dev + 0.0001)
                # new_pdf_values = new_normal_dist.prob(np.array(votes[:,action_index]))
                # curr_pdf_values = curr_normal_dist.prob(np.array(votes[:,action_index]))
                # log(π(a|s)) = -0.5 * ((a - μ)² / σ² + log(2πσ²))
                const = tf.constant(2*np.pi)
                vote = votes[:,action_index]
                log_p_new = (-0.5 * ((tf.math.square(vote - new_mean) / new_std_dev)
                    + tf.math.log(const*tf.math.square(new_std_dev))))
                log_p_old = (-0.5 * ((tf.math.square(vote - curr_mean) / curr_std_dev)
                    + tf.math.log(const*tf.math.square(curr_std_dev))))
                log_ratio = log_p_new - log_p_old
                ratios = tf.math.exp(log_ratio)

                # ratios = tf.math.divide(new_pdf_values, curr_pdf_values + 0.0000001)
                action_advantages = advantages[:,1 + action_index]
                surrogate_1 = tf.math.multiply(ratios, action_advantages)
                surrogate_2 = tf.math.multiply(tf.clip_by_value(
                    ratios, 1.0 - self.configs.CLIP, 1.0 + self.configs.CLIP), action_advantages)
                surrogate_1 = tf.stack(surrogate_1)
                surrogate_2 = tf.stack(surrogate_2)
                loss = tf.math.negative(tf.reduce_mean(
                    tf.math.minimum(surrogate_1, surrogate_2)) + self.configs.VOTE_ENTROPY_SCALAR * entropy)
                losses.append(loss)

            return sum(losses)


        def update_policy(self, trajectories: Trajectories):
            '''
            Update the policy using the trajectories and advantage estimates
            Use stochastic gradient descent using ADAM
            '''
            if True:
            # try:
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
                    mod_games_played / (self.configs.GAMES_TO_TRAIN_OVER / (self.configs.NUM_AGENTS)))
                actor = self.network_controller.actors[index]
                with tf.GradientTape(persistent=True) as actor_tape, tf.GradientTape(persistent=True) as critic_tape:
                    actor_tape.watch(actor.model.trainable_variables)
                    # watch the trainable variables of each actor
                    for actor, actor_tape in zip(self.network_controller.actors, actor_tapes):
                        actor_tape.watch(actor.model.trainable_variables)
                    # watch the trainable critic variables
                    critic_tape.watch(
                        self.network_controller.critic.model.trainable_variables)
                    # get a list of probabilities for each observation for each actor
                    # concatenate the probabilities to the inputs
                    dist_shape = trajectories.distributions.shape
                    obv_dists = np.concatenate((trajectories.observations,
                                    np.reshape(trajectories.distributions,
                                        (dist_shape[0], 1, dist_shape[1] * dist_shape[2])))
                                    , axis = 2)
                    sub_distributions = actor(obv_dists, training=True, multi_dim=True)
                    # get a set of state values using the critic for each actor
                    values=self.network_controller.critic(
                        trajectories.observations, training=True, multi_dim=True)
                    # calculate the sum of the sub values
                    values_summed=tf.math.reduce_sum(values, axis=1)
                    values_summed=tf.reshape(values_summed, (len(values),))
                    # calculate the critic loss with respect to the discount cumulative rewards
                    critic_loss=kls.mean_squared_error(
                        discount_cumulative_rewards, values_summed)

                    # here the sub probs will give the actual probs outputted by the switch at the last index
                    actor_loss=self.actor_loss(
                        sub_distributions,
                        trajectories.distributions[:,index,:],
                        trajectories.votes[:,:,index],
                        trajectories.actions,
                        tf.transpose(tf.gather(
                            tf.transpose(advantage_estimates),
                                [index] +
                                [index + (action_index + 1) * self.configs.NUM_AGENTS for action_index in range(self.action_dims)]
                            )
                        ),
                        # (index, index + self.configs.NUM_AGENTS)))
                        # tf.transpose(tf.gather(tf.transpose(advantage_estimates), (index, index + self.configs.NUM_AGENTS)))
                    )
                    actor_vote_loss = self.actor_voting_loss(
                        sub_distributions,
                        trajectories.distributions[:,index,:],
                        trajectories.votes[:,:,index],
                        trajectories.actions,
                        tf.transpose(tf.gather(
                            tf.transpose(advantage_estimates),
                                [index] +
                                [index + (action_index + 1) * self.configs.NUM_AGENTS for action_index in range(self.action_dims)]
                            )
                        ),
                    )
                    print('actor_vote_loss',actor_vote_loss)
                if tf.math.is_nan(actor_loss).numpy() or tf.math.is_nan(critic_loss).numpy() or tf.math.is_nan(actor_vote_loss).numpy():
                    raise Exception
                # calculate and apply gradients
                actor_gradient=actor_tape.gradient(
                    actor_loss, actor.model.trainable_variables)
                critic_gradients=critic_tape.gradient(
                    critic_loss, self.network_controller.critic.model.trainable_variables)
                self.network_controller.actor_optimisers[index].apply_gradients(
                    zip(actor_gradient, actor.model.trainable_variables)
                )
                actor.set_none_voting_freeze(True)
                actor_vote_gradients=actor_tape.gradient(
                    actor_vote_loss, actor.model.trainable_variables)
                self.network_controller.actor_optimisers[index].apply_gradients(
                    zip(actor_vote_gradients, actor.model.trainable_variables)
                )
                actor.set_none_voting_freeze(False)
                self.network_controller.critic_optimiser.apply_gradients(
                    zip(critic_gradients, self.network_controller.critic.model.trainable_variables))
                print('Good Loss')
                del actor_tape
                del critic_tape
                return actor_losses, critic_loss
            # except:
            #     ...
            #     print('Bad Loss')
            #     return [0 for _ in range(self.configs.NUM_AGENTS)], 0

        def save(self):
            '''Save the trained models'''
            if not os.path.exists(f'{self.save_location}'):
                os.mkdir(f'{self.save_location}')
            self.network_controller.save(self.save_location)
            with open(f'{self.save_location}/configs.json', 'w', encoding='utf8') as file:
                file.write(json.dumps({
                    'game_type': str(self.game_type),
                    'network_type': str(self.actor_network_type),
                    'algo_info': MultiModelPPO4.INFO
                }, indent=4))
            self.configs.save(f'{self.save_location}/training_configs.json')
            self.training_state.save(
                f'{self.save_location}/training_state.json')
            self.network_controller.save(self.save_location)

        def create_workers(self):
            '''Create Workers to collect training data'''
            if self.configs.NUM_WORKERS > 1:
                BaseManager.register('Trajectories', Trajectories)
                manager=BaseManager()
                manager.start()
                self.workers=[]

                for _ in range(self.configs.NUM_WORKERS):
                    process=Process(target=create_trajectories_process, args=[
                        self.worker_args
                    ])
                    self.workers.append(process)
                    process.start()

        def stop_workers(self):
            '''Send a message to each worker to trigger a stop'''
            if self.configs.NUM_WORKERS > 1:
                self.stop_queue.put(True)

        def train(self):
            '''Train the model by running games'''
            # should also save the game type, network type and the training parameters
            self.create_workers()
            # initialise variables
            trajectories: Trajectories | None=None
            # run for the specified number of time steps
            while self.training_state.time_steps < self.configs.TOTAL_TIME_STEPS:
                # create the trajectories using the parallel workers
                self.save()
                trajectories=self.create_trajectories_parallel()
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
                    actor_losses, critic_loss=self.update_policy(
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
        model_input=game.get_model_input()
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
        input_vector=game.get_model_input()
        _, actions, _ =self.network_controller.get_prob_action(input_vector)
        return actions

    @ classmethod
    def train(cls,
              game: type[Game],
              actor_network: type[Network],
              critic_network: type[Network],
              save_location='model',
              stats_location='ppo_stats1'):
        '''Train the model on a specific game and network'''
        trainer=cls.Trainer(game,
                              actor_network,
                              critic_network,
                              save_location,
                              stats_location=stats_location)
        trainer.train()

    def __init__(self, load_location) -> None:
        self.network_controller=self.NetworkController.load_saved_model(
            load_location)

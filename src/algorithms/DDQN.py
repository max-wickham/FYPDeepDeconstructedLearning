'''DDQN Implementation'''
from typing import NamedTuple, Type
import random
import multiprocessing

import numpy as np
import numpy.typing as npt
import tensorflow as tf
keras = tf.keras
kls = tf.keras.losses
import tensor_annotations.tensorflow as ttf

from src.interfaces.network import Network
from src.interfaces.game import Game, Reward, Done


class MemoryItem(NamedTuple):
    '''Tuple that stores memory of an individual frame'''
    state: npt.NDArray
    new_state: npt.NDArray
    action: int
    reward: Reward
    done: Done


def action_to_action_array(action: int) -> list[float]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


def collect_training_data(
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
        # actions = []
        # observations = []
        # rewards = []
        # probs = []
        # scores = []
        # discount_cumulative_rewards = []


        network_controller = DDQN.NetworkController(
            network_type, observation_dims, action_dims, load=True)
        network_controller.load(load_location)
        while observation_count < num_observations:
            game = game_type()
            done = False
            new_actions = []
            new_observations = []
            new_rewards = []
            new_probs = []
            prev_action = 0
            prev_prob = None
            action_repetitions = 0
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
                    # prob, action = network_controller.get_prob_action(
                    #     observation)
                    q_vals = network_controller.get_actions_q_vals(observation)
                    action = int(np.argmax(q_vals))
                    # prev_prob = prob
                    prev_action = action
                done, reward = game.step(
                    action_to_action_array(prev_action))
                if done or action_repetitions == 0:
                    new_observations.append(observation)
                    new_actions.append(prev_action)
                    new_rewards.append(reward)
                    new_probs.append(prev_prob)
                action_repetitions = (action_repetitions + 1) % 4
            scores.append(frame)
            new_discount_cumulative_rewards = []
            discount_cumulative_reward = 0
            for reward in reversed(new_rewards):
                discount_cumulative_reward = reward + discount_cumulative_reward * 0.95
                new_discount_cumulative_rewards.append(
                    (discount_cumulative_reward))

            observations += new_observations
            actions += new_actions
            rewards += new_rewards
            probs += new_probs
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
            scores
        )
        print('Putting trajectories on queue')
        output_queue.put(trajectories)

        continue

class DDQN:
    '''Double Deep Q Learning Implementation'''

    class NetworkController:
        '''Manage the neural network'''

        def __init__(self, network_type: type[Network], input_dims: int, output_dims: int, load=True) -> None:
            if load:
                return
            self.network = network_type(input_dims, output_dims * output_dims)
            self.network_type = network_type
            self.input_dims = input_dims
            print('input dims', self.input_dims)
            self.output_dims = output_dims

        def load(self, load_location : str):
            '''Load a saved model'''
            self.network : Network = keras.models.load_model(
                f'{load_location}',  custom_objects={'tf': tf})  # type: ignore

        def get_actions_q_vals(self, input_state: npt.NDArray) -> npt.NDArray:
            '''Return the q-val of each action type'''
            actions = self.network.model.call(
                tf.convert_to_tensor(
                    np.array(input_state).reshape(-1).reshape(1, self.input_dims)),
                training=False
            )
            return actions.numpy()[0]

        def get_random_action(self):
            '''Random between 0 and 3 inclusive'''
            return np.random.randint(0, 4)

        def reward_for_action(self, input_state, action: int) -> float:
            '''Output the reward for taking a given action in a give state'''
            return self.get_actions_q_vals(input_state)[action]

        def fit(self, states, targets):
            '''Fit the model to a set of q-vals and input states'''
            self.network.model.fit(
                np.array(
                    [tf.convert_to_tensor(np.array(
                        state).reshape(-1).reshape(1, self.input_dims)) for state in states]
                ).reshape(len(states), self.input_dims),
                np.array(targets).reshape(len(targets), len(targets[0])),
                epochs=3,
                verbose=0
            )

        def clone(self, network_controller: 'DDQN.NetworkController'):
            '''Clone the weights of another model'''
            self.network.model.set_weights(network_controller.network.model.get_weights())

        def save(self, save_location: str):
            '''Save the model at the given location'''
            self.network.model.save(f'{save_location}')

    class Trainer:
        '''Class for training a DDQN implementation using a generic game interface'''

        INIT_MEMORY = 100
        # update interval to use target network
        TARGET_C = 3000
        #number of frames to run
        NUM_FRAMES = 10000
        MAX_MEM_FRAMES = 600

        def __init__(self,
                     game_type: type[Game],
                     network_type: type[Network],
                     save_location: str) -> None:
            self.action_shape = game_type.get_action_shape()
            print(self.action_shape)
            self.observation_space = game_type.get_input_shape()
            self.network_controller = DDQN.NetworkController(
                network_type, self.observation_space, self.action_shape)
            self.target_network_controller = DDQN.NetworkController(
                network_type, self.observation_space, self.action_shape)
            self.game_type = game_type

            self.memory = []
            self.save_location = save_location

        def epsilon_greedy(self, eps: float, input_state: npt.NDArray) -> int:
            ''' Returns an decision using epsilon greedy strategy'''
            if np.random.random() < eps:
                # exploration
                return self.network_controller.get_random_action()
            else:
                # exploitation
                return int(np.argmax(self.network_controller.get_actions_q_vals(input_state)))

        def experience_replay(self,
                              memory: list['MemoryItem'], discount_factor: float, batch_size: int):
            '''Fit the model to a mini batch of states from memory'''

            if len(memory) < batch_size:
                print('Memory size too small')
                return

            states: list[np.ndarray] = []
            targets: list[npt.NDArray] = []

            # sample a batch
            minibatch = random.sample(memory, batch_size)

            # iterate through batch

            for state, new_state, action, reward, done in minibatch:

                target = reward

                # if game not over, target q val includes discounted future utility
                # we use a cloned model to predict here for stability. Model is changed every C frames
                # we use the online model to choose best action to deal with overestimation error (Double-Q learning)
                if not done:
                    best_actions = int(np.argmax(
                        self.network_controller.get_actions_q_vals(new_state)))
                    target = reward + discount_factor * \
                        self.target_network_controller.reward_for_action(
                            new_state, best_actions)

                current_q_values = self.network_controller.get_actions_q_vals(
                    new_state)
                current_q_values[action] = target

                # add to states
                states.append(state)

                # add to targets
                targets.append(current_q_values)
            # print(np.min(np.array(targets)))
            self.network_controller.fit(np.array(states), np.array(targets))

        def prefill_mem(self):
            '''Prefill memory'''
            while len(self.memory) < self.INIT_MEMORY:
                # reset game
                game = self.game_type()
                state = game.get_model_input()
                done = False

                while not done:
                    game_action = self.network_controller.get_random_action()
                    done, reward = game.step(
                        action_to_action_array(game_action))
                    new_state = game.get_model_input()
                    self.memory.append(MemoryItem(
                        state, new_state, game_action, reward, done))
                    state = new_state

        def run_frames(self):
            '''Run through the game to train the model'''
            total_frames = 0
            eps = 1
            eps_min = 0.2
            eps_linear_decay = (eps-eps_min)/(self.NUM_FRAMES/100)

            discount_factor = 0.9
            batch_size = 128
            num_updates = 0
            discount_factor = 0.9

            game_mem: list[MemoryItem] = []
            num_games = 0
            while total_frames < self.NUM_FRAMES:
                print(total_frames  )
                num_games += 1
                game = self.game_type()
                state = game.get_model_input()

                done = False
                game_frames = 0
                while not done:
                    game_frames += 1
                    action = self.epsilon_greedy(eps, state)

                    done, reward = game.step(action_to_action_array(action))
                    new_state = game.get_model_input()
                    game_mem.append(MemoryItem(
                        state, new_state, action, reward, done))
                    if done:
                        total_frames += game_frames
                        break

                    state = new_state
                    eps = max(eps - eps_linear_decay, eps_min)

                    if len(game_mem) > batch_size:
                        self.memory += game_mem
                        if len(self.memory) > self.MAX_MEM_FRAMES:
                            self.memory = self.memory[-1*self.MAX_MEM_FRAMES:]
                        game_mem.clear()
                        num_updates += batch_size
                        self.experience_replay(
                            self.memory, discount_factor, batch_size)
                        if num_updates > self.TARGET_C:
                            print('Cloning and Saving')
                            num_updates = 0

                            self.target_network_controller.clone(
                                self.network_controller)
                            self.network_controller.save(self.save_location)

        def train(self):
            '''Train a model'''
            print('Prefilling memory')
            self.prefill_mem()
            print('Running Frames')
            self.run_frames()



    def train(self,
              game: type[Game],
              actor_network: type[Network],
              save_location='model'):
        '''Train the model on a specific game and network'''
        trainer = DDQN.Trainer(game, actor_network, save_location)
        trainer.train()

    def load(self, load_location: str):
        '''Load a pre-trained model'''
        self.model = keras.models.load_model(
            f'{load_location}',  custom_objects={'tf': tf})  # type: ignore

    def compute_action(self, game: Game):
        '''Compute the actions of a current game state of the loaded model'''
        input_vector = game.get_model_input()
        input_vector = tf.convert_to_tensor(
            np.array(input_vector).reshape(-1).reshape(
                1, len(input_vector)
            ))
        q_vals = self.model(input_vector).numpy()[0]
        action = int(np.argmax(q_vals))

        return action_to_action_array(action), q_vals[action]

    model : keras.Model

    def __init__(self) -> None:
        pass

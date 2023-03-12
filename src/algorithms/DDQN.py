'''DDQN Implementation'''
from typing import NamedTuple
import random

import numpy as np
import tensorflow as tf
from progress.bar import Bar

from src.interfaces.network import Network
from src.interfaces.game import Game


class MemoryItem(NamedTuple):
    '''Tuple that stores memory of an individual frame'''
    state: np.ndarray
    new_state: np.ndarray
    action: int
    reward: float
    done: bool


def action_to_action_array(action: int) -> list[int]:
    '''Convert the combination action index to a action array'''
    # print(action.numpy()[0])
    return [
        1 if action < 2 else 0,
        1 if action % 2 == 1 else 0
    ]


class DDQN:
    '''Double Deep Q Learning Implementation'''

    class NetworkController:
        '''Manage the neural network'''

        def __init__(self, network: type[Network], input_dims: int, output_dims: int) -> None:
            self.network = network
            self.input_dims = input_dims
            self.output_dims = output_dims

        def get_actions_q_vals(self, input_state: np.array) -> np.array:
            '''Return the q-val of each action type'''
            actions = self.network.call(
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
                ).reshape(len(states), self.input_dims*self.input_dims),
                np.array(targets).reshape(len(targets), len(targets[0])),
                epochs=3,
                verbose=0
            )

    class Trainer:
        '''Class for training a DDQN implementation using a generic game interface'''

        def __init__(self, game_type: type[Game], network_type: type[Network]) -> None:
            self.action_shape = game_type.get_action_shape()
            self.observation_space = game_type.get_input_shape()
            self.network_controller = DDQN.NetworkController(
                network_type, self.observation_space, self.action_shape)
            self.target_network_controller = DDQN.NetworkController(
                network_type, self.observation_space, self.action_shape)

        def epsilon_greedy(self, eps: float, input_state: np.ndarray) -> int:
            ''' Returns an decision using epsilon greedy strategy'''
            if np.random.random() < eps:
                # exploration
                return self.network_controller.get_random_action()
            else:
                # exploitation
                return np.argmax(self.network_controller.get_actions_q_vals(input_state))

        def experience_replay(self,
                              memory: list['MemoryItem'], discount_factor: float, batch_size: int):
            '''Fit the model to a mini batch of states from memory'''

            if len(memory) < batch_size:
                print('Memory size too small')
                return

            states: list[np.ndarray] = []
            targets: list[float] = []

            # sample a batch
            minibatch = random.sample(memory, batch_size)

            # iterate through batch
            for state, new_state, action, reward, done in minibatch:

                target = reward

                # if game not over, target q val includes discounted future utility
                # we use a cloned model to predict here for stability. Model is changed every C frames
                # we use the online model to choose best action to deal with overestimation error (Double-Q learning)
                if not done:
                    best_actions = np.argmax(
                        self.network_controller.get_actions_q_vals(new_state))
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
            print('Prefilling Memory')
            with Bar('Processing', max=self.INIT_MEMORY) as bar:
                while len(self.memory) < self.INIT_MEMORY:
                    # reset game
                    self.model_controller.initialise()
                    self.game = self.game_type()
                    state = self.game.generate_input_vector()
                    buffer_input = [
                        np.zeros(INPUT_DIMENSIONS)
                    ] * PREV_FRAMES
                    buffer_input.append(state)
                    buffer_input = buffer_input[-1 * PREV_FRAMES:]
                    # state = np.array(buffer_input).reshape(-1)
                    state = tuple(buffer_input)
                    done = False

                    while not done:
                        game_action = self.network_controller.get_random_action()
                        done, reward = self.network_controller.process(
                            self.game, convert_action_to_game_action(game_action))
                        new_state = self.game.generate_input_vector()
                        buffer_input.append(new_state)
                        buffer_input = buffer_input[-1 * PREV_FRAMES:]
                        # new_state = np.array(buffer_input).reshape(-1)
                        new_state = tuple(buffer_input)
                        # self.memory.append((np.copy(state),np.copy(new_state), game_action,reward,done))
                        self.memory.append(MemoryItem(
                            state, new_state, game_action, reward, done))
                        bar.next()
                        state = new_state

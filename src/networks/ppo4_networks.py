'''Implementation of a simple neural network for testing'''
from src.interfaces.network import Network
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes
keras = tf.keras
K = keras.backend
Layer = keras.layers.Layer
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses



class PPO4ActorNetwork(Network):
    '''Simple Neural network'''

    def __init__(self,
                 input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        num_actions = output_dim // 4
        self.none_voting_layers = []

        input_layer = keras.layers.Input(input_dims)
        dense_layer1 = keras.layers.Dense(10, activation="relu")(input_layer)
        dense_layer2 = keras.layers.Dense(8, activation="relu")(dense_layer1)
        dense_layer3 = keras.layers.Dense(8, activation="relu")(dense_layer2)

        self.none_voting_layers += [input_layer, dense_layer1, dense_layer2, dense_layer3]

        output_layers = []
        for _ in range(num_actions):
            final_dense_action = keras.layers.Dense(5, activation="relu")(dense_layer3)
            output_layer_action = keras.layers.Dense(
            2, activation="softmax")(final_dense_action)


            penultamate_dense_vote = keras.layers.Dense(5, activation="relu")(dense_layer3)
            final_dense_vote = keras.layers.Dense(3, activation="relu")(penultamate_dense_vote)
            output_layer_vote = keras.layers.Dense(
            2, activation="sigmoid")(final_dense_vote)
            # output_layer_vote_increased = keras.layers.Lambda(lambda x: x + 0.00001)(output_layer_vote)
            output_layers += [output_layer_vote,output_layer_action]
            self.none_voting_layers += [final_dense_action, output_layer_action]

        output_layer = keras.layers.Concatenate()(output_layers)

        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        # self.optimiser = tf.keras.optimizers.RMSprop(
        #     learning_rate=learning_rate, momentum=0.9, epsilon=0.01)
        # self.model.compile(loss="mse", optimizer=self.optimiser)

    def set_none_voting_freeze(self, freeze: bool):
        '''Set the freeze state of none voting layers'''
        for layer in self.none_voting_layers:
            layer.trainable = freeze

    def __call__(self, input_data: npt.NDArray, training=False, multi_dim=False) -> ttf.Tensor1[ttf.float32, axes.Height]:
        return self.model.call(
            tf.convert_to_tensor(
                np.array(input_data).reshape(-1).reshape(
                    1 if not multi_dim else len(input_data), self.input_dims
                )), training=training
        )


class PPO4CriticNetwork(Network):
    '''Simple Neural network'''

    def __init__(self,
                 input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        input_layer = keras.layers.Input(input_dims)
        dense_layer1 = keras.layers.Dense(10, activation="relu")(input_layer)
        dense_layer2 = keras.layers.Dense(10, activation="relu")(dense_layer1)
        dense_layer3 = keras.layers.Dense(5, activation="relu")(dense_layer2)
        output_layer = keras.layers.Dense(
            output_dim, activation=None)(dense_layer2)
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        # self.optimiser = tf.keras.optimizers.RMSprop(
        #     learning_rate=learning_rate, momentum=0.9, epsilon=0.01)
        # self.model.compile(loss="mse", optimizer=self.optimiser)

    def __call__(self, input_data: npt.NDArray, training=False, multi_dim=False) -> ttf.Tensor1[ttf.float32, axes.Height]:
        return self.model.call(
            tf.convert_to_tensor(
                np.array(input_data).reshape(-1).reshape(
                    1 if not multi_dim else len(input_data), self.input_dims
                )), training=training
        )

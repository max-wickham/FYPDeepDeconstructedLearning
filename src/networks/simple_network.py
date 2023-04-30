'''Implementation of a simple neural network for testing'''
from src.interfaces.network import Network
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes
# from tensorflow impor t keras
keras = tf.keras
Adam = tf.keras.optimizers.Adam
kls = tf.keras.losses


class SimpleActorNetwork(Network):
    '''Simple Neural network'''

    def __init__(self,
                 input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        input_layer = keras.layers.Input(input_dims)
        dense_layer1 = keras.layers.Dense(10, activation="relu")(input_layer)
        dense_layer2 = keras.layers.Dense(5, activation="relu")(dense_layer1)
        dense_layer3 = keras.layers.Dense(5, activation="relu")(dense_layer2)
        output_layer = keras.layers.Dense(
            output_dim, activation="softmax")(dense_layer3)
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


class SimpleCriticNetwork(Network):
    '''Simple Neural network'''

    def __init__(self,
                 input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        input_layer = keras.layers.Input(input_dims)
        dense_layer1 = keras.layers.Dense(10, activation="relu")(input_layer)
        dense_layer2 = keras.layers.Dense(5, activation="relu")(dense_layer1)
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


class SimpleDDQNNetwork(Network):
    '''Simple Neural network'''

    def __init__(self,
                 input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        input_layer = keras.layers.Input(input_dims)
        dense_layer1 = keras.layers.Dense(10, activation="relu")(input_layer)
        dense_layer2 = keras.layers.Dense(5, activation="relu")(dense_layer1)
        output_layer = keras.layers.Dense(
            output_dim, activation=None)(dense_layer2)
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        self.optimiser = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate, momentum=0.9, epsilon=0.01)
        self.model.compile(loss="mse", optimizer=self.optimiser)

    def __call__(self,
                 input_data: npt.NDArray,
                 training=False,
                 multi_dim=False) -> ttf.Tensor1[ttf.float32, axes.Height]:
        return self.model.call(
            tf.convert_to_tensor(
                np.array(input_data).reshape(-1).reshape(
                    1 if not multi_dim else len(input_data), self.input_dims
                )), training=training
        )

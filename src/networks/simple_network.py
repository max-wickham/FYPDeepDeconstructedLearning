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

class MultiplicationLayer(Layer):
    '''Does a matrix multiplication of the input with itself and then appends the input'''
    def __init__(self, input_dims, **kwargs):
        self.input_dims = input_dims
        super(MultiplicationLayer, self).__init__(**kwargs)

    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'input_dims': self.input_dims,
    #     })
    #     return config

    # def build(self, input_shape):
    #     '''No trainable variables'''
    #     super(MultiplicationLayer, self).build(input_shape) #

    def call(self, inputs, *args, **kwargs):
        '''Multiply the inputs together to form a matrix
        and then concatenate with the original input'''
        # n * 8 | 8
        print('layer',inputs.shape)
        multi_dim = not (len(inputs.shape) == 1 or inputs.shape[0] is None)
        length = inputs.shape[1 if len(inputs.shape) > 1 else 0]
        if inputs.shape[0] is None:
            # None * 8
            inputs = tf.reshape(inputs,(length,))
        # n * 8 * 1 | 8 * 1
        inputs_reshaped = tf.expand_dims(inputs, axis=len(inputs.shape))
        if len(inputs_reshaped.shape) == 2:
            # 8 * 1
            inputs_reshaped = tf.expand_dims(inputs_reshaped, axis=0)
            # 1 * 8 * 1
        inputs_transposed = tf.reshape(inputs_reshaped, (inputs_reshaped.shape[0], 1, length))
        multiplication_matrix  = tf.matmul(inputs_reshaped, inputs_transposed)
        multiplication_array = tf.reshape(multiplication_matrix,
          (inputs_reshaped.shape[0], length*length,1))
        concatenated_response = tf.concat([inputs_reshaped, multiplication_array], 1)
        outputs = tf.reshape(concatenated_response,
            (inputs_reshaped.shape[0],length*length + length))
        if multi_dim:
            outputs = tf.reshape(outputs, (outputs.shape[0], 1, outputs.shape[1]))
        print(outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        if len(input_shape) > 1:
            if input_shape[0] is not None:
                return (input_shape[0],input_shape[1] * input_shape[1] + input_shape[1])
        return (1,input_shape[0] * input_shape[0] + input_shape[0])

class SimpleSwitchNetwork(Network):
    '''Used to combine the action prob distributions of a set of sub agents'''

    def __init__(self,
                input_dims: int,
                output_dim: int,
                learning_rate=0.005, train=True, load=False, model_file=''):
        super().__init__(input_dims, output_dim, learning_rate, train, load, model_file)
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Input(input_dims))
        # model.add(keras.layers.Input(input_dims))
        # model.add(MultiplicationLayer(input_dims))
        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dense(
            output_dim, activation="softmax"))
        self.model = model
        # multiplication_layer = MultiplicationLayer(input_dims)
        # dense_layer = keras.layers.Dense(10, activation="relu")(multiplication_layer)
        # output_layer = keras.layers.Dense(
        #     output_dim, activation="softmax")(dense_layer)
        # self.model = keras.Model(inputs=multiplication_layer, outputs=output_layer)


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

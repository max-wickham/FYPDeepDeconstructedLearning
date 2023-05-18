'''Neural Network Interface'''
from abc import ABC, abstractmethod

import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes
import numpy as np
import tensorflow as tf
keras = tf.keras


class Network(ABC):
    '''Generic Neural Network Base Class'''

    model: tf.keras.Model

    def __init__(self, input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.train = train
        self.model_file = model_file

    def save(self, location: str):
        '''save a model'''
        self.model.save(location)

    def load(self, location: str):
        '''load a model'''
        self.model = keras.models.load_model(location,  custom_objects={
                                'tf': tf}, compile=False)  # type: ignore

    def __call__(self,
                 input_data: np.ndarray,
                 training=False,
                 multi_dim=False) -> ttf.Tensor1[ttf.float32, axes.Height]:
        '''Run the model on a set of input data'''
        return self.model.call(
            tf.convert_to_tensor(
                np.array(input_data).reshape(-1).reshape(
                    1 if not multi_dim else len(input_data), self.input_dims
                )), training=training
        )

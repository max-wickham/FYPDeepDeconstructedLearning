'''Neural Network Interface'''
from abc import ABC, abstractmethod

import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes
import numpy as np


class Network(ABC):
    '''Generic Neural Network Base Class'''

    def __init__(self, input_dims: int,
                 output_dim: int,
                 learning_rate=0.005, train=True, load=False, model_file='') -> None:
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.train = train
        self.load = load
        self.model_file = model_file

    @abstractmethod
    def __call__(self, input_data: np.ndarray, training=False, multi_dim=False) -> ttf.Tensor1[ttf.float32, axes.Height]:
        '''Run the model on a set of input data'''

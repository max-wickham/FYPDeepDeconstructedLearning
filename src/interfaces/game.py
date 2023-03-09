'''Generic Interface for a Game'''
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

Reward = float
Done = bool

class Coordinate(NamedTuple):
    '''Coordinate'''
    x: int
    y: int

class Colour(NamedTuple):
    '''Colour'''
    r : float
    g : float
    b : float


@dataclass
class DrawInformation:
    '''List of rectangles required to draw the game state'''
    Width = int
    Height = int

    rectangles : list[tuple[Height,Width,Coordinate,Colour]]


class Game(ABC):
    '''Game abstract base class'''

    @abstractmethod
    def __init__(self, load : bool = False, load_file_name : str = ''):
        '''Initialise the game'''

    @abstractmethod
    def step(self, actions: list[int], time_step: float = 0.05) -> tuple[Reward, Done]:
        '''Step the game forward one time-step
        and return the reward change whether the game has finished'''

    @abstractmethod
    def get_model_input(self) -> np.ndarray:
        '''Generate a numpy array with the input to the model'''

    @classmethod
    @abstractmethod
    def get_input_shape(cls) -> int:
        '''Returns the shape of the model input matrix'''

    @classmethod
    @abstractmethod
    def get_action_shape(cls) -> int:
        '''Length of the action vector'''

    @abstractmethod
    def get_draw_information(self) -> DrawInformation:
        '''Returns the information required to draw the current game frame'''

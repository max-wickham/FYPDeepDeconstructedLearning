'''Generic Interface for a Game'''
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, field
import math
from typing import NamedTuple

import numpy.typing as npt
import numpy as np

Reward = float
Done = bool

class Vector2D(NamedTuple):
    '''2D Vector'''
    x: int
    y: int

    def __add__(self, right : 'Vector2D'):
        '''Add two vectors'''
        return Vector2D(
            self.x + right.x,
            self.y + right.y
        )

    def __sub__(self, right : 'Vector2D'):
        '''Add two vectors'''
        return Vector2D(
            self.x - right.x,
            self.y - right.y
        )

    def len(self) -> float:
        '''Returns the length of the vector'''
        return np.sqrt(pow(self.x, 2) + pow(self.y, 2))

class Coordinate(Vector2D):
    '''Coordinate'''

class Velocity(Vector2D):
    '''Velocity'''

class PolarVector2D(NamedTuple):
    '''Polar 2D Vector'''
    theta: float
    magnitude: float

    def to_cartesian(self):
        '''Convert to cartesian vector'''
        return Vector2D(np.cos(self.theta) * self.magnitude, np.sin(self.theta) * self.magnitude)

    # @classmethod
    # def from_cartesian(cls, cartesian: Vector2D) -> 'PolarVector2D':
    #     ...

class Colour(NamedTuple):
    '''Colour'''
    r : float
    g : float
    b : float


class Object(NamedTuple):
    '''Generic description of a game object'''
    height: int
    width: int
    center_pos: Coordinate
    rotation: float

def object_overlap(object_1: Object, object_2: Object) -> bool:
    '''Check for overlap between objects'''
    # Convert rotation angles to radians
    angle1 = math.radians(object_1.rotation)
    angle2 = math.radians(object_2.rotation)

    if (object_1.center_pos - object_2.center_pos).len() >  \
        (object_1.width + object_1.height + object_2.width + object_2.height):
        return False

    # Calculate the rotated width and height for each rectangle
    rect1_width_rotated = object_1.width * math.cos(angle1) + object_1.height * math.sin(angle1)
    rect1_height_rotated = object_1.height * math.cos(angle1) + object_1.width * math.sin(angle1)
    rect2_width_rotated = object_2.width * math.cos(angle2) + object_2.height * math.sin(angle2)
    rect2_height_rotated = object_2.height * math.cos(angle2) + object_2.width * math.sin(angle2)

    # Calculate the distance between the centers of the rectangles along the x-axis and y-axis
    x_distance = abs(object_1.center_pos.x - object_2.center_pos.x)
    y_distance = abs(object_1.center_pos.y - object_2.center_pos.y)

    # Calculate the maximum distance between the centers of the rectangles along the x-axis and y-axis
    max_distance_x = (rect1_width_rotated + rect2_width_rotated) / 2
    max_distance_y = (rect1_height_rotated + rect2_height_rotated) / 2

    # Check if the rectangles overlap
    if x_distance <= max_distance_x and y_distance <= max_distance_y:
        return True
    else:
        return False


@dataclass
class DrawInformation:
    '''List of rectangles required to draw the game state'''
    Width = int
    Height = int
    Rotation = float

    rectangles : list[tuple[Width,Height,Coordinate,Colour,str]]
    rotated_rectangles: list[
        tuple[Width,Height,Coordinate,Rotation,Colour,str]] = field(default_factory=list)


class Game(ABC):
    '''Game abstract base class'''

    @abstractmethod
    def __init__(self, load : bool = False, load_file_name : str = ''):
        '''Initialise the game'''

    @abstractmethod
    def step(self, actions: list[float], time_step: float = 0.05) -> tuple[Done, Reward]:
        '''Step the game forward one time-step
        and return the reward change whether the game has finished'''

    @abstractmethod
    def get_model_input(self) -> npt.NDArray:
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

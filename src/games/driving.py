'''Basic Driving Game'''
from dataclasses import dataclass, field
import functools
import math
import random
from typing import NamedTuple
import numpy.typing as npt
import numpy as np
from scipy.ndimage import rotate
# import multiprocessing
# work = multiprocessing.Process(target=show_graph)
# work.start()


from src.interfaces.game import (Done,
Game, Reward, DrawInformation, Coordinate, Velocity, Colour, Object, object_overlap, PolarVector2D)

class Wall(NamedTuple):
    '''Walls witimport multiprocessing
# work = multiprocessing.Process(target=show_graph)
# work.start()
h which the player must not collide or representing the finish line'''
    height: DrawInformation.Height
    width: DrawInformation.Width
    center_pos: Coordinate

def generate_maze(rows, cols) -> tuple[tuple[int,int], tuple[int,int], list[list[bool]]]:
    # rows = 7
    # cols = 12
    maze = [[False for _ in range(cols)] for _ in range(rows)]

    # TODO add min distance enforcement here
    start_row = random.randint(1, rows - 2)
    start_col = random.randint(1, cols - 2)
    goal_row = random.randint(1, rows - 2)
    goal_col = random.randint(1, cols - 2)

    if abs(start_row - goal_row) + abs(start_col - goal_col) < 5:
        return generate_maze(rows, cols)


    maze[start_row][start_col] = True
    maze[goal_row][goal_col] = True

    stack = [(start_row, start_col)]
    visited = set()

    while stack:
        row, col = stack[-1]
        maze[row][col] = True

        neighbors = [(row - 2, col), (row + 2, col), (row, col - 2), (row, col + 2)]
        random.shuffle(neighbors)

        found = False
        for n_row, n_col in neighbors:
            if 0 < n_row < rows - 1 and 0 < n_col < cols - 1 and (n_row, n_col) not in visited:
                maze[(row + n_row) // 2][(col + n_col) // 2] = True
                maze[n_row][n_col] = True
                stack.append((n_row, n_col))
                visited.add((n_row, n_col))
                found = True
                break

        if not found:
            stack.pop()

    return  (start_col, start_row),(goal_col, goal_row), maze



@dataclass
class GameState:
    '''Stores the information relating to the current game'''
    player_pos: Coordinate = Coordinate(500,100)
    # player_velocity: Velocity = Velocity(0,0)
    player_velocity: PolarVector2D = PolarVector2D(0,0)
    walls: list[Wall] = field(default_factory=list)
    finish_line: Wall = Wall(100,50,Coordinate(500,400))


class DrivingGame(Game):
    '''A basic driving game'''
    ACCEL = 1 # pixels per second per second
    ROTATIONAL_SPEED = 2 # radians per second
    MAX_SPEED = 3
    GAME_WIDTH = 1000  # pixels
    GAME_HEIGHT = 600  # pixels
    FRAC = 4
    WINDOW_LENGTH = int(300 / FRAC)
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 15
    WALL_WIDTH = 80
    TIMEOUT = 20

    PLAYER_COLOUR = Colour(20,20,250)
    WALL_COLOUR = Colour(200,200,200)
    FINISH_COLOUR = Colour(0,200,0)

    def __init__(self, load : bool = False, load_file_name : str = ''):
        '''Initialise the game'''
        num_rows = int((self.GAME_HEIGHT - self.WALL_WIDTH / 2) / self.WALL_WIDTH)
        num_columns = int((self.GAME_WIDTH - self.WALL_WIDTH / 2) / self.WALL_WIDTH)
        start, goal, maze = generate_maze(num_rows, num_columns)
        walls = []
        half_wall_width = int(self.WALL_WIDTH / 2)
        start = Coordinate(start[0] * self.WALL_WIDTH + half_wall_width,self.GAME_HEIGHT - start[1] * self.WALL_WIDTH)
        goal = Coordinate(goal[0] * self.WALL_WIDTH + half_wall_width,self.GAME_HEIGHT - goal[1] * self.WALL_WIDTH)
        for y, row in enumerate(maze):
            y = self.GAME_HEIGHT - y * self.WALL_WIDTH
            for x, tile in enumerate(row):
                x = x * self.WALL_WIDTH + half_wall_width
                if not tile:
                    walls.append(Wall(self.WALL_WIDTH,self.WALL_WIDTH,Coordinate(x,y)))


        self.game_state = GameState(walls = walls, player_pos=start, finish_line= Wall(self.WALL_WIDTH,self.WALL_WIDTH,goal))
        self.game_time = 0
        self.grid = self._create_grid()

    def _create_grid(self):
        frac = self.FRAC
        wall_span = int(self.WALL_WIDTH / frac)
        half_wall_span = int(wall_span / 2)
        grid = np.zeros((int(self.GAME_WIDTH / frac), int(self.GAME_HEIGHT / frac)))
        for wall in self.game_state.walls:
            x, y = int(wall.center_pos.x / frac), int(wall.center_pos.y / frac)
            grid[
                x - half_wall_span : x + half_wall_span,
                y - half_wall_span : y + half_wall_span] = 1
        grid = np.pad(grid,self.WINDOW_LENGTH,
            'constant', constant_values = 1)
        return grid



    def _update_player_pos(self, actions: list[float], time_step: float):
        forward_accel = (1 if actions[1] > 0 else -2) * self.ACCEL * time_step
        rotational_vel = (actions[0] - 0.5) * 2 * self.ROTATIONAL_SPEED * time_step
        theta = self.game_state.player_velocity.theta
        speed = self.game_state.player_velocity.magnitude
        theta += rotational_vel
        new_speed = forward_accel + speed
        new_speed = np.clip(new_speed, 0, self.MAX_SPEED)
        self.game_state.player_velocity = PolarVector2D(theta, new_speed)
        self.game_state.player_pos = Coordinate(
            *(self.game_state.player_pos + self.game_state.player_velocity.to_cartesian()))
        self.game_state.player_pos = Coordinate(
            np.clip(self.game_state.player_pos.x, 0, self.GAME_WIDTH),
            np.clip(self.game_state.player_pos.y, 0, self.GAME_HEIGHT),
        )

    def _check_collision_with_walls(self) -> bool:
        '''Checks if any of the walls are touching the player'''
        player_rect = Object(self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.game_state.player_pos,
            self.game_state.player_velocity.theta)
        for wall in self.game_state.walls:
            wall_rect = Object(*wall, 0)
            if object_overlap(wall_rect, player_rect):
                return True
        return False

    def _distance_to_finish(self) -> float:
        return (self.game_state.player_pos - self.game_state.finish_line.center_pos).len()

    def _check_complete(self) -> bool:
        player_rect = Object(self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.game_state.player_pos,
            self.game_state.player_velocity.theta)
        wall_rect = Object(*self.game_state.finish_line, 0)
        return object_overlap(player_rect, wall_rect)

    def step(self, actions: list[float], time_step: float = 0.05) -> tuple[Done, Reward]:
        '''Step the game forward one time-step
        and return the reward change whether the game has finished
        Action are [Steer, Accel]
        Accel = 1 gives forward, 0 gives break
        Steer = 1 gives left, -1 gives right
        '''
        self.game_time += time_step
        self._update_player_pos(actions, time_step)
        collision = self._check_collision_with_walls()
        if collision or self.game_time > self.TIMEOUT:
            return True, -1 * (self._distance_to_finish() / 100)
        done = self._check_complete()
        if done:
            print('WINNNNN')
        return (done, 0 if not done else 40 - self.game_time)

    def get_model_input(self, flatten = True) -> npt.NDArray:
        '''Generate a numpy array with the input to the model'''
        x,y = self.game_state.player_pos.x // self.FRAC, self.game_state.player_pos.y // self.FRAC
        x += self.WINDOW_LENGTH
        y += self.WINDOW_LENGTH
        sub_grid = self.grid[
            int(x - self.WINDOW_LENGTH / 2) : int(x + self.WINDOW_LENGTH / 2),
            int(y - self.WINDOW_LENGTH / 2) : int(y + self.WINDOW_LENGTH / 2),
        ]
        matrix = np.zeros((int(self.WINDOW_LENGTH * 1.5), int(self.WINDOW_LENGTH * 1.5)))
        # pad_size = ((matrix.shape[0] - sub_grid.shape[0]) // 2,
        #     (matrix.shape[1] - sub_grid.shape[1]) // 2)
        # sub_grid = np.pad(sub_grid,
        #                     ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])),
        #                     mode='constant', constant_values=0.5)
        sub_grid = rotate(sub_grid, self.game_state.player_velocity.theta * 180/np.pi, reshape=False, cval=0.5)
        pad_size = ((matrix.shape[0] - sub_grid.shape[0]) // 2,
            (matrix.shape[1] - sub_grid.shape[1]) // 2)
        sub_grid_padded = np.pad(sub_grid,
                            ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])),
                            mode='constant', constant_values=0)
        if sub_grid_padded.shape != matrix.shape:
            sub_grid_padded = np.pad(
                sub_grid_padded, ((0,1), (0,1)), mode='constant', constant_values=0)
        if not flatten:
            return sub_grid
        sub_grid = sub_grid.flatten()
        player_state = np.array(
            (*self.game_state.player_pos, *self.game_state.player_velocity, self.game_time))
        goal_pos = np.array(
            (*(self.game_state.player_pos - self.game_state.finish_line.center_pos),))
        return np.concatenate((player_state, goal_pos, sub_grid))

    @classmethod
    @functools.cache
    def get_input_shape(cls) -> int:
        '''Returns the shape of the model input matrix'''
        game = cls()
        return len(game.get_model_input())

    @classmethod
    @functools.cache
    def get_action_shape(cls) -> int:
        '''Length of the action vector'''
        return 2

    def get_draw_information(self) -> DrawInformation:
        '''Returns the information required to draw the current game frame'''
        rectangles = []
        rectangles += [
            (wall.width, wall.height, wall.center_pos, self.WALL_COLOUR, 'wall')
            for wall in self.game_state.walls
        ]
        finish = self.game_state.finish_line
        rectangles += [
            (finish.width, finish.height, finish.center_pos, self.FINISH_COLOUR, 'finish')
        ]
        # rectangles += [
        #     (10, 10, self.game_state.player_pos, self.PLAYER_COLOUR, 'player')
        # ]
        theta = self.game_state.player_velocity.theta
        polar_vector = PolarVector2D(theta, self.PLAYER_WIDTH / 2 - 3)
        head_pos = polar_vector.to_cartesian() + self.game_state.player_pos

        # rectangles += [
        #     (self.WINDOW_LENGTH * self.FRAC, self.WINDOW_LENGTH * self.FRAC, self.game_state.player_pos, (100,100,100), 'frame')
        # ]
        return DrawInformation(rectangles, [
            (self.PLAYER_WIDTH, self.PLAYER_HEIGHT, self.game_state.player_pos,
            theta, self.PLAYER_COLOUR, 'player'),
            (6,self.PLAYER_HEIGHT, Coordinate(*head_pos), theta, Colour(250,0,0), 'player head'),
            (self.WINDOW_LENGTH * self.FRAC, self.WINDOW_LENGTH * self.FRAC, self.game_state.player_pos, theta, Colour(100,100,100), 'frame')
        ])

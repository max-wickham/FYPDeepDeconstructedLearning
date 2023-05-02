'''Space Invaders Game Implementation'''
from dataclasses import dataclass, field
from math import copysign
import pickle
import random
import numpy as np

from src.interfaces.game import Colour, Coordinate, Done, DrawInformation, Game, Reward

Health = float
Direction = int
POS = -1
NEG = 1
STATIONARY = 0


POS_REWARD = 0.2
NEG_REWARD = -70


@dataclass
class Enemy:
    '''Enemy'''
    pos: Coordinate
    health: Health
    direction: Direction


@dataclass
class Shield:
    '''Shield'''
    pos: int
    health: Health


@dataclass
class GameState:
    '''DrawState'''
    player_bullets: list[Coordinate] = field(default_factory=list)
    enemy_bullets: list[Coordinate] = field(default_factory=list)
    enemies: list[Enemy] = field(default_factory=list)
    shields: list[Shield] = field(default_factory=list)
    player_pos: int = 100
    player_health: int = 10
    player_speed: int = 0

    # CONSTANTS
    BULLET_SIZE = 10
    PLAYER_WIDTH = 50
    PLAYER_HEIGHT = 20

    ENEMY_WIDTH = 50
    ENEMY_HEIGHT = 20

    SHIELD_WIDTH = 80
    SHIELD_HEIGHT = 40
    SHIELD_HEIGHT_POS = 30

    def to_draw_information(self) -> DrawInformation:
        rectangles = (
            [
                (self.PLAYER_WIDTH, self.PLAYER_HEIGHT, Coordinate(self.player_pos, int(self.PLAYER_HEIGHT / 2) ), Colour(0,255,255), 'player')
            ]+
            [
                (self.BULLET_SIZE, self.BULLET_SIZE, bullet, Colour(255,255,255), 'bullet') for bullet in self.player_bullets
            ] +
            [
                (self.BULLET_SIZE, self.BULLET_SIZE, bullet, Colour(255,0,0), 'bullet') for bullet in self.enemy_bullets
            ] +
            [
                (self.ENEMY_WIDTH, self.ENEMY_HEIGHT, enemy.pos, Colour(0,255,0), 'enemy') for enemy in self.enemies
            ] +
            [
                (self.SHIELD_WIDTH, self.SHIELD_HEIGHT,
                    Coordinate(shield.pos, self.SHIELD_HEIGHT_POS), Colour(100,100,0), 'shield') for shield in self.shields
            ]
        )
        return DrawInformation(rectangles)


class SpaceInvadersLarge(Game):
    '''Space Invaders Game'''

    BULLET_DAMAGE = 10
    ENEMY_SPEED: float = 200  # pixels per second (left,right)
    ENEMY_SHOOT_MULTIPLIER = 4  # shots per second
    ENEMY_DOWN_SPEED: float = 15  # pixels per second (down)
    BULLET_SPEED: float = 400  # pixels per second
    PLAYER_ACC: float = 20  # pixels per second^2
    MAX_PLAYER_SPEED = 300  # pixels per second
    ENEMY_LOSE_HEIGHT: int = 80  # pixels
    SHOOT_COOL_DOWN = 0.5  # seconds
    ENEMY_SPAWN_TIME = 5  # seconds
    GAME_WIDTH = 1000  # pixels
    GAME_HEIGHT = 600  # pixels

    _game_state: GameState

    _player_speed: int = 0
    _enemy_direction: Direction = POS
    _enemy_boundary_buffer: int = 0
    _bullet_ground_buffer: int = 0

    _store: bool = False
    _store_file_name: str = ''

    def __init__(self, store: bool = False, store_file_name: str = ''):
        self._store = store
        self._store_file_name = store_file_name

        self._game_state = GameState()

        self._game_time = 0
        self._frame_time = 0.02
        self._last_shoot_time = self._game_time
        self._last_enemy_spawn = self._game_time - 10
        self._enemy_boundary_buffer = int(self._game_state.ENEMY_WIDTH / 2)
        self._bullet_ground_buffer = self._game_state.BULLET_SIZE
        self._lost = False
        self._log = []

        self.removed_enemy_bullets = set()
        self.removed_player_bullets = set()
        self.removed_shields = set()
        self.removed_enemies = set()

        self.prev_model_input = None

    def _process_enemies(self, removed_enemies):
        # Enemies
        for index, enemy in enumerate(self._game_state.enemies):
            if enemy.health <= 0:
                # print('ENEMY DESTROYED')
                removed_enemies.add(index)
                continue
            # Check direction
            if enemy.pos.y < self.ENEMY_LOSE_HEIGHT:
                # print('BAD ENEMY HEIGHT')
                self._lost = True
                return

            # Fire bullet
            distance_to_player = abs(
                enemy.pos.x - self._game_state.player_pos)
            threshold = self.GAME_WIDTH - pow(
                (self.GAME_WIDTH - distance_to_player) / self.GAME_WIDTH, 1.2
            ) * (self.GAME_WIDTH)
            rand = random.randint(0, self.GAME_WIDTH)
            if rand > threshold and random.randint(0, 1000) < (1000*self._frame_time*self.ENEMY_SHOOT_MULTIPLIER):
                self._game_state.enemy_bullets.append(
                    Coordinate(enemy.pos.x, enemy.pos.y))

            # Update Position
            y = enemy.pos.y - self.ENEMY_DOWN_SPEED * self._frame_time
            x = enemy.pos.x - self.ENEMY_SPEED * self._frame_time * enemy.direction

            if x > self.GAME_WIDTH:
                x = 0
            if x < 0:
                x = self.GAME_WIDTH

            enemy.pos = Coordinate(x, y)

        if self._game_time - self._last_enemy_spawn > self.ENEMY_SPAWN_TIME:
            self._last_enemy_spawn = self._game_time
            # self.last_enemy_spawn = 1000
            new_x_pos = random.randint(
                self._game_state.ENEMY_WIDTH, self.GAME_WIDTH - self._game_state.ENEMY_WIDTH)
            new_y_pos = self.GAME_HEIGHT - self._game_state.ENEMY_HEIGHT
            self._game_state.enemies.append(
                Enemy(
                    pos=Coordinate(new_x_pos, new_y_pos),
                    health=50,
                    direction=POS if random.random() > 0.5 else NEG
                )
            )

    def _process_enemy_bullets(self, removed_enemy_bullets):
        # Enemy Bullets
        for index, bullet in enumerate(self._game_state.enemy_bullets):
            x, y = bullet
            # Check if bullet is at the ground or at player
            if y < self._bullet_ground_buffer:
                removed_enemy_bullets.add(index)
                distance_to_player = abs(x - self._game_state.player_pos)
                distance_to_player = min(
                    (distance_to_player, abs(self.GAME_WIDTH - distance_to_player)))
                if distance_to_player < (self._game_state.PLAYER_WIDTH / 2):
                    self._game_state.player_health -= self.BULLET_DAMAGE

            self._game_state.enemy_bullets[index] = (
                x, y - self.BULLET_SPEED * self._frame_time)

    def _process_player_bullets(self, removed_player_bullets, removed_enemies):
        for index, bullet in enumerate(self._game_state.player_bullets):
            x, y = bullet
            if y > self.GAME_HEIGHT - self._bullet_ground_buffer:
                removed_player_bullets.add(index)
                continue

            for e_index, enemy in enumerate(self._game_state.enemies):
                x_distance = abs(x - enemy.pos.x)
                y_distance = abs(y - enemy.pos.y)
                x_distance = min(
                    [x_distance, abs(self.GAME_WIDTH - x_distance)])
                if x_distance <= self._game_state.ENEMY_WIDTH / 2 and y_distance <= self._game_state.ENEMY_HEIGHT/2:
                    enemy.health -= self.BULLET_DAMAGE
                    self.removed_player_bullets.add(index)
                    if enemy.health <= 0:
                        removed_enemies.add(e_index)
                        # print('SHIELD DESTROYED')

            self._game_state.player_bullets[index] = (
                x, y + self.BULLET_SPEED * self._frame_time)

    def _process_shields(self, removed_shields, removed_player_bullets, removed_enemy_bullets):

        def _process_bullet(x, y):
            for s_index, shield in enumerate(self._game_state.shields):
                x_distance = abs(x - shield.pos)
                y_distance = abs(y - self._game_state.SHIELD_HEIGHT_POS)
                if x_distance <= self._game_state.SHIELD_WIDTH / 2 and y_distance <= self._game_state.SHIELD_HEIGHT/2:
                    shield.health -= self.BULLET_DAMAGE
                    if shield.health <= 0:
                        removed_shields.add(s_index)
                    return True
            return False

        for index, bullet in enumerate(self._game_state.player_bullets):
            x, y = bullet
            if _process_bullet(x, y):
                removed_player_bullets.add(index)

        for index, bullet in enumerate(self._game_state.enemy_bullets):
            x, y = bullet
            if _process_bullet(x, y):
                removed_enemy_bullets.add(index)

    def _process_player_movement(self, direction: int):
        # Movement Control
        acc = self.PLAYER_ACC * direction
        self._game_state.player_speed += acc
        speed = self._game_state.player_speed

        if abs(speed) > self.MAX_PLAYER_SPEED:
            speed = copysign(self.MAX_PLAYER_SPEED, speed)
            self._game_state.player_speed = speed

        self._game_state.player_pos += speed * self._frame_time

        if self._game_state.player_pos > self.GAME_WIDTH:
            self._game_state.player_pos = 0

        if self._game_state.player_pos < 0:
            self._game_state.player_pos = self.GAME_WIDTH

    def _process_player_shots(self, shoot: bool):
        if shoot:
            if self._game_time - self._last_shoot_time < self.SHOOT_COOL_DOWN:
                shoot = False
        if shoot:
            self._last_shoot_time = self._game_time
            self._game_state.player_bullets.append(
                (self._game_state.player_pos, self._game_state.PLAYER_HEIGHT / 2))

    def step(self, actions: list[float], time_step: float = 0.05) -> tuple[Done, Reward]:
        self._frame_time = time_step
        self._game_time += time_step

        if abs(actions[0]-0.5) < 0.1:
            direction = 0
        else:
            direction = POS if actions[0] > 0. else NEG
        shoot = actions[1] > 0.5

        self._lost = False
        # if self.load:
        #     return self._step_saved_game()

        # if self.record:
        #     self._log.append(pickle.dumps(self._game_state))

        self._game_state.enemies = [enemy for index, enemy in
                                    enumerate(self._game_state.enemies) if index not in self.removed_enemies]
        self._game_state.enemy_bullets = [bullet for index, bullet in
                                          enumerate(self._game_state.enemy_bullets) if index not in self.removed_enemy_bullets]
        self._game_state.player_bullets = [bullet for index, bullet in
                                           enumerate(self._game_state.player_bullets) if index not in self.removed_player_bullets]
        self._game_state.shields = [shield for index, shield in
                                    enumerate(self._game_state.shields) if index not in self.removed_shields]

        self.removed_enemy_bullets = set()
        self.removed_player_bullets = set()
        self.removed_shields = set()
        self.removed_enemies = set()

        self._process_enemies(self.removed_enemies)
        self._process_enemy_bullets(self.removed_enemy_bullets)
        self._process_player_bullets(
            self.removed_player_bullets, self.removed_enemies)
        self._process_shields(
            self.removed_shields, self.removed_player_bullets, self.removed_enemy_bullets)
        self._process_player_movement(direction)
        self._process_player_shots(shoot)

        if self._game_state.player_health <= 0:
            self._lost = True

        # if self._lost:
        #     if self.record:
        #         with open("game4.pickle", "wb") as file:
        #             file.write(pickle.dumps([pickle.loads(l) for l in self._log]))
        return self._lost, NEG_REWARD if self._lost else POS_REWARD

    def get_model_input(self, flatten = True) -> np.array:
        '''Generate input for AI'''
        SCALAR = 0.1
        player_index = int(self._game_state.player_pos * SCALAR)
        # print(player_index)
        shift = int(self.GAME_WIDTH * SCALAR)
        # print(shift)
        def mod(n, mult = 1):
            return int((n*SCALAR - player_index + shift + int(shift / 2) -1)*mult) % int(shift*mult)

        def scale(n, mult = 1):
            return int(n*SCALAR*mult)

        # player_bullets = numpy.zeros((scale(self.GAME_WIDTH) + 1, scale(self.GAME_HEIGHT) + 3))
        # for bullet in self.draw_state.player_bullets:
        #     x,y = bullet
        #     if y > self.GAME_HEIGHT:
        #         y = self.GAME_HEIGHT
        #     player_bullets[scale(x)][scale(y)] += 1
        enemy_bullets = np.zeros((scale(self.GAME_WIDTH) + 2, scale(self.GAME_HEIGHT) + 2))
        for bullet in self._game_state.enemy_bullets:
            x,y = bullet
            if y > self.GAME_HEIGHT:
                y = self.GAME_HEIGHT
            if y < 0:
                y = 0
            enemy_bullets[mod(x)][scale(y)] += 1

        enemies = np.zeros((scale(self.GAME_WIDTH,0.5) + 1, scale(self.GAME_WIDTH,0.5) + 3))
        for enemy in self._game_state.enemies:
            enemies[mod(enemy.pos.x,0.5)][scale(enemy.pos.y,0.5)] += 1


        shields = np.zeros(scale(self.GAME_WIDTH,0.5) + 1)
        for shield in self._game_state.shields:
            shields[mod(shield.pos,0.5)] = shield.health
        if flatten:
            vector = np.concatenate((
                enemy_bullets.flatten(),
                enemies.flatten(),
                shields.flatten()
            ))
        else:
            vector = (enemy_bullets[10:-10,0:30],enemies)
        # print(player_index)
        # print(numpy.transpose(enemy_bullets)[0:10,0:10])
        # print(vector)
        # print(enemy_bullets[9:17,0:5].shape)
        if self.prev_model_input is None:
            self.prev_model_input = np.zeros(len(vector))
        response = np.concatenate((self.prev_model_input, vector))
        self.prev_model_input = vector
        return response

    @classmethod
    def get_input_shape(cls) -> int:
        '''Length of the observation vector'''
        game = cls()
        return len(game.get_model_input())

    @classmethod
    def get_action_shape(cls) -> int:
        '''Length of the action vector'''
        return 2

    def get_draw_information(self) -> DrawInformation:
        '''Returns the information required to draw the current game frame'''
        return self._game_state.to_draw_information()

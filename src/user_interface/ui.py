'''Implementation of a UI for a game'''
import time
import pickle

import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from src.interfaces.game import DrawInformation, Game

class GameUI:
    '''Implements the game UI using Pygame'''

    def __init__(self, game: type[Game], model=None, record=False, save_location='game.pickle') -> None:

        pygame.init()
        self.screen = pygame.display.set_mode([1500, 600])
        self.screen.fill((0, 0, 0))
        self.game = game()
        self.direction = 0
        self.shoot = 0
        self.model = model
        self.player_img = pygame.image.load('./src/assets/space/player.png')
        self.player_img.convert()
        self.critic = 0
        self.frame_states = []
        self.record = record
        self.save_location = save_location

    def run(self):
        last_frame_time = time.time()
        done = False
        total_reward = 0
        while not done:
            self.screen.fill((0, 0, 0))
            draw_state = self.game.get_draw_information()
            self.draw_game(draw_state)
            # for event in pygame.event.get():
            actions = self.get_actions()
            # print(actions)
            time_since_last_frame = time.time() - last_frame_time
            done, reward = self.game.step(
                actions, time_since_last_frame)
            total_reward += reward
            # print(time.time() - last_frame_time)
            last_frame_time = time.time()
            time.sleep(max(0,0.02 - time_since_last_frame))
        pickle.dump(self.frame_states, open(self.save_location, "wb"))
        print('Score', total_reward)

    def playback(self):
        '''Playback a recorded game'''
        recorded_game : list[DrawInformation] = pickle.load(open(self.save_location, "rb"))
        last_frame_time = time.time()
        for frame in recorded_game:
            time_since_last_frame = time.time() - last_frame_time
            last_frame_time = time.time()
            self.screen.fill((0, 0, 0))
            self.draw_game(frame)
            # print(max(0,0.02 - time_since_last_frame))
            # time.sleep(max(0,0.02 - time_since_last_frame))
            time.sleep(0.02)

    def draw_game(self, draw_information: DrawInformation):
        '''Draw the game state in rectangles'''
        self.frame_states.append(draw_information)
        for rectangle in draw_information.rectangles:
            left = int(rectangle[2][0] - rectangle[0] / 2)
            top = 600 - int(rectangle[2][1] + rectangle[1] / 2)
            # if rectangle[4] == 'player':
            #     self.screen.blit(self.player_img, (int(rectangle[2][0]  - rectangle[0]), 600 - int(rectangle[2][1] + rectangle[1] + 10)))
            # else:
            pygame.draw.rect(self.screen, rectangle[3], pygame.Rect(
                left, top, rectangle[0], rectangle[1]))
        pygame.draw.rect(self.screen, (255, 255, 0),
                         pygame.Rect(700, 300, 50, -1*self.critic * 2))
        pygame.display.flip()

    def get_actions(self) -> list[float]:
        '''Get the actions for the current frame using the model'''
        if self.model is not None:
            actions = self.model.compute_action(self.game)
            return actions
            # print(actions[1])
            # self.critic = actions[1]
            # return actions[0]
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.direction = 0.5
                if event.key == pygame.K_SPACE:
                    self.shoot = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = 1
                if event.key == pygame.K_RIGHT:
                    self.direction = 0
                if event.key == pygame.K_SPACE:
                    self.shoot = 1
        return [self.direction, self.shoot]

'''Implementation of a UI for a game'''
import time

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
import time


class GameUI:
    '''Implements the game UI using Pygame'''

    def __init__(self, game : type[Game], model = None) -> None:

        pygame.init()
        self.screen = pygame.display.set_mode([1500, 600])
        self.screen.fill((0, 0, 0))
        self.game = game()
        self.direction = 0
        self.shoot = 0
        self.model = model
        self.player_img = pygame.image.load('./src/assets/space/player.png')
        self.player_img.convert()

    def run(self):
        last_frame_time = time.time()
        done = False
        total_reward = 0
        while not done:
            time.sleep(0.02)
            self.screen.fill((0, 0, 0))
            draw_state = self.game.get_draw_information()
            self.draw_game(draw_state)
            # for event in pygame.event.get():
            done,reward = self.game.step(self.get_actions(), time.time() - last_frame_time)
            total_reward += reward
            # print(time.time() - last_frame_time)
            last_frame_time = time.time()
        print('Score', total_reward)


    def draw_game(self, draw_information : DrawInformation):
        for rectangle in draw_information.rectangles:
            left = int(rectangle[2][0] - rectangle[0] / 2)
            top = 600 - int(rectangle[2][1] + rectangle[1] / 2)
            # if rectangle[4] == 'player':
            #     self.screen.blit(self.player_img, (int(rectangle[2][0]  - rectangle[0]), 600 - int(rectangle[2][1] + rectangle[1] + 10)))
            # else:
            pygame.draw.rect(self.screen, rectangle[3], pygame.Rect(left, top, rectangle[0], rectangle[1]))
a
        pygame.display.flip()


    def get_actions(self) -> list[int]:
        if self.model is not None:
            return self.model.compute_action(self.game)
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

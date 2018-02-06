from .abstract_player import Snake, COLORS
from .player import Player
import random
import math


class BotPlayer(Player):

    def __init__(self, id):
        super(BotPlayer, self).__init__(id)
        self.is_human = False

    def process(self, message, game_state):
        '''
        Bots dont pass message from the interface
        They process the last state of the grid
        '''

        raise NotImplemented

    def update(self, grid):
        if not self.alive:
            return
        if self.command == 'left':
            self.snake.turn_left()
        if self.command == 'right':
            self.snake.turn_right()
        self.snake.move(grid.width, grid.height)
        if self.snake.collision(grid):
            self.alive = False
        self.snake.update_grid(grid)

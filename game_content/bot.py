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

        projection = (self.snake.x, self.snake.y) +\
                     5 * self.snake.speed * (math.cos(self.snake.direction),
                                             math.sin(self.snake.direction))

        pass

    def update(self, grid):
        self.command = random.choice(['left', 'right', 'straight'])
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

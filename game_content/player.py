import json
from .abstract_player import AbstractPlayer, Snake, COLORS


class Player(AbstractPlayer):

    def __init__(self, id):
        self.id = id
        self.is_human = True
        self.score = 0
        self.name = 'guest'
        self.command = None
        self.color = COLORS[id - 1]

    def spawn(self, x, y):
        self.alive = True
        self.snake = Snake(self.id, x, y)

    def process(self, message, game_state):
        message = json.loads(message)
        if 'name' in message:
            self.name = message['name']
        if 'command' in message:
            if message['command'] in ('left', 'right', 'straight'):
                self.command = message['command']

    def update(self, grid):
        if not self.alive:
            return
        if self.command is not None:
            print("Player {} did {}".format(self.id, self.command))
        if self.command == 'left':
            self.snake.turn_left()
        if self.command == 'right':
            self.snake.turn_right()
        self.snake.move(grid.width, grid.height)
        if self.snake.collision(grid):
            self.alive = False
        self.snake.update_grid(grid)

    def get_snake(self):
        return {'id': self.id, 'x': self.snake.x, 'y': self.snake.y, 'color': self.color}


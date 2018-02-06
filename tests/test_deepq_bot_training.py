from unittest import TestCase
import math
import numpy as np
from game_content.deepq_bot import AbstractDeepQGameAdapter
from game_content.zatacka import Grid

GRID_SIZE = 5

class DummyGameDeepQ(AbstractDeepQGameAdapter):

    def __init__(self):
        AbstractDeepQGameAdapter.__init__(self, game_size=(GRID_SIZE, GRID_SIZE),
                                          num_layers=2, action_size=5)
        self.id = 1
        self.grid = Grid(GRID_SIZE, GRID_SIZE)
        self.grid.set(int(GRID_SIZE / 2), int(GRID_SIZE / 2), 1)

        # Commands 
        self.command_to_action = {0: None, 1: "top", 2: "left", 3: "right", 4: 'bottom'}

        # Internals
        self.command = None
        self.reward = 0
        self.alive = 1
        self.time_step = 0
        self.score = 0

    def process(self):
        self.process_game_state(self.grid, self.time_step, self.reward, self.alive)

    def update(self):
        # Move the player
        self.navigate()

        # Set the reward
        list_y, list_x = np.where(np.array(self.grid.grid) == 1)
        y, x = list_y[0], list_x[0]
        self.reward = math.exp(-(x + y) / 10.)
        self.score += self.reward

        if self.time_step > self.buffer_size + self.time_frame_size:
            self.network_backward(self.time_step)

        self.time_step += 1

    def index_to_command(self, action):
        return self.command_to_action[action]

    def navigate(self):
        list_y, list_x = np.where(np.array(self.grid.grid) == 1)
        y, x = list_y[0], list_x[0]
        self.grid = Grid(GRID_SIZE, GRID_SIZE)
        if self.command == "top":
            self.grid.set(x, y + 1, 1)
        elif self.command == "bottom":
            self.grid.set(x, y - 1, 1)
        elif self.command == "left":
            self.grid.set(x + 1, y, 1)
        elif self.command == "right":
            self.grid.set(x - 1, y, 1)
        else:
            self.grid.set(x, y, 1)


class TestDeepQTrainingOnDummyGame(TestCase):

    def setUp(self):
        self.bot = DummyGameDeepQ()

    def test_one_step(self):

        self.assertEqual(self.bot.command, None, "Bot was not correctly initialized")
        self.bot.process()
        self.bot.update()
        self.assertIsNotNone(self.bot.command, "Didn't create an action")

    def test_strategy_learning(self):

        n_iter = 20000
        for i in range(n_iter):
            self.bot.process()
            self.bot.update()
        assert False



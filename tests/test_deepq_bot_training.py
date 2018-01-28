from unittest import TestCase
import numpy as np
from game_content.deepq_bot import AbstractDeepQGameAdapter
from game_content.zatacka import Grid

class DummyGameDeepQ(AbstractDeepQGameAdapter):

    def __init__(self):
        AbstractDeepQGameAdapter.__init__(game_size=(7, 7), num_layers=2, action_size=5)
        self.grid = Grid(7, 7)
        self.grid.set(2, 2, 1)

        # Internals
        self.action = None
        self.reward = 0
        self.time_step = 0
        self.score = 0

    def process(self):
        self.process_game_state(self.grid, self.time_step, self.reward)

    def update(self):
        # Move the player
        self.navigate()

        # Set the reward
        if self.grid.get(0, 0) == 1:
            self.reward = 1
        else:
            self.reward = 0
        self.score += self.reward

        self.time_step += 1

    def index_to_command(self, action):
        command_to_action = {0: None, 1: "top", 2: "left", 3: "right", 4: 'bottom'}
        return command_to_action[action]

    def navigate(self):
        list_y, list_x = np.where(np.array(self.grid.grid))
        y, x = list_y[0], list_x[0]
        self.grid.set(x, y, 0)

        if self.action == "top":
            self.grid.set(x, y + 1, 1)
        elif self.action == "bottom":
            self.grid.set(x, y - 1, 1)
        elif self.action == "left":
            self.grid.set(x + 1, y, 1)
        else:
            self.grid.set(x - 1, y, 1)


class TestDeepQTrainingOnDummyGame(TestCase):

    def setUp(self):
        self.bot = DummyGameDeepQ()

    def test_one_step(self):

        self.assertEqual(self.bot.action, None, "Bot was not correctly initialized")
        self.bot.process()
        self.bot.update()
        self.assertIsNotNone(self.bot.action, "Didn't create an action")

    def test_strategy_learning(self):

        n_iter = 10000
        for i in range(n_iter):

            self.bot.process()
            self.bot.update()

        print(float(self.bot.score) / n_iter)



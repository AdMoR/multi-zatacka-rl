import copy
import time
import gevent
import random
from . import BotPlayer, DeepQBotPlayer
from .zatacka import Grid, Zatacka
import cv2


class ZatackaPlayground(Zatacka):

    def register_bots(self, nb_bots):
        # player ids must start at 1 because 0 on the grid means nothing
        ids = [player.id for player in self.players]
        free_ids = [i for i in range(1, 7) if i not in ids]

        for k, id_ in enumerate(free_ids):
            if k == nb_bots:
                break
            bot_player = DeepQBotPlayer(id_, game_size=(self.width, self.height))
            print('created bot %d' % bot_player.id)
            self.players.append(bot_player)

    def spawn_bots(self, nb_bots=10):
        self.register_bots(nb_bots)
        self.spawn_players()

    def display_debug_frame(self, player, grid):
        player_grid_vision = player.replay_adapter._transform_grid_to_nd_mat(grid,
                                                                             player,
                                                                             debug=False).asnumpy()
        player_grid_vision += 1.
        player_grid_vision *= 125.
        cv2.imwrite("Last_grid_from_player_{}.jpg".format(player.id), player_grid_vision)

    def generate_context(self):
        print('creating new game')
        self.grid = Grid(self.width, self.height)
        self.game_history = []
        self.start_time = time.time()
        self.frame_time = 0.0133
        self.frame = -1

        # Communication with the front end
        for client in self.clients:
            self.send_size(client)
        for client in self.clients:
            self.send(client, {'type': 'restart'})
        self.broadcast_players()

        # Create the players for the game
        self.spawn_bots()
        print("Game created and bot spawned !")

    def run_bot_training(self, context=False, nb_loops=10000):
        """
        Master loop of the game :
        Handles the games and runs the step
        """
        if not context:
            self.generate_context()
        alive_players = copy.copy(self.players)

        # Main game loop
        loop_iter = 0
        while len(alive_players) > 1 and loop_iter < nb_loops:

            if loop_iter % 100 == 0:
                print("Loop {} : players alive {}".format(loop_iter,
                                                          [(player.id, player.score)
                                                           for player in alive_players]))
            self.run_action_step(alive_players)
            self.run_display(alive_players)
            loop_iter += 1

    def start(self):
        gevent.spawn(self.run_bot_training)

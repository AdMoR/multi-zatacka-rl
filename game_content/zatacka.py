import copy
import time
import json
import gevent
import random
from . import Player, BotPlayer, COLORS


class Grid(object):

    def __init__(self, width, height, wrap=True):
        self.width = width
        self.height = height
        self.wrap = wrap
        self.grid = self.empty_grid(width, height)

    def empty_grid(self, w, h):
        return [[0] * h for _ in xrange(w)]

    def get(self, x, y):
        if self.wrap:
            if not 0 <= x < self.width:
                x = x % self.width
            if not 0 <= y < self.height:
                y = y % self.height

        return self.grid[x][y]

    def set(self, x, y, v):
        if self.wrap:
            if not 0 <= x < self.width:
                x = x % self.width
            if not 0 <= y < self.height:
                y = y % self.height

        self.grid[x][y] = v


def too_close(x, y, xx, yy, min_dist):
    dx = x - xx
    dy = y - yy
    return dx * dx + dy * dy < min_dist * min_dist


class Zatacka(object):

    def __init__(self, width=200, height=200):
        self.clients = list()
        self.players = list()
        self.game_history = []
        self.width = width
        self.height = height

    def register_observer(self, socket):
        self.clients.append(socket)
        self.send_size(socket)
        self.broadcast_players()
        for step in self.game_history:
            self.send_frame(socket, step)

    def broadcast_players(self):
        for client in self.clients:
            self.send_scores(client)

    def send_frame(self, client, data):
        self.send(client, {'type': 'step', 'content': data})

    def send_scores(self, client):
        players = [{
            'id': player.id,
            'name': player.name,
            'score': player.score,
            'color': player.color,
            } for player in self.players]
        self.send(client, {'type': 'players', 'content': players})

    def send_size(self, client):
        self.send(client, {'type': 'size', 'width': self.width, 'height': self.height})

    def register_player(self):
        if len(self.players) == 0:
            ids = [player.id for player in self.players]
            # player ids must start at 1 because 0 on the grid means nothing
            free_ids = [i for i in range(1, 7) if i not in ids]
            bot_player = BotPlayer(free_ids[0])
            print('created bot %d' % player.id)
            self.players.append(bot_player)
        if len(self.players) < 6:
            ids = [player.id for player in self.players]
            # player ids must start at 1 because 0 on the grid means nothing
            free_ids = [i for i in range(1, 7) if i not in ids]
            player = Player(free_ids[0])
            print('created player %d' % player.id)
            self.players.append(player)
            return player
        else:
            return None

    def remove_player(self, player):
        self.players.remove(player)

    def send(self, client, data):
        try:
            data = json.dumps(data)
            client.send(data)
        except Exception:
            self.clients.remove(client)

    def spawn_players(self):
        positions = []
        for player in self.players:
            need_spot = True
            while need_spot:
                x = random.random() * self.width
                y = random.random() * self.height
                need_spot = False
                for (xx, yy) in positions:
                    if too_close(x, y, xx, yy, 20):
                        need_spot = True
            player.spawn(x, y)

    def run_action_step(self, alive_players):
        """
        Step of the game
        Each step correspond to one change in the game state
        The Bot has to sample from this events
        """
        self.frame += 1

        for player in alive_players:
            if not player.is_human:
                player.process(self.frame, self.grid)

        for player in alive_players:
            player.update(self.grid)

    def run_display(self, alive_players):
        """
        Sharable step of the game
        Display only from the previous processing
        """

        data = list()
        for player in alive_players:
            data.append(player.get_snake())

        self.game_history.append(data)

        # remove dead players only after broadcasting their last state
        someone_died = False
        for player in alive_players:
            if not player.alive:
                someone_died = True
                alive_players.remove(player)

        if someone_died:
            print('someone died')
            for player in alive_players:
                player.score += 1

        for client in self.clients:
            self.send_frame(client, data)
            if someone_died:
                self.send_scores(client)

        next_frame_time = self.start_time + self.frame * self.frame_time
        sleep_time = max(0, next_frame_time - time.time())
        gevent.sleep(sleep_time)

    def run(self):
        """
        Master loop of the game :
        Handles the games and runs the step
        """

        while True:
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
            self.spawn_players()
            alive_players = copy.copy(self.players)

            # Main game loop
            while len(alive_players) > 0:
                self.run_action_step(alive_players)
                self.run_display(alive_players)

            gevent.sleep(3)

    def start(self):
        gevent.spawn(self.run_server)

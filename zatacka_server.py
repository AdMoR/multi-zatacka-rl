import copy
import os
import time
import logging
import numpy
import json
import gevent
from flask import Flask, render_template
from flask_sockets import Sockets
import math
import random
from game_content import Player, BotPlayer, COLORS, Zatacka


app = Flask(__name__)
app.debug = True #'DEBUG' in os.environ

sockets = Sockets(app)


@app.route('/')
def index():
    return render_template('index.html')


@sockets.route('/submit')
def submit(ws):
    player = zatacka.register_player()
    if player is None: # there were already 6 players in the game
        return

    print('submit', ws, dir(ws))
    while ws is not None:
        gevent.sleep(0.01)
        message = ws.receive()

        if message:
            app.logger.info('received: {}'.format(message))
            player.process(message, None)
    #zatacka.remove_player(player)


@sockets.route('/receive')
def receive(ws):
    # register ws
    zatacka.register_observer(ws)

    print('receive', ws, dir(ws))
    while ws is not None:
        gevent.sleep(0.01)


zatacka = Zatacka()
zatacka.start()

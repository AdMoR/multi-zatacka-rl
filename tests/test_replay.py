import sys
import time
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
import copy
from unittest import TestCase
from game_content.double_q_network import GridReplayAdapter, DoubleQNetwork
from game_content import ZatackaPlayground, Grid


class TestZatackaReplay(TestCase):
    '''
    Class testing the Zatack replay.
    '''

    def test_game_replaying(self):

        zatacka = ZatackaPlayground()
        zatacka.run_bot_training(context=False, nb_loops=301)

        for player in zatacka.players:
            zatacka.display_debug_frame(player, zatacka.grid.grid)
            print(len(player.replay_adapter.phi_history))

    def test_batch_learn_network(self):
        """
        Test if the backward training of the alg is functionning
        """
        dqzat = DoubleQNetwork(batch_size=10, time_frame_size=6,
                               image_size=(300, 300), action_size=3)

        st = nd.zeros((10, 6, 300, 300))
        stpo = nd.zeros((10, 6, 300, 300))
        rt = nd.zeros((10,))
        at = nd.zeros((10,))
        tt = nd.zeros((10,))
        loss = dqzat.train_one_batch(st=st, stpo=stpo, at=at, rt=rt, tt=tt)
        print(loss)



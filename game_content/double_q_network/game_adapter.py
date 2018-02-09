from abc import abstractmethod, ABCMeta
import mxnet as mx
import mxnet.ndarray as nd
import numpy
import random


class AbstractReplayAdapter(object, metaclass=ABCMeta):
    """
    Class handling the replays
    """

    def __init__(self, time_frame_size):
        pass

    @abstractmethod
    def build_phi_replay(self, batch_size):
        pass


class GridReplayAdapter(AbstractReplayAdapter):

    def __init__(self, buffer_size, time_frame_size, game_size, action_size):
        # Vars used on the NN side to store data as mx.nd
        self.memory = []
        self.buffer_size = buffer_size
        self.time_frame_size = time_frame_size
        self.game_size = game_size
        self.action_size = action_size

        # Game status object storage
        self.grid_history = {}
        self.action_history = {}
        self.reward_history = {}
        self.termination_history = {}
        self.phi_history = {}

        # Performance logging
        # TODO : not used yet
        self.loss_history = {}

    def build_phi_replay(self, batch_size, time_step):
        """
        This function creates a random minibatch from the play history
        We chose a set of random t from the reward history
        from it we retrieve the corresponding phi_t,phi_t+1 and a_t
        """
        replay_dict = {"st": nd.zeros((batch_size, self.time_frame_size,
                                          self.game_size[0], self.game_size[1])),
                       "stpo": nd.zeros((batch_size, self.time_frame_size,
                                                   self.game_size[0], self.game_size[1])),
                       "at": nd.zeros((batch_size,)),
                       "rt": nd.zeros((batch_size,)),
                       "tt": nd.ones((batch_size,))}
        for i, t in zip(range(batch_size), random.sample(list(range(self.time_frame_size + 1, time_step)), batch_size)):
            replay_dict["st"][i, :, :, :] = self.phi_history[t]
            replay_dict["stpo"][i, :, :, :] = self.phi_history[t + 1]
            replay_dict["at"][i] = self.action_history[t]
            replay_dict["rt"][i] = self.reward_history[t]

        return replay_dict

    #####################
    #  History storage  #
    #####################

    def store_grid_in_history(self, grid, t, player):
        """
        The grid type is a list of list
        It must be transformed to a player invariant view and a mxnet array to be used later
        """
        print("Storing grid with timestep t {}".format(t))
        nd_grid = self._transform_grid_to_nd_mat(grid, player)
        self.grid_history[t] = nd_grid

    def store_action_in_history(self, action, t):
        """
        Store each action after each time stamp
        """
        #print("Storing action {} with timestep t {}".format(action, t))
        self.action_history[t] = action

    def store_reward_in_history(self, reward, t):
        """
        Store the reward given by the game
        """
        #print("Storing reward {} with timestep t {}".format(reward, t))
        self.reward_history[t] = reward

    def store_death_in_history(self, alive, t):
        self.termination_history[t] = alive

    def build_phi_t(self, t):
        """
        Phi_t correspond to the last n sample from the game state
        n is in fact the time_frame_size parameters on the init

        We cant produce phi_t if there is less than n time stamps
        """
        if t > self.time_frame_size:
            self.phi_history[t] = nd.zeros((self.time_frame_size, self.game_size[0], self.game_size[1]))
            for i in range(self.time_frame_size):
                self.phi_history[t][i, :, :] = self.grid_history[len(self.grid_history) - (i + 1)]

        if t > self.time_frame_size:
            return self.phi_history[t]
        else:
            return None

    def _transform_grid_to_nd_mat(self, grid, player, debug=False):
        """
        transform a grid (list of list) to nd_array with self centered view
        """

        # First adapt the grid format to reach an image like representation
        self_view_grid = [[0] * len(vec) for vec in grid]

        # Transform the map to be self centered
        for i, vec in enumerate(grid):
            for j, val in enumerate(vec):
                if val == 0:
                    continue
                elif val == player.id:
                    self_view_grid[i][j] = 1
                else:
                    self_view_grid[i][j] = -1

        return nd.array(numpy.array(self_view_grid))




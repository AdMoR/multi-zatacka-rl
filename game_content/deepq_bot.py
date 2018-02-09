from abc import abstractmethod, ABCMeta
from .bot import BotPlayer
import random
import numpy
import mxnet as mx
from .double_q_network import GridReplayAdapter, DoubleQNetwork


class AbstractDeepQGameAdapter(object, metaclass=ABCMeta):

    def __init__(self, buffer_size=10, time_frame_size=10,
                 game_size=(80, 80), action_size=4, batch_size=10,
                 gamma=0.9, ctx=mx.cpu(), num_layers=2):
        """
        The double Q has this order for its steps
        1 : current state of the game
        2 : get the action depending on the state at time t
        3 : collect the new state after all the players have moved in update s_t+1
        4 : get the reward at time t r_t

        However given the flow of the bot calls the order is : ??????????
        reward, state, next_action, next_state
        In other words we store : reward[t-1] state[t] action[t]
        On update, we can get the transition at t-1


        action[t] is seen two times in the process but should be registered only once
        Thus it will be done in the update step
        """
        self.replay_adapter = GridReplayAdapter(buffer_size, time_frame_size,
                                                game_size, action_size)
        self.commands = range(action_size)
        self.buffer_size = buffer_size
        self.time_frame_size = time_frame_size
        self.game_size = game_size
        self.network = DoubleQNetwork(batch_size, time_frame_size, game_size, action_size,
                                      ctx=ctx, gamma=gamma, num_layers=num_layers)
        self.epsilon = 0.5
        self.batch_size = 10

        # Update and freeze parameter for the parameter changes
        self.update_time = 1000
        self.freeze_time = 1000

    def process_game_state(self, game_state, time_step, score, alive):
        """
        In order to have everything working in one step, the reward from past
        grid update is stored now,
        The rest works pretty smoothly, get grid_t, action_t and grid_t+1

        :param game_state:
        :param time_step:
        :param score:
        :return:
        """
        # We are at time t
        # According to the process defined above, we register reward t-1

        self.time_step = time_step
        self.replay_adapter.store_reward_in_history(score, time_step - 1)
        self.replay_adapter.store_death_in_history(alive, time_step - 1)
        self.replay_adapter.store_grid_in_history(grid=game_state.grid, t=time_step, player=self)

        # Retrieve grid feature to feed the network if possible and not random step
        # We have already the state t in memory from last round
        state = self.replay_adapter.build_phi_t(t=time_step)

        # This is the epsilon greedy switch
        if random.random() < self.epsilon and state is not None:

            # Retrieve action from advantage network
            action = int(self.network.get_action(state).asnumpy()[0])
        else:
            print("random")
            action = random.sample(list(self.command_to_action.keys()), 1)[0]
        self.command = self.index_to_command(action)

        # Store partial state of the game
        self.replay_adapter.store_action_in_history(action=action, t=time_step)

    def network_backward(self, time_step):
        """
        Function used to learn something to the Q_net
        """
        # The replay adapter retrieved batch size game state ( time_frame * grid_size )
        # We have then a batch_size * time_frame * grid_size * grid_size
        # The target is laso defined in this function as the error on the discounted reward
        # Dy ~ || Rt + gamma * Qt+1 - Qt ||
        # Dy = || Rt + gamma *
        data_dict = self.replay_adapter.build_phi_replay(self.batch_size, time_step)
        self.network.train_one_batch(**data_dict)

    @abstractmethod
    def index_to_command(self, action):
        """
        The Q network returns something in the self.command from an index
        This is used to make the proper conversion
        """
        pass



class DeepQBotPlayer(AbstractDeepQGameAdapter, BotPlayer):
    """
    The double Q has this order for its steps
    1 : current state of the game
    2 : get the action depending on the state at time t
    3 : collect the new state after all the players have moved in update s_t+1
    4 : get the reward at time t r_t

    However given the flow of the bot calls the order is : ??????????
    reward, state, next_action, next_state
    In other words we store : reward[t-1] state[t] action[t]
    On update, we can get the transition at t-1


    action[t] is seen two times in the process but should be registered only once
    Thus it will be done in the update step
    """

    def __init__(self, id_, buffer_size=10, time_frame_size=10,
                 game_size=(80, 80), action_size=4, num_layers=2):
        BotPlayer.__init__(self, id_)
        AbstractDeepQGameAdapter.__init__(self, buffer_size, time_frame_size,
                                          game_size, action_size, num_layers=num_layers)
        self.is_human = False
        self.command_to_action = {0: None, 1: "straight", 2: "left", 3: "right"}
        self.time_step = 0
        self.reward = 0

    def process(self, message, game_state):
        '''
        Bots dont pass message from the interface
        They process the last state of the grid
        '''

        # The time and score are assigned from the game through the Player interface
        self.process_game_state(game_state, self.time_step, self.reward, int(self.alive))

    def update(self, grid):
        # Do the drawing etc
        BotPlayer.update(self, grid)

        # Here we have all the data for transition at t-1
        # (because we stored the reward from step t-1 on step t)

        if self.time_step % self.update_time == 0 and\
           self.time_step > self.buffer_size + self.time_frame_size:
            self.network_train()

        if self.time_step % self.freeze_time == 0:
            # Freeze the parameters learnt and change the forward network
            self.network.copy_to_freezed_network()

    def index_to_command(self, action):
        return self.command_to_action[action]


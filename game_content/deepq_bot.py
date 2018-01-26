from .abstract_player import Snake, COLORS
from .bot import BotPlayer
import random
import math
from .double_q_network import ZatackaReplayAdapter
import numpy


class DeepQBotPlayer(BotPlayer):

    def __init__(self, id, buffer_size=10, time_frame_size=10, game_size=(80, 80), action_size=4):
        """
        The double Q has this order for its steps
        1 : current state of the game
        2 : get the action depending on the state at time t
        3 : collect the new state after all the player have moved in update s_t+1
        4 : get the reward at time t r_t

        However given the flow of the bot calls the order is : ??????????
        reward, state, next_action, next_state
        In other words we store : reward[t-1] state[t] action[t]
        On update, we can get the transition at t-1


        action[t] is seen two times in the process but should be registered only once
        Thus it will be done in the update step
        """
        super(DeepQBotPlayer, self).__init__(id)
        self.is_human = False
        self.replay_adapter = ZatackaReplayAdapter(buffer_size, time_frame_size,
                                                   game_size, action_size)
        self.commands = range(action_size)
        self.buffer_size = buffer_size
        self.time_frame_size = time_frame_size
        self.game_size = game_size
        self.network = None
        self.epsilon = 1.0
        self.batch_size = 10

        # Update and freeze parameter for the parameter changes
        self.update_time = 1000
        self.freeze_time = 1000

    def process(self, message, game_state):
        '''
        Bots dont pass message from the interface
        They process the last state of the grid
        '''

        # We are at time t
        # According to the process defined above, we register reward t-1

        self.time_step = message
        self.replay_adapter.store_reward_in_history(self.score, self.time_step - 1)
        self.replay_adapter.store_grid_in_history(grid=game_state.grid, t=self.time_step, player=self)

        # Retrieve grid feature to feed the network if possible and not random step
        # We have already the state t in memory from last round
        state = self.replay_adapter.build_phi_t(t=self.time_step)
        if random.random() > self.epsilon and state is not None:

            # Get the network feature
            V, A = self.network.q_forward(state)

            # Retrieve action from advantage network
            action = numpy.argmax(A.asnumpy())
            self.command = self.index_to_command(action)
        else:
            self.command = random.choice(self.commands)
            action = self.command

        # Store partial state of the game
        self.replay_adapter.store_action_in_history(action=action, t=self.time_step)

    def update(self, grid):
        # Do the drawing etc
        super(DeepQBotPlayer, self).update(grid)
        pass

        # Here we have all the data for transition at t-1
        # (because we stored the reward from step t-1 on step t)

        if self.time_step % self.update_time == 0 and\
           self.time_step > self.buffer_size + self.time_frame_size:
            # Here get the backward
            data_dict = self.replay_adapter.build_phi_replay(self.batch_size)
            self.network.backward_pass(**data_dict)

        if self.time_step % self.freeze_time == 0:
            # Freeze the parameters learnt and change the forward network
            do_stuff = 1

    def index_to_command(self, action):
        command_to_action = {0: None, 1: "straight", 2: "left", 3: "right"}
        return command_to_action[action]






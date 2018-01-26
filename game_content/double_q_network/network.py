import mxnet as mx
import mxnet.ndarray as nd
import numpy
from resnet_sym import get_symbol
from collections import namedtuple
from .custom_loss import *


class DQNInitializer(mx.initializer.Xavier):

    def _init_bias(self, _, arr):
        arr[:] = .1

    def _init_default(self, name, _):
        pass


class DoubleQZatacka(object):

    def __init__(self, batch_size, time_frame_size, image_size, action_size, ctx=mx.cpu(), gamma=0.9):

        # Define some parameters
        input_shape = (batch_size, time_frame_size, image_size[0], image_size[1])
        self.gamma = gamma

        # Build the base network from the symbols
        flat = get_symbol(num_classes=action_size, num_layers=18,
                          image_shape=(time_frame_size, image_size[0], image_size[1]))
        value_head = mx.sym.FullyConnected(data=flat, num_hidden=1, name='fc_v')
        advantage_head = mx.sym.FullyConnected(data=flat, num_hidden=action_size, name='fc_a')
        broadcasted_advantage_mean = (1. / action_size) *\
            mx.sym.broadcast_axis(mx.sym.expand_dims(mx.sym.sum(advantage_head, axis=1), axis=1),
                                  axis=1, size=action_size)
        advantage_head = advantage_head - broadcasted_advantage_mean
        q_value = mx.sym.broadcast_axis(value_head, axis=1, size=action_size) + advantage_head

        # The first module uses the symbols defined for the Q to build the yddqn target value
        self.target_q_mod = q_value.simple_bind(ctx=ctx, grad_req='write', data=input_shape)

        # Now we have to build the loss function for the network update
        loss = mx.sym.Custom(data=q_value, name='loss', op_type='dqnloss')
        self.loss_q_mod = loss.simple_bind(ctx=ctx, grad_req='write', data=input_shape)

        initializer = DQNInitializer(factor_type='in')
        names = loss.list_arguments()
        for name in names:
            initializer(name, self.loss_q_mod.arg_dict[name])

        # Copy weights and init optimizer and updater
        self.copy_to_freezed_network()
        self.optimizer = mx.optimizer.create(
            name='adagrad',
            learning_rate=0.01,
            eps=0.01,
            wd=0.0,
            clip_gradient=None,
            rescale_grad=1.0)
        self.updater = mx.optimizer.get_updater(self.optimizer)

    def train_one_batch(self, st, stpo, at, rt, tt, unfreeze_weight=False):
        """
        From all the state needed for a forward backward,
        each sub result is calculated to reach the loss backward
        """

        # 1 : get the y_ddqn
        target_q = self.target_q_mod.forward(is_train=False, data=stpo)
        a_q = self.loss_q_mod.forward(is_train=False, data=stpo)

        # Get the a = argmax Q(st, theta_i)
        a = nd.argmax_channel(a_q[0])
        y_ddqn = rt + (1.0 - tt) * self.gamma * nd.choose_element_0index(target_q[0], a)

        # 2 : build the loss on the current batch
        current_q = self.loss_q_mod.forward(is_train=True,
                                            data=st, loss_action=at, loss_target=y_ddqn)
        self.loss_q_mod.backward()

        # 3 : Update parameters
        self.update_weights(self.loss_q_mod, self.updater)

        # 4 : Calculate the loss
        loss = nd.sum((y_ddqn - nd.choose_element_0index(current_q[0], a)) ** 2, axis=0)

        # 5 : the occasional forward weight updater
        if unfreeze_weight:
            self.copy_to_freezed_network()

        return loss[0]

    def copy_to_freezed_network(self):
        main_net_params = {k: v for k, v in self.loss_q_mod.arg_dict.iteritems()
                           if k in self.target_q_mod.arg_dict.keys()}
        main_net_aux = {k: v for k, v in self.loss_q_mod.aux_dict.iteritems()
                        if k in self.target_q_mod.aux_dict.keys()}
        self.target_q_mod.copy_params_from(arg_params=main_net_params, aux_params=main_net_aux)

    @staticmethod
    def update_weights(executor, updater):
        for ind, k in enumerate(executor.arg_dict):
            if k.endswith('weight') or k.endswith('bias'):
                updater(
                    index=ind,
                    grad=executor.grad_dict[k],
                    weight=executor.arg_dict[k])

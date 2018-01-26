import mxnet as mx
import numpy


class DQNLoss(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        """
        Trcikz here, we don't forward the action and the reward
        """
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """
        Define the backward func of the loss with input parameters
         defined from list arguments from the prop op
        """
        x = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        target = in_data[2].asnumpy()
        dx = in_grad[0]
        ret = numpy.zeros(shape=dx.shape, dtype=numpy.float32)
        ret[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - target, -1, 1)
        self.assign(dx, req[0], ret)


@mx.operator.register("dqnloss")
class DQNLossProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(DQNLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'target']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        target_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, target_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return DQNLoss()


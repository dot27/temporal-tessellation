"""Building the RNN model."""

import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def get_config(split, debug=False):
  if split == 'train':
    config = MpiiConfig()
    if debug:
      config.hidden_size = 20
      config.input_feature_num = 2 * 20

    return config

  elif split == 'prediction':
    config = Predictionconfig()
    if debug:
      config.hidden_size = 20
      config.input_feature_num = 2*20

    return config


class MpiiConfig(object):
  """Small config."""
  init_scale = 0.04
  learning_rate = 0.5
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.86
  lr_decay = 0.69
  batch_size = 20
  hidden_size = 2000
  input_feature_num = 2 * hidden_size
  nn_num = 5


class Predictionconfig(MpiiConfig):
  """Configuration for prediction"""
  batch_size = 1
  num_steps = 1


class MPIImodel(object):
  """The PTB model."""

  def __init__(self, is_training, config):

    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    input_feature_num = config.input_feature_num

    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, input_feature_num])
    self._targets = tf.placeholder(tf.float32, [batch_size, num_steps, size])

    inputs = self._input_data
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]

    target_outputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, num_steps, self._targets)]

    self._input = tf.reshape(tf.concat(1, inputs), [-1, size])
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

    target_output = tf.reshape(tf.concat(1, target_outputs), [-1, size])
    output = tf.reshape(tf.concat(1, outputs), [-1, size])

    self._output = output

    diff_square = tf.pow(output - target_output, 2)
    loss = tf.reduce_sum(diff_square, reduction_indices=1)

    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def output(self):
    return self._output

  @property
  def input(self):
    return self._input

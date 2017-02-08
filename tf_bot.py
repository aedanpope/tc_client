
import tc_client
import argparse
import sys
import numpy as np
import tensorflow as tf
import random
import itertools


MOVES = [(5,0), (-5,0), (0,5), (0,-5), (3,3), (3,-3), (-3,3), (-3,-3)]
MOVE_RANGE = (70,150)


# Learning params:
EPS = 0.1
GAMMA = 0.99


## Generally, try and keep everything in the [-1,1] range.
## TODO: consider trying [0,1] range for some stuff e.g. HP.

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


class TFBot:
  # Infra
  tf_session = None

  # Q-network config.
  # TF-class objects.
  inp_x = None
  out_y = None
  calc_q = None
  predict_y = None
  train = None

  # State variables.
  prev_inp = None
  prev_action = None
  target_q = None
  total_reward = None

  def __init__(self):
    self.setup_q_nn()

    self.tf_session = tf.InteractiveSession()
    self.tf_session.run(tf.initialize_all_variables())
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())


  def setup_q_nn():
    self.inp_x = tf.placeholder(tf.float32, [1, 30])
    self.out_y = tf.placeholder(tf.float32, [1, 14])
    # inp_x = tf.placeholder(tf.float32, [None, 10,3])
    """ inp_x = [1,30], out_y = [1,14] """
    # x_flat = tf.reshape(x, [-1, 30])

    # Densely Connected Layer - fully-connected layer with 1024
    w = weight_variable([30, 14])
    b = bias_variable([14])
    self.calc_q = tf.matmul(self.inp_x, w) + b
    self.predic_y = tf.argmax(self.calc_q,1)
    # h_fc1 = tf.nn.relu(softmax)

    # Obtain the loss by taking the sum of squares
    # difference between the target and prediction Q values.
    loss = tf.reduce_sum(tf.square(self.out_y - self.calc_q))
    self.train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.zf9kblra3


  def get_commands(self, state):
    inp_val = TFBot.state_to_input(state)

    # Process reward for previous action taken.
    # TODO calc this as unit deaths or something.
    reward = 1
    total_reward += reward

    # Obtain the Q' values for previous action by feeding the new state
    # through our network
    q1 = sess.run(self.calc_q,feed_dict={inp_x:inp_val})
    max_q1 = np.max(q1)
    # Set the reward for taking the previous action, based on how good
    # the current state is.
    # BELLMAN EQUATION
    target_q[self.prev_action] = reward + GAMMA*max_q1
    # Train.
    # For the previous input & subsequent prediction - apply bellman based on the
    # actual reward.
    sess.run(self.train,feed_dict={inp_x:self.prev_inp,self.out_y:target_q})

    # Generate new action.

    # Inputs

    # out_val = [TFBot.generate_random_onehot(14) for i in range(0, 5)]

    # Choose an action by greedily (with e chance of random action) from the Q-network
    # We need all the Q vals for learning.
    action,all_q = sess.run([self.predict_y, self.calc_q],feed_dict={inp_x:inp_val})
    # Consider getting a random action instead of what the model tells us.
    # E is a hyperparam of how often to do this, EPS = 0.1
    if random.random() < EPS:
        action = random.randint(0,13)

    # Save output of q-network, so we can train after state update.
    self.prev_action = action
    self.prev_inp = inp_val
    self.target_q = all_q

    # Outputs
    return TFBot.output_to_command(action, state)

  # @staticmethod
  # def generate_random_onehot(n):
  #   xs = range(0, n)
  #   random.shuffle(xs)
  #   return xs

  @staticmethod
  def output_to_command(action, state):
    """ out_t = [14] """
    """ action in [0 .. 13]"""

    friendly_units = state.get_friendly_units()
    enemy_units = state.get_enemy_units()


    commands = []
    for i in range(0, min(len(friendly_units), 5)):
      friendly = friendly_units[i]
      # one-hot
      # 0 = do nothing
      # 1-8 = move 5 units in dir
      # 9-13 = attack unit num 0-4
      a = action
      # a = np.argmax(y[i])
      if a == 0: continue
      elif 1 <= a and a <= 8:
        del_x, del_y = MOVES[a-1]
        move_x = TFBot.constrain(friendly.x + del_x, MOVE_RANGE)
        move_y = TFBot.constrain(friendly.y + del_y, MOVE_RANGE)
        commands.append([friendly.id, tc_client.UNIT_CMD.Move, -1, move_x, move_y])
      elif 9 <= a and a <= 13:
        e_index = a - 9
        if e_index >= len(enemy_units): continue # Do nothing, can't attack unit that doesn't exist.
        commands.append([friendly.id, tc_client.UNIT_CMD.Attack_Unit, enemy_units[e_index].id])
      else:
        raise Exception("Invalid one-hot: " + str(a))
    return commands


  @staticmethod
  def state_to_input(state):
    """ returns [10,3] """
    #max 10 units
    # First 5 inputs are friendly units
    # second 5 are enemy units
    friendly_tensor = TFBot.pack_unit_tensor(TFBot.units_to_tensor(state.get_friendly_units()), 5)
    enemy_tensor = TFBot.pack_unit_tensor(TFBot.units_to_tensor(state.get_enemy_units()), 5)
    return itertools.chain.from_iterable(friendly_tensor + enemy_tensor)

  @staticmethod
  def units_to_tensor(units):
    return [TFBot.unit_to_vector(unit) for unit in units]

  @staticmethod
  def unit_to_vector(unit):
    return [
            # 1.0 if is_friendly else -1.0,
            TFBot.norm(unit.x, (MOVE_RANGE)),
            TFBot.norm(unit.y, (MOVE_RANGE)),
            TFBot.norm(unit.health, (0, 40)),
            ]

  @staticmethod
  def pack_unit_tensor(unit_tensor, final_rows):
    return unit_tensor + [[0,0,0] for i in range(len(unit_tensor), final_rows)]

  @staticmethod
  def norm(x, min_max):
    # Map to range [-1.0,1.0]
    return (2.0*(x - min_max[0]) / (min_max[1] - min_max[0])) - 1

  @staticmethod
  def constrain(x, min_max):
    # truncate X to fit in [x_min,x_max]
    return min(max(x,min_max[0]),min_max[1])



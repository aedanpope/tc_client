
import tc_client
import argparse
import sys
import numpy as np
import tensorflow as tf
import random
import itertools
import sys
import math



MOVES = [(5,0), (-5,0), (0,5), (0,-5), (3,3), (3,-3), (-3,3), (-3,-3)]
MOVE_RANGE = (70,150)


# Learning params:

# Exploration: probability of taking a random action instead of that suggested by the q-network.
EPS = 0.2
# Dicount-rate: how much we depreciate future rewards in value for each state step.
# GAMMA = 0.99
GAMMA = 0.99
# Initial weights on neuron connections, need to start small so max output of NN isn't >> reward.
W_INIT = 0.01
# Reward val
REWARD = 1


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
  sess = None

  # Q-network config.
  # TF-class objects.
  tf_inp = None # Prob should rename input to state to be consistent with Bellman.
  tf_q = None
  tf_target_q = None
  tf_action = None
  tf_train = None

  # State variables.
  prev_inp = None
  prev_action = None
  prev_q = None
  prev_unit_diff = None
  total_reward = None

  def __init__(self):
    self.setup_q_nn()

    self.sess = tf.InteractiveSession()
    self.sess.run(tf.initialize_all_variables())

    self.total_reward = 0
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())


  def setup_q_nn(self):
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.zf9kblra3
    self.tf_inp = tf.placeholder(tf.float32, [1, 30])

    # Densely Connected Layer - fully-connected layer with 1024
    # w = weight_variable([30, 14])
    # Weights need to start of small, so that we don't get randomly high Q values.
    w = tf.Variable(tf.random_uniform([30,14],0,W_INIT))
    # b = bias_variable([14])
    # self.tf_q = tf.matmul(self.tf_inp, w) + b
    self.tf_q = tf.matmul(self.tf_inp, w)
    self.tf_action = tf.argmax(self.tf_q,1)
    # h_fc1 = tf.nn.relu(softmax)

    # Obtain the loss by taking the sum of squares
    # difference between the target and prediction Q values.
    self.tf_target_q = tf.placeholder(tf.float32, [1, 14])
    loss = tf.reduce_sum(tf.square(self.tf_target_q - self.tf_q))
    self.tf_train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


  def get_commands(self, state):
    inp = TFBot.state_to_input(state)
    num_friendly_units = len(state.friendly_units)
    num_enemy_units = len(state.enemy_units)

    # Choose an action by greedily (with e chance of random action) from the Q-network
    # We need all the Q vals for learning.
    print "inp = " + str(inp)
    print "inp_len = " + str(len(inp[0]))
    action,q = self.sess.run([self.tf_action, self.tf_q],
                                   feed_dict={self.tf_inp:inp})
    action = action[0]
    # Sometimes get a random action instead of what the model tells us, to explore.
    # Hyperparam EPS = 0.1
    if random.random() < EPS:
        action = random.randint(0,13)

    # Now we can train for the *previous* action taken, since we have:
    # - the state following that action
    # - the possible set of q-values for the actions possible on the next state
    # and thus can compute the bellman equation.
    if (self.prev_action is not None):
      # Determine reward based on improvement in unit-advantage, or 10 if we won the whole battle.
      reward = 0
      if state.battle_just_ended:
        if state.battle_won:
          reward = REWARD
      else:
        if (num_friendly_units - num_enemy_units) > self.prev_unit_diff:
          reward = REWARD
        # reward = (num_friendly_units - num_enemy_units) - self.prev_unit_diff
      self.total_reward += reward
      self.train(self.prev_inp, self.prev_action, self.prev_q, q, reward)

    # Save the latest output of q-network, so we can train when the next state update arrives.
    self.prev_action = action
    self.prev_inp = inp
    self.prev_q = q
    self.prev_unit_diff = num_friendly_units - num_enemy_units

    print "action = " + str(action)
    print "total_reward = " + str(self.total_reward)

    # Outputs
    return TFBot.output_to_command(action, state)


  def train(self, prev_inp, prev_action, prev_q, next_q, reward):
      # Train with loss from Bellman equation
      print "TRAIN"
      print "reward = " + str(reward)
      print "prev_inp = " + str(prev_inp)
      print "prev_action = " + str(prev_action)
      print "prev_q = " + str(prev_q)
      print "next_q = " + str(next_q)
      target_q = prev_q
      cur_v = target_q[0, prev_action]
      new_v = reward + GAMMA*np.max(next_q)
      print "cur_v = " + str(cur_v) + ", new_v = " + str(new_v)
      print "delta = " + str(cur_v - new_v) + ", pc = " + str(new_v / cur_v)
      if math.isnan(prev_q[0,0]): sys.exit()
      if new_v > 100: sys.exit()
      # TODO: should there be a hyperparam learning rate here?
      target_q[0, prev_action] = reward + GAMMA*np.max(next_q)
      print "target_q2 = " + str(target_q)
      # print "target_q = " + str(target_q)

      # By passing prev_inp the network will regenerate prev_q, and then apply loss where
      # it differs from target_q - to train it to remember target_q instead of prev_q
      self.sess.run(self.tf_train,
                    feed_dict={self.tf_inp:prev_inp,
                               self.tf_target_q:target_q})

  @staticmethod
  def output_to_command(action, state):
    """ out_t = [14] """
    """ action in [0 .. 13]"""


    friendly_units = state.friendly_units.values()
    enemy_units = state.enemy_units.values()


    commands = []
    # for i in range(0, max(len(friendly_units), 5)):
    for i in range(0, len(friendly_units)):
      if i == 5: break
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
    """ returns [1,30] """
    # """ returns [10,3] """
    #max 10 units
    # First 5 inputs are friendly units
    # second 5 are enemy units
    friendly_units = state.friendly_units.values()
    enemy_units = state.enemy_units.values()
    if len(friendly_units) > 5: friendly_units = friendly_units[:5]
    if len(enemy_units) > 5: enemy_units = enemy_units[:5]

    friendly_tensor = TFBot.pack_unit_tensor(TFBot.units_to_tensor(friendly_units), 5)
    enemy_tensor = TFBot.pack_unit_tensor(TFBot.units_to_tensor(enemy_units), 5)
    # ts = friendly_tensor + enemy_tensor
    # return
    # return [ [x] for x in itertools.chain.from_iterable(friendly_tensor + enemy_tensor)]
    return [list(itertools.chain.from_iterable(friendly_tensor + enemy_tensor))]

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
    # return (2.0*(x - min_max[0]) / (min_max[1] - min_max[0])) - 1
    # Map to range [0,1.0]
    return (float(x) - min_max[0]) / (min_max[1] - min_max[0])

  @staticmethod
  def constrain(x, min_max):
    # truncate X to fit in [x_min,x_max]
    return min(max(x,min_max[0]),min_max[1])



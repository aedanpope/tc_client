
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
FRIENDLY_TENSOR_SIZE = 7
ENEMY_TENSOR_SIZE = 3
MAX_FRIENDLY_UNITS = 1 # 5 for marines
MAX_ENEMY_UNITS = 1 # 5 for marines
INP_SHAPE = MAX_FRIENDLY_UNITS * FRIENDLY_TENSOR_SIZE + MAX_ENEMY_UNITS * ENEMY_TENSOR_SIZE
OUT_SHAPE = 9 + MAX_ENEMY_UNITS
V = True  # Verbose


# Learning params:

# Exploration: probability of taking a random action instead of that suggested by the q-network.
EPS = 0.9
EPS_EPOCH = 0.999
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

  # Learning params
  explore = None

  # State variables.
  prev_inp = None
  prev_action = None
  prev_q = None
  prev_unit_diff = None
  prev_hp_diff = None
  total_reward = None

  n = None
  v = True # verbose

  def __init__(self):
    self.setup_q_nn()

    self.sess = tf.InteractiveSession()
    # self.sess.run(tf.initialize_all_variables())
    self.sess.run(tf.global_variables_initializer())

    self.total_reward = 0
    self.n = 0
    self.explore = 0.9
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())


  def setup_q_nn(self):
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.zf9kblra3

    self.tf_inp = tf.placeholder(tf.float32, [1, INP_SHAPE])

    # Densely Connected Layer - fully-connected layer with 1024
    # w = weight_variable([30, 14])
    # Weights need to start of small, so that we don't get randomly high Q values.

    # OLD Q NETWORK:
    # w = tf.Variable(tf.random_uniform([30,14],0,W_INIT))
    # self.tf_q = tf.nn.relu(tf.matmul(self.tf_inp, w))


    # b = bias_variable([14])
    # self.tf_q = tf.matmul(self.tf_inp, w) + b
    # h_fc1 = tf.nn.relu(softmax)
    # self.tf_q = tf.matmul(self.tf_inp, w)

    ## MATMUL
    # self.tf_q = tf.matmul(self.tf_inp,
    #                tf.Variable(tf.random_uniform([INP_SHAPE,OUT_SHAPE],0,W_INIT)))

    ## 1-layer RELU
    # self.tf_q = tf.nn.relu(tf.matmul(self.tf_inp,
    #                tf.Variable(tf.random_uniform([INP_SHAPE,OUT_SHAPE],0,W_INIT))))

    ## 2-layer RELU
    layer_1 = tf.nn.relu(tf.matmul(self.tf_inp,
                   tf.Variable(tf.random_uniform([INP_SHAPE,20],0,W_INIT))))
    self.tf_q = tf.nn.relu(tf.matmul(layer_1,
                   tf.Variable(tf.random_uniform([20,OUT_SHAPE],0,W_INIT))))

    # TODO: Consider adding multiple channels.

    self.tf_action = tf.argmax(self.tf_q,1)

    # Obtain the loss by taking the sum of squares
    # difference between the target and prediction Q values.
    self.tf_target_q = tf.placeholder(tf.float32, [1, OUT_SHAPE])
    loss = tf.reduce_sum(tf.square(self.tf_target_q - self.tf_q))
    self.tf_train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


  def get_commands(self, state):
    # print "n = " + str(self.n)
    self.n += 1

    # Skip every second frame.
    # This is because marine attacks take 2 frames, and so it can finish an attack started in a prev frame.
    # Need to make the current order an input into the NN so it can learn to return order-0 (no new order)
    # if already performing a good attack.
    # if self.n % 2 != 1: return []
    inp = TFBot.state_to_input(state)
    num_friendly_units = len(state.friendly_units)
    num_enemy_units = len(state.enemy_units)
    friendly_hp = 0 if not state.friendly_units else state.friendly_units.values()[0].health
    enemy_hp = 0 if not state.enemy_units else state.enemy_units.values()[0].health

    if V: print "inp = " + str(inp)
    if V: print "inp_len = " + str(len(inp[0]))

    self.v = num_enemy_units != 0 and num_friendly_units != 0

    # if self.v:
    # print "inp = " + str(inp)
    # print "inp_len = " + str(len(inp[0]))

    # Choose an action by greedily (with e chance of random action) from the Q-network
    # We need all the Q vals for learning.
    action,q = self.sess.run([self.tf_action, self.tf_q],
                                   feed_dict={self.tf_inp:inp})
    action = action[0]
    # Sometimes get a random action instead of what the model tells us, to explore.
    # Hyperparam EPS = 0.1
    if random.random() < self.explore:
        action = random.randint(0,OUT_SHAPE-1)
    # EPS = EPS * EPS_EPOCH
    self.explore *= EPS_EPOCH

    # Now we can train for the *previous* action taken, since we have:
    # - the state following that action
    # - the possible set of q-values for the actions possible on the next state
    # and thus can compute the bellman equation.
    if (self.prev_action is not None):
      # Determine reward based on improvement in unit-advantage, or 10 if we won the whole battle.
      reward = 0
      # if state.battle_just_ended:
      #   if state.battle_won:
      #     reward = REWARD
      #   self.prev_unit_diff = 0
      # else:
      # TODO: Make this work for more than a 1v1
      if (num_friendly_units > num_enemy_units and self.prev_unit_diff < 1):
      # if (num_friendly_units - num_enemy_units) > self.prev_unit_diff:
        reward = REWARD
      # elif (num_enemy_units > num_friendly_units and self.prev_unit_diff > -1):
        # reward = -float(REWARD)/5
        #
      elif (num_enemy_units == 1 and num_friendly_units == 1):
        if (friendly_hp - enemy_hp - self.prev_hp_diff > 0):
          # reward = float(REWARD)/20
          reward = float(REWARD)/20
          # reward = np.sign(friendly_hp - enemy_hp - self.prev_hp_diff) * float(REWARD)/20
      # else:
        # reward = float(REWARD)/100
        # print "unit_diff reward"
      # reward = (num_friendly_units - num_enemy_units) - self.prev_unit_diff
      self.total_reward += reward

      print "inp = " + str(inp)
      print "inp_len = " + str(len(inp[0]))
      print "prev_unit_diff = " + str(self.prev_unit_diff)
      print "prev_hp_diff = " + str(self.prev_hp_diff)
      print "action = " + str(action)
      print "total_reward = " + str(self.total_reward)
      print "explore = " + str(self.explore)

      self.train(self.prev_inp, self.prev_action, self.prev_q, q, reward)

    if num_enemy_units != 0 and num_friendly_units != 0:
      # Save the latest output of q-network, so we can train when the next state update arrives.
      self.prev_action = action
      self.prev_inp = inp
      self.prev_q = q
      self.prev_unit_diff = num_friendly_units - num_enemy_units
      self.prev_hp_diff = friendly_hp - enemy_hp

      # print "prev_unit_diff = " + str(self.prev_unit_diff)
      # print "prev_hp_diff = " + str(self.prev_hp_diff)
      # print "action = " + str(action)
      # print "total_reward = " + str(self.total_reward)
      # print "explore = " + str(self.explore)
    else:
      # But we don't train on the action taken this cycle if all the friendly or enemy units were already dead.
      # Because our actions in those states are bogus, and also half the input is zeros, so it makes the network
      # over-rotated on noise.
      self.prev_action = None
      # print "already all dead, don't train this move"

    # Outputs
    return TFBot.output_to_command(action, state)


  def train(self, prev_inp, prev_action, prev_q, next_q, reward):
      # Train with loss from Bellman equation
      # if self.v:
      print "TRAIN"
      print "reward = " + str(reward)
      print "prev_inp = " + str(prev_inp)
      print "prev_action = " + str(prev_action)
      print "prev_q = " + str(prev_q)
      print "next_q = " + str(next_q)

      target_q = prev_q
      target_q[0, prev_action] = reward + GAMMA*np.max(next_q)
      print "target_q2 = " + str(target_q)
      if all(r[0] == 0 for r in target_q):
        print "all zeros = " + str(target_q)
        sys.exit()

      if False:
        cur_v = target_q[0, prev_action]
        if (reward == 1):
          new_v = reward
        else:
          new_v = max(0.001, GAMMA*min(1, np.max(next_q)))
        for i in range(0, len(target_q[0])):
          target_q[0, i] = max(0.001,min(target_q[0, i], GAMMA))
        # new_v = reward + GAMMA*np.max(next_q)
        print "cur_v = " + str(cur_v) + ", new_v = " + str(new_v)
        print "delta = " + str(cur_v - new_v) + ", pc = " + str(new_v / cur_v)
        # TODO: should there be a hyperparam learning rate here?
        target_q[0, prev_action] = reward + GAMMA*np.max(next_q)
        target_q[0, prev_action] = new_v
        print "target_q2 = " + str(target_q)
        if math.isnan(prev_q[0,0]): sys.exit()
        if new_v > 100: sys.exit()
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
      if a == 0: continue # No new order, so prob continue with current order.
      elif 1 <= a and a <= 8:
        del_x, del_y = MOVES[a-1]
        move_x = TFBot.constrain(friendly.x + del_x, MOVE_RANGE)
        move_y = TFBot.constrain(friendly.y + del_y, MOVE_RANGE)
        commands.append([friendly.id, tc_client.UNIT_CMD.Move, -1, move_x, move_y])
      elif 9 <= a and a <= OUT_SHAPE - 1:
        e_index = a - 9
        if e_index >= len(enemy_units): continue # Do nothing, can't attack unit that doesn't exist.
        commands.append([friendly.id, tc_client.UNIT_CMD.Attack_Unit, enemy_units[e_index].id])
      else:
        raise Exception("Invalid action: " + str(a))
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
    if len(friendly_units) > MAX_FRIENDLY_UNITS: friendly_units = friendly_units[:MAX_FRIENDLY_UNITS]
    if len(enemy_units) > MAX_ENEMY_UNITS: enemy_units = enemy_units[:MAX_ENEMY_UNITS]

    friendly_tensor = TFBot.pack_unit_tensor(
        TFBot.units_to_tensor(friendly_units, True), FRIENDLY_TENSOR_SIZE, MAX_FRIENDLY_UNITS)

    enemy_tensor = TFBot.pack_unit_tensor(
        TFBot.units_to_tensor(enemy_units, False), ENEMY_TENSOR_SIZE, MAX_ENEMY_UNITS)

    # ts = friendly_tensor + enemy_tensor
    # return
    # return [ [x] for x in itertools.chain.from_iterable(friendly_tensor + enemy_tensor)]
    return [list(itertools.chain.from_iterable(friendly_tensor + enemy_tensor))]


  @staticmethod
  def units_to_tensor(units, is_friendly):
    return [TFBot.unit_to_vector(unit, is_friendly) for unit in units]


  @staticmethod
  def unit_to_vector(unit, is_friendly):
    unit_vector = [
            # 1.0 if is_friendly else -1.0,
            TFBot.norm(unit.x, (MOVE_RANGE)),
            TFBot.norm(unit.y, (MOVE_RANGE)),
            TFBot.norm(unit.health, (0, 40)),
            ]

    if is_friendly:
      is_guard = False
      is_move = False
      is_attack = False
      # Seee https://bwapi.github.io/namespace_b_w_a_p_i_1_1_orders_1_1_enum.html
      if len(unit.orders) > 0:
        order_type = unit.orders[0].type
        is_guard = order_type in [2,3] #
        is_move = order_type == 6  #
        is_attack = order_type == 10 #
      unit_vector = unit_vector + [int(is_guard), int(is_move), int(is_attack)]

    return unit_vector

  @staticmethod
  def pack_unit_tensor(unit_tensor, tensor_shape, final_rows):
    return unit_tensor + [[0 for i in range(0, tensor_shape)] for i in range(len(unit_tensor), final_rows)]

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



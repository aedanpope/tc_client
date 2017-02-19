# Policy Functions
# https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.axhe0juti
# http://karpathy.github.io/2016/05/31/rl/


import tc_client
import argparse
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import itertools
import sys
import math


MOVES = [(6,0), (-6,0), (0,6), (0,-6), (4,4), (4,-4), (-4,4), (-4,-4)]
# MOVES = [(10,0), (-10,0), (0,10), (0,-10), (7,7), (7,-7), (-7,7), (-7,-7)]
# friendly starts at (70,140), enemy starts at (100,140)
X_MOVE_RANGE = (60,120) # X_MOVE_RANGE and Y_MOVE_RANGE should be the same magnitude.
Y_MOVE_RANGE = (110,190)
MAX_FRIENDLY_LIFE = None
MAX_ENEMY_LIFE = None

FRIENDLY_TENSOR_SIZE = 14
ENEMY_TENSOR_SIZE = 8
MAX_FRIENDLY_UNITS = 1 # 5 for marines
MAX_ENEMY_UNITS = 1 # 5 for marines
INP_SHAPE = MAX_FRIENDLY_UNITS * FRIENDLY_TENSOR_SIZE + MAX_ENEMY_UNITS * ENEMY_TENSOR_SIZE
HID_SHAPE = 20 # Hidden layer shape.
OUT_SHAPE = 9 + MAX_ENEMY_UNITS
V = False  # Verbose
LEARNING_RATE = 0.01

# Learning and env params:

# Exploration: probability of taking a random action instead of that suggested by the q-network.
INITIAL_EXPLORE = 0.9
# How much we multiply the current explore value by each cycle.
EXPLORE_FACTOR = 0.9995
# Dicount-rate: how much we depreciate future rewards in value for each state step.
GAMMA = 0.99
# GAMMA = 0.9
# Initial weights on neuron connections, need to start small so max output of NN isn't >> reward.
W_INIT = 0.01
# Reward val
REWARD = 1
MINI_REWARD = 0.1
MICRO_REWARD = 0.01


FRAMES_PER_ACTION = 1


## Generally, try and keep everything in the [-1,1] range.
## TODO: consider trying [0,1] range for some stuff e.g. HP.

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


class Battle:
  stages = None
  is_end = None
  is_won = None
  trained = False

  def __init__(self):
    self.stages = []
    self.is_end = False


  def add_stage(self, stage):
    self.stages.append(stage)
    if stage.is_end:
      self.is_end = True
      self.is_won = stage.is_won

  def num_stages(self):
    return len(self.stages)

  def size(self):
    return len(self.stages)

  def get_stage(self, index):
    return self.stages[index]
  def __getitem__(self, key):
      return self.stages.__getitem__(key)

  def to_str(self):
    return ("Battle {" +
            "stages: " + str(self.stages) +
            ", is_end: " + str(self.is_end) +
            ", is_won: " + str(self.is_won) +
            "}")
  def __str__(self):
    return self.to_str()
  def __repr__(self):
    return self.to_str()


# Assumes one unit per side.
class Stage:
  # Ctor vars.
  # state = None
  friendly_life = None # Friendly HP in the stage
  enemy_life = None # Enemy HP in the stage
  is_end = None
  is_won = None
  friendly_unit = None
  enemy_unit = None

  # Vars added later:
  inp = None # Input into the neural network.
  q = None # The Q generated by the network from the inp, which determined the action if we didn't explore.
  action = None # The action that was taken on the state.
  # The reward we attribute to taking the action on the input. Changes over time as we learn more about
  # the consequences of having taken that action on that input.
  reward = None

  def __init__(self, state):
    # self.state = state

    self.friendly_unit = 0 if not state.friendly_units else state.friendly_units.values()[0]
    self.enemy_unit = 0 if not state.enemy_units else state.enemy_units.values()[0]

    # Derived values:
    self.friendly_life = 0 if not state.friendly_units else state.friendly_units.values()[0].get_life()
    self.enemy_life = 0 if not state.enemy_units else state.enemy_units.values()[0].get_life()
    self.is_end = self.friendly_life == 0 or self.enemy_life == 0
    if self.is_end:
      self.is_won = self.friendly_life > 0
    self.reward = 0

  def to_str(self):
    return ("Stage {" +
            "inp: " + str(self.inp) +
            ", q: " + str(self.q) +
            ", action: " + str(self.action) +
            ", friendly_life: " + str(self.friendly_life) +
            ", enemy_life: " + str(self.enemy_life) +
            ", is_end: " + str(self.is_end) +
            ", is_won: " + str(self.is_won) +
            "}")
  def __str__(self):
    return self.to_str()
  def __repr__(self):
    return self.to_str()


class Agent:
  sess = None

  # TF-class objects.
  state_in = None # Prob should rename input to state to be consistent with Bellman.
  output = None
  # tf_target_q = None
  # action = None
  # tf_train = None
  reward_holder = None
  action_holder = None
  gradient_holders = None
  gradients = None
  update_batch = None

  def __init__(self):
    # Policy agent from
    # https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

    # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
    self.state_in= tf.placeholder(shape=[None,INP_SHAPE],dtype=tf.float32)
    hidden = slim.fully_connected(self.state_in,HID_SHAPE,biases_initializer=None,activation_fn=tf.nn.relu)
    self.output = slim.fully_connected(hidden,OUT_SHAPE,activation_fn=tf.nn.softmax,biases_initializer=None)
    # self.action = tf.argmax(self.out,1)

    # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    # to compute the loss, and use it to update the network.
    self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
    self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

    # TODO Understand the formula here a little better.

    # Make a 1-hot vector that corresponds to the action chosen?
    indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
    # Grab the output value that corresponds to that 1-hot vector?
    responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), indexes)

    # Take the log of the value in output corresponds to the action.
    # This forumula allows the network to accept Positive and Negative rewards/advantages.
    # Taking a Log I think will stop divergence.
    # Loss = Log(output_vals)*Advantage
    # https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149#.kbxmx7lfm
    self.loss = -tf.reduce_mean(tf.log(responsible_outputs)*self.reward_holder)

    tvars = tf.trainable_variables()
    self.gradient_holders = []
    for idx,var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
        self.gradient_holders.append(placeholder)

    # Get the gradients that should apply to every variable to minimize loss?
    # So basically the derivative that should be applied to the network.
    self.gradients = tf.gradients(self.loss,tvars)

    # Apply those gradients with the learning rate.
    # The caller of the agent grab .gradients for each action+reward, and then apply all the
    # gradients at once by inputting them into gradient_holders to train.
    # So we apply gradients in batches.
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())


  # You prob want to call argmax(get_output())
  # The output is an array of probability that the ith action should be taken.
  # Example usage:
  #
  # agent_out = Agent.get_output(state)
  # tmp = np.random.choice(agent_out,p=agent_out) # pick i based on probabilities
  # action = np.argmax(tmp == tmp)
  def get_output(self, state):
    return self.sess.run(self.output,feed_dict={self.state_in:[state]})[0]


  # Get the gradients to train network for each (state,action,reward) tuple.
  def get_gradients(self, states, actions, rewards):
    feed_dict={self.state_in:states,
               self.action_holder:actions,
               self.reward_holder:rewards}
    return self.sess.run(self.gradients, feed_dict=feed_dict)


  # Caller can batch gradients and apply a bunch at once.
  # You could just call:
  # train_with_gradients(get_gradients())
  # or you could store sum of a few gradients, and call train once for a batch.
  def train_with_gradients(self, gradients):
    feed_dict = dict(zip(self.gradient_holders, gradients))
    self.sess.run(self.update_batch, feed_dict=feed_dict)


class Bot:
  # Infra

  agent = None
  total_reward = None
  total_reward_p = None
  total_reward_n = None

  # Learning params
  explore = None

  n = None
  v = True # verbose

  battles = []
  current_battle = None
  MAX_FRIENDLY_LIFE = None
  MAX_ENEMY_LIFE = None

  def __init__(self):
    self.agent = Agent()
    self.total_reward = 0
    self.total_reward_p = 0
    self.total_reward_n = 0
    self.n = 0
    self.explore = INITIAL_EXPLORE


  def update_battle(self, stage):
    if self.current_battle is None or self.current_battle[-1].is_end:
      self.current_battle = Battle()
      self.battles.append(self.current_battle)
    self.current_battle.add_stage(stage)

  def get_commands(self, game_state):
    commands = []
    # Skip every second frame.
    # This is because marine attacks take 2 frames, and so it can finish an attack started in a prev frame.
    # Need to make the current order an input into the NN so it can learn to return order-0 (no new order)
    # if already performing a good attack.):
    self.n += 1
    if self.n % FRAMES_PER_ACTION == 1: return []

    stage = Stage(game_state)
    self.update_battle(stage)

    if not self.current_battle.is_end:
      # Figure out what action to take next.
      inp_state = Bot.battle_to_input(self.current_battle)
      if V: print "inp = " + str(inp_state)

      # Take action based on a probability returned from the policy network.
      agent_out = self.agent.get_output(inp_state)
      tmp = np.random.choice(agent_out,p=agent_out) # pick i based on probabilities
      action = np.argmax(agent_out == tmp)
      print "agent_out = " + str(agent_out)
      print "tmp = " + str(tmp)
      print "action = " + str(action)
      stage.inp = inp_state
      stage.action = action
      commands = Bot.output_to_command(action, game_state)

    # TODO train the same number of positive and negative battles.
    # I guess we have to find a positive battle first.

    print ("total_reward = " + str(self.total_reward) +
      ", total_reward_p = " + str(self.total_reward_p) +
      ", total_reward_n = " + str(self.total_reward_n))

    if self.current_battle.is_end:
      # Once the battle is over, train for it wholistically:
      self.train_battle(self.current_battle)

    # Outputs
    return commands

  def train_battle(self, battle):
    print "\nTRAIN BATTLE"

    if battle.trained:
      raise Exception("Battle already trained")
    else:
      battle.trained = True
    if battle.size() == 1:
      print "skipping training empty battle"
      return

    print "battle.size() = " + str(battle.size())
    if not battle[0].friendly_unit or not battle[0].enemy_unit:
      raise Exception("No units in initial battle state.")
    global MAX_FRIENDLY_LIFE
    global MAX_ENEMY_LIFE
    MAX_FRIENDLY_LIFE = battle[0].friendly_unit.get_max_life()
    MAX_ENEMY_LIFE = battle[0].enemy_unit.get_max_life()

    # First calculate rewards.
    rewards = np.zeros(battle.size())
    # for i in range(1, battle.size()):
    #   # Advantage is a percentage of total life difference. 1.0 in a single round = won the game.
    #   rewards[i-1] = 0.1 * Bot.calculate_advantage(battle[i-1], battle[i])

    # Now give gradually discounted rewards to earlier actions.
    if battle.is_won:
      rewards[-1] += 1
    else:
      rewards[-1] += -1

    for i in reversed(xrange(0, rewards.size-1)):
      self.total_reward += rewards[i+1]
      if rewards[i+1] > 0:
        self.total_reward_p += rewards[i+1]
      else:
        self.total_reward_n += rewards[i+1]
      rewards[i] = rewards[i] * GAMMA + rewards[i+1]

    rewards = rewards[:-1] # We drop the input/action/reward for the end state.
    inputs = []
    actions = []
    for i in range(0, battle.size()-1):
      inputs.append(battle[i].inp)
      actions.append(battle[i].action)

    grads = self.agent.get_gradients(inputs, actions, rewards)
    # TODO: Consider only training once ever ~5 rounds, using a grad_buffer
    self.agent.train_with_gradients(grads)

    # for idx,grad in enumerate(grads):
    #     grad_buffer[idx] += grad
    # if i % update_frequency == 0 and i != 0:
    #   agent.train_with_gradients(grads)
    #   for ix,grad in enumerate(gradBuffer):
    #       gradBuffer[ix] = grad * 0


  @staticmethod
  def calculate_advantage(stage_0, stage_1):
    """ What is the delata in advantage in moving from stage_a to stage_b """
    # Improvement in hp difference is good.
    hp_pct_0 = (float(stage_0.friendly_life)/MAX_FRIENDLY_LIFE) - (float(stage_0.enemy_life)/MAX_ENEMY_LIFE)
    hp_pct_1 = (float(stage_1.friendly_life)/MAX_FRIENDLY_LIFE) - (float(stage_1.enemy_life)/MAX_ENEMY_LIFE)
    return hp_pct_1 - hp_pct_0


  @staticmethod
  def output_to_command(action, state):
    """ out_t = [14] """
    """ action in [0 .. 13]"""
    commands = []

    if not state.friendly_units or not state.enemy_units: return commands

    friendly = state.friendly_units.values()[0]
    enemy = state.enemy_units.values()[0]


    # 0 = do nothing
    # 1-8 = move 5 units in dir
    # 9-13 = attack unit num 0-4
    a = action
    if a < 0 or 9 < a:
      raise Exception("Invalid action: " + str(a))

    if a == 0: return commands # 0 means keep doing what you were doing before.
    # Consider simplifying this to just run away from the enemy... So we only have 2 actions.
    elif 1 <= a and a <= 8:
      del_x, del_y = MOVES[a-1]
      move_x = Bot.constrain(friendly.x + del_x, X_MOVE_RANGE)
      move_y = Bot.constrain(friendly.y + del_y, Y_MOVE_RANGE)
      commands.append([friendly.id, tc_client.UNIT_CMD.Move, -1, move_x, move_y])
    elif a == 9:
      commands.append([friendly.id, tc_client.UNIT_CMD.Attack_Unit, enemy.id])
    else:
      raise Exception("Failed to grok action: " + str(a))

    return commands


  @staticmethod
  def battle_to_input(battle):
    if battle.size() == 0:
      raise Exception("No input without at least 1 battle frames")
    if battle.is_end:
      raise Exception("No input from last battle frame")

    # So there should always be both a friendly+enemy unit in the last 2 stages.

    i0 = -2
    i1 = -1
    if battle.size() == 1:
      i0 = -1 # Just use the first frame for both, so there's no movement.

    f0 = battle[i0].friendly_unit
    f1 = battle[i1].friendly_unit
    e0 = battle[i0].enemy_unit
    e1 = battle[i1].enemy_unit
    if f0.id != f1.id or e0.id != e1.id:
      raise Exception("Units in adjoind frames must have the same IDs, we assume one unit.")

    print "f1 = " + str(f1)
    print "e1 = " + str(e1)

    return (Bot.unit_to_vector(f0, True) + Bot.unit_to_vector(f1, True) +
            Bot.unit_to_vector(e0, False) + Bot.unit_to_vector(e1, False))


  @staticmethod
  def unit_to_vector(unit, is_friendly):
    unit_vector = [
            # 1.0 if is_friendly else -1.0,
            Bot.norm(unit.x, (X_MOVE_RANGE)),
            Bot.norm(unit.y, (Y_MOVE_RANGE)),
            float(unit.get_life()) / unit.get_max_life(),
            float(unit.groundCD) / unit.maxCD
            ]

    if is_friendly:
      # Then we add the Order we're currently giving it. Simulate that we can't see the order for enemies.

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
      # unit_vector = unit_vector + [int(is_guard), int(is_move), int(is_attack)]

    return unit_vector


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



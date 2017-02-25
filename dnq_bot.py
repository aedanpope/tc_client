# Policy Functions
# https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.axhe0juti
# http://karpathy.github.io/2016/05/31/rl/
#
# TODO:
# 1. DNQ network first:
# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.1p0qomyrr
#
# 2.Way better exploration ala
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf#.rr1b6cl6e
# I should implement them all as hyperparameters and experiment.
#
# 3. Explore better reward functions, as per "Advanced Approaches" at the bottom of the exploration article above.
#
# 4. AC3!@
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.449tze1hh

#
# DNQ Bot:
# - SKIP: Convolution network.
# - Target Network
# - Experience Replay


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

V_PER_FRAME = False  # Verbose

FRAMES_PER_ACTION = 1

# MOVES = [(6,0), (-6,0), (0,6), (0,-6), (4,4), (4,-4), (-4,4), (-4,-4)]
MOVES = [(6,0), (-6,0), (0,6), (0,-6)]
# MOVES = [(10,0), (-10,0), (0,10), (0,-10), (7,7), (7,-7), (-7,7), (-7,-7)]
# friendly starts at (70,140), enemy starts at (100,140)
X_MOVE_RANGE = (60,120) # X_MOVE_RANGE and Y_MOVE_RANGE should be the same magnitude.
Y_MOVE_RANGE = (110,190)
MAX_FRIENDLY_LIFE = None
MAX_ENEMY_LIFE = None

# Parameterization
FRIENDLY_TENSOR_SIZE = 14
ENEMY_TENSOR_SIZE = 8
MAX_FRIENDLY_UNITS = 1 # 5 for marines
MAX_ENEMY_UNITS = 1 # 5 for marines

# TOPOLOGY
INP_SHAPE = MAX_FRIENDLY_UNITS * FRIENDLY_TENSOR_SIZE + MAX_ENEMY_UNITS * ENEMY_TENSOR_SIZE
HID_SHAPE = 20 # Hidden layer shape.
OUT_SHAPE = 5 + MAX_ENEMY_UNITS

# Learning and env params:

# LEARNING_RATE = 0.01
LEARNING_RATE = 0.001
BUFFER_SIZE = 50000
BATCH_SIZE = 32 #How many experiences to use for each training step.
UPDATE_FREQ = 4 #How often to perform a training step.
FUTURE_Q_DISCOUNT = .99 #Discount factor on future Q-values, discount on expected future reward.
START_E = 1 #Starting chance of random action
END_E = 0.1 #Final chance of random action
ANNEALING_STEPS = 10000 #How many steps of training to reduce startE to endE.
E_STEP = (START_E - END_E)/ANNEALING_STEPS
PRE_TRAIN_STEPS = 10000 #How many steps of random actions before training begins.
TAU = 0.001 # Rate to update target network toward primary network


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


# TF Session
sess = None

## Generally, try and keep everything in the [-1,1] range.
## TODO: consider trying [0,1] range for some stuff e.g. HP.



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

class ExperienceBuffer():

  def __init__(self):
      self.buffer = []

  def append(self, state, action, reward, new_state, done):
    vec = [state, action, reward, new_state, done]
    self.buffer.append(vec)
    if len(self.buffer) >= BUFFER_SIZE:
      # MAybe slow, consider using a boolean mask to change in-place
      self.buffer = self.buffer[1:]
        # numpy.delete(self.buffer, (0), axis=0)

  # def add(self,experience):
  #     if len(self.buffer) + len(experience) >= BUFFER_SIZE:
  #         self.buffer[0:(len(experience)+len(self.buffer))-BUFFER_SIZE] = []
  #     self.buffer.extend(experience)

  def states(self):
    return np.array(self.buffer)[:,0]
  def actions(self):
    return np.array(self.buffer)[:,1]
  def rewards(self):
    return np.array(self.buffer)[:,2]
  def states2(self):
    return np.array(self.buffer)[:,3]
  def dones(self):
    return np.array(self.buffer)[:,4]

  def sample(self,size):
    # return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
    result = ExperienceBuffer()
    result.buffer = random.sample(self.buffer,size)
    return result

def vsstr(vars):
  return ',\n'.join([vstr(v) for v in vars])

def vstr(var):
    return str(var.initial_value)
    # return ("Variable: " + str(var.initial_value))
            # ", initializer: " + str(var.initializer) +
            # ", op: " + str(var.op) +
            # ", friendly_life: " + str(self.friendly_life) +
            # ", enemy_life: " + str(self.enemy_life) +
            # ", is_end: " + str(self.is_end) +
            # ", is_won: " + str(self.is_won) +
            # "}")


class DNQNetwork:

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

  trainable_variables = None
  update_from_super_ops = None

  def __init__(self, myname):
    existing_tvars = tf.trainable_variables()

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

    self.trainable_variables = []
    updated_tvars = tf.trainable_variables()
    # print "updated_tvars = \n" + vsstr(updated_tvars)
    for var in updated_tvars:
      if var not in existing_tvars:
        self.trainable_variables.append(var)
    # for idx,var in enumerate(updated_tvars)
    # print "trainable_variables = \n" + vsstr(self.trainable_variables)

    self.gradient_holders = []
    for idx,var in enumerate(self.trainable_variables):
        placeholder = tf.placeholder(tf.float32,name=myname+'_'+str(idx)+'_holder')
        self.gradient_holders.append(placeholder)

    # Get the gradients that should apply to every variable to minimize loss?
    # So basically the derivative that should be applied to the network.
    self.gradients = tf.gradients(self.loss,self.trainable_variables)

    # Apply those gradients with the learning rate.
    # The caller of the agent grab .gradients for each action+reward, and then apply all the
    # gradients at once by inputting them into gradient_holders to train.
    # So we apply gradients in batches.
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,self.trainable_variables))


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

  def update_from_main_graph(self, main_graph):
    # ASSUME that .trainable_vars as in-step
    if update_from_main_ops is None:
      update_from_main_ops = []
      for i in range(0, len(self.trainable_variables)):
        var = self.trainable_variables[i]
        main_var = main_graph.trainable_variables[i]
        # weighted average with weight tau on main_var.
        # main_var = tau*main_var + (1-tau)*var
        update_from_main_ops.append(
            var.assign((TAU*main_var.value()) + ((1-TAU)*var.value())))
    for op in update_from_main_ops:
        sess.run(op)


class Bot:
  # Infra

  main_network = None
  target_network = None
  total_reward = None
  total_reward_p = None
  total_reward_n = None

  # Learning params
  explore = None
  experience_buffer = None
  total_steps = None

  n = None
  v = True # verbose

  battles = []
  current_battle = None
  MAX_FRIENDLY_LIFE = None
  MAX_ENEMY_LIFE = None
  total_battles = 0
  total_wins = 0
  last_10_results = []

  def __init__(self):
    sess = tf.InteractiveSession()
    self.main_network = DNQNetwork('main')
    self.target_network = DNQNetwork('target')
    sess.run(tf.global_variables_initializer())
    # Init the target network to be equal to the primary network.
    self.target_network.update_from_main_graph(self.main_network)
    self.experience_buffer = ExperienceBuffer()

    self.total_reward = 0
    self.total_reward_p = 0
    self.total_reward_n = 0
    self.n = 0
    self.explore = START_E
    self.total_steps = 0


  def update_battle(self, stage):
    if not self.current_battle:
      self.total_battles += 1
      self.current_battle = Battle()
      self.battles.append(self.current_battle)

    if self.current_battle.is_end:
      if stage.is_end:
        return # Don't accumulate more end states.
      else:
        # Make a new battle.
        self.total_battles += 1
        if self.current_battle.is_won:
          self.total_wins += 1
        self.last_10_results.append(self.current_battle.is_won)
        if (len(self.last_10_results) > 10):
          self.last_10_results = self.last_10_results[-10:]

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
      self.total_steps += 1

      # Figure out what action to take next.
      inp_state = Bot.battle_to_input(self.current_battle)
      if V_PER_FRAME: print "inp = " + str(inp_state)


      # Take action based on a probability returned from the policy network.
      agent_out = self.main_network.get_output(inp_state)
      action = np.argmax(agent_out)

      # TODO implement exploration algorithm here.
      # For now E-Greedy
      if self.total_steps < PRE_TRAIN_STEPS or np.random.rand(1) < self.explore:
        action = np.random.randint(0,OUT_SHAPE)
      if PRE_TRAIN_STEPS < self.total_steps and self.explore > END_E:
        self.explore -= E_STEP

      # Boltzmann etc. TODO
      # tmp = np.random.choice(agent_out,p=agent_out) # pick i based on probabilities
      # action = np.argmax(agent_out == tmp)

      if V_PER_FRAME: print "agent_out = " + str(agent_out)
      # if V_PER_FRAME: print "tmp = " + str(tmp)
      if V_PER_FRAME: print "action = " + str(action)
      stage.inp = inp_state
      stage.action = action
      commands = Bot.output_to_command(action, game_state)

    # TODO train the same number of positive and negative battles.
    # I guess we have to find a positive battle first.

    if V_PER_FRAME: print ("total_reward = " + str(self.total_reward) +
                           ", total_reward_p = " + str(self.total_reward_p) +
                           ", total_reward_n = " + str(self.total_reward_n))

    if self.current_battle.is_end and not self.current_battle.trained:
      # Calculate rewards, and add experiences to buffer.
      self.add_battle_to_buffer(self.current_battle, self.experience_buffer)

      print "battle.size() = " + str(self.current_battle.size())
      print "total_battles = " + str(self.total_battles)
      print "total_wins = " + str(self.total_wins)
      print "win_ratio = " + str(self.total_wins / float(self.total_battles))
      print "last_10 = " + str(sum(self.last_10_results)) + " / " + str(len(self.last_10_results))


    if self.total_steps > PRE_TRAIN_STEPS and self.total_steps % (UPDATE_FREQ) == 0:
      self.train()
    # Outputs
    return commands

  def train(self):
    # Train the Main and Target networks.
    train_batch = self.experience_buffer.sample(BATCH_SIZE)
    # Get actions from Main, but Q-values for those actions from Target
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.odnj51rop
    # Q[s,a] = Q[s,a] + lr*(r + FUTURE_Q_DISCOUNT*np.max(Q[s1,:]) - Q[s,a])
    # Q-Target = r + FUTURE_Q_DISCOUNT*Q(s’,argmax(Q(s',a,\theta),'\theta')

    actions_out = sess.run(main_network.predict,feed_dict={main_network.scalarInput:np.vstack(train_batch.states2())})
    q2_out = sess.run(target_network.Qout,feed_dict={target_network.scalarInput:np.vstack(train_batch.states2())})

    end_multiplier = 1 - train_batch.dones() # Set the predicted future reward to 0 if it's the end state.
    doubleQ = q_out[range(BATCH_SIZE),actions_out]
    targetQ = train_batch.rewards() + (FUTURE_Q_DISCOUNT*doubleQ * end_multiplier)


    # Train the network with our target values.
    sess.run(main_network.updateModel,
        feed_dict={main_network.scalarInput:np.vstack(train_batch.states()),
                   main_network.targetQ:targetQ,
                   main_network.actions:train_batch.actions()})
    # def states(self):
    #   return np.array(self.buffer)[:,0]
    # def actions(self):
    #   return np.array(self.buffer)[:,1]
    # def rewards(self):
    #   return np.array(self.buffer)[:,2]
    # def states2(self):
    #   return np.array(self.buffer)[:,3]
    # def dones(self):
    #   return np.array(self.buffer)[:,4]

    # Set the target network to be equal to the primary network, with factor TAU = 0.001
    self.target_network.update_from_main_graph(self.main_network)


  def add_battle_to_buffer(self, battle, buffer):
    print "\nTRAIN BATTLE"

    if battle.trained:
      raise Exception("Battle already trained")
    else:
      battle.trained = True
    if battle.size() == 1:
      print "skipping training empty battle"
      return
    if not battle[0].friendly_unit or not battle[0].enemy_unit:
      raise Exception("No units in initial battle state.")

    global MAX_FRIENDLY_LIFE
    global MAX_ENEMY_LIFE
    MAX_FRIENDLY_LIFE = battle[0].friendly_unit.get_max_life()
    MAX_ENEMY_LIFE = battle[0].enemy_unit.get_max_life()

    # First calculate rewards.
    rewards = np.zeros(battle.size())

    # Final reward = 1 for winning, -0.5 to 0.5 for doing some damage.
    # if battle.is_won:
    #   rewards[-1] = 1
    # else:
    #   rewards[-1] += 0.5 + Bot.calculate_advantage(battle[0], battle[-1])
    # PARTIAL REWARDS teach it not to take risks, so make the game easier and give complete rewards.
    if battle.is_won:
      rewards[-1] += 1
    else:
      rewards[-1] += -1

    for i in range(0, battle.size()):
      self.total_reward += rewards [i]
      if rewards[i] > 0:
        self.total_reward_p += rewards[i]
      else:
        self.total_reward_n += rewards[i]

    # Now give gradually discounted rewards to earlier actions.
    for i in reversed(xrange(0, rewards.size-1)):
      rewards[i] = rewards[i] * GAMMA + rewards[i+1]

    for i in range(0, battle.size()-1):
      # def append(self, state, action, reward, new_state, done):
      buffer.append(battle[i].inp, battle[i].action, rewards[i], battle[i+1].inp,i == battle.size()-1)


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
    if a < 0 or len(MOVES)+1 < a:
      raise Exception("Invalid action: " + str(a))

    if a == 0: return commands # 0 means keep doing what you were doing before.
    # Consider simplifying this to just run away from the enemy... So we only have 2 actions.
    elif 1 <= a and a <= len(MOVES):
      del_x, del_y = MOVES[a-1]
      move_x = Bot.constrain(friendly.x + del_x, X_MOVE_RANGE)
      move_y = Bot.constrain(friendly.y + del_y, Y_MOVE_RANGE)
      commands.append([friendly.id, tc_client.UNIT_CMD.Move, -1, move_x, move_y])
    elif a == len(MOVES)+1:
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

    if V_PER_FRAME: print "f1 = " + str(f1)
    if V_PER_FRAME: print "e1 = " + str(e1)

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


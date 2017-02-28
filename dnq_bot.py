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
from map import Map
import argparse
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import itertools
import math
from agent import Mode

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
EXTRA = 1
INP_SHAPE = EXTRA + MAX_FRIENDLY_UNITS * FRIENDLY_TENSOR_SIZE + MAX_ENEMY_UNITS * ENEMY_TENSOR_SIZE
OUT_SHAPE = 5 + MAX_ENEMY_UNITS


Act = Map(
  Greedy = 0, # highest Q value
  Boltzmann = 1, # infer that softmax(q_values) are probabilities.
  Boltzmann_B = 2, # infer that softmax(q_values) are probabilities.
  Boltzmann_C = 3, # infer that softmax(q_values) are probabilities.
)

# For harder learning, increase these params:




# Hyper Parameters
# TODO move params into this as they need to be set for experiments.
HP = Map(
ACTION_STRATEGY = Act.Boltzmann,

# Configurable topology
HID_1_SHAPE = 50,
HID_2_SHAPE = 30,

# Learning and env params:
LEARNING_RATE = 0.01,
TAU = 0.001, # Rate to update target network toward primary network
BUFFER_SIZE = 50000,
FUTURE_Q_DISCOUNT = .99, #Discount factor on future Q-values, discount on expected future reward.
START_E = 1, #Starting chance of random action
END_E = 0.05, #0.1 #Final chance of random action

BATCH_SIZE = 100, # 32 #How many experiences to use for each training step.
UPDATE_FREQ = 4, # 4 #How often to perform a training step.

# PRE_TRAIN_STEPS needs to be to be more than BATCH_SIZE
# PRE_TRAIN_STEPS requires a lot so we have at least a few wins once we start learning.
PRE_TRAIN_STEPS = 100, # 10000#How many steps of random actions before training begins.
ANNEALING_STEPS = 20000, # 10000#How many steps of training to reduce startE to endE.
)

E_STEP = (HP.START_E - HP.END_E)/HP.ANNEALING_STEPS


# TF Session
SESS = None
SETTINGS = Map({})
TF_WRITER = None

## Generally, try and keep everything in the [-1,1] range.
## TODO: consider trying [0,1] range for some stuff e.g. HP.

def verbose(v=10):
  return V_PER_FRAME or SETTINGS.verbosity >= v


def parse_hyperparameter_sets(hyperparameter_sets_string):
  return eval("Map("+hyperparameter_sets_string+")")


def process_hyperparameters(hyperparameters):
  if not hyperparameters: return
  for (param,val) in hyperparameters.items():
    if not param.isupper():
      raise Exception("All hyperparameters must be UPPER_CASE, bad hyperparameters: " + str(hyperparameters))
    if not param in HP:
      raise Exception("hyperparameters contains param not in HP: " + str(hyperparameters))
    HP[param] = val
    print "set hyperparameter " + param + " to " + str(val)
    if not HP[param]:
      raise Exception("param " + param + " was set to none. Check for typos in flag value of hyperparameters")



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
    self.is_end = state.battle_ended
    self.is_won = state.battle_won

    # self.is_end = self.friendly_life == 0 or self.enemy_life == 0
    # if self.is_end:
    #   self.is_won = self.friendly_life > 0
    # self.reward = 0

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

  buffer = None
  win_buffer = None
  lose_buffer = None

  def __init__(self):
    self.buffer = []
    self.win_buffer = []
    self.lose_buffer = []

  def append(self, state, action, reward, new_state, done, is_won):
    vec = [state, action, reward, new_state, done]
    if None in vec:
      raise Exception("Can't have a none in experience " + str(vec))

    if is_won:
      buf = self.win_buffer
    else:
      buf = self.lose_buffer

    buf.append(vec)
    # MAybe slow, consider using a boolean mask to change in-place
    # numpy.delete(self.buffer, (0), axis=0)
    if len(buf) >= HP.BUFFER_SIZE:
      buf.pop(0)


  def states(self):
    return np.array(self.buffer)[:,0]
  def actions(self):
    return np.array(self.buffer)[:,1]
  def rewards(self):
    return np.array(self.buffer)[:,2]
  def states2(self):
    return np.array(np.array(self.buffer)[:,3])
  def dones(self):
    return np.array(self.buffer)[:,4]

  def sample(self, size):
    # return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
    wins = random.sample(self.win_buffer, min(size/2, len(self.win_buffer)))
    losses = random.sample(self.lose_buffer , min(size/2, len(self.lose_buffer)))
    result = ExperienceBuffer()
    result.buffer = wins + losses
    random.shuffle(result.buffer)
    return result

def vsstr(vars):
  return ',\n'.join([vstr(v) for v in vars])

def vstr(var):
    return str(var.initial_value)


UID = 0
def get_uid():
  global UID
  UID += 1
  return UID

class DNQNetwork:
  state_in = None
  q_out = None
  action_out = None

  target_q_holder = None
  action_holder = None
  action_holder = None

  update_model = None

  trainable_variables = None
  update_from_main_ops = None

  boltzmann_denom = None
  boltzmann_out = None

  summaries = None
  step = 0
  uid = None
  myname = None

  def __init__(self, myname):
    self.uid = get_uid()
    self.myname = myname

    existing_tvars = tf.trainable_variables()

    # Policy agent from
    # https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

    # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
    # self.state_in= tf.placeholder(shape=[INP_SHAPE],dtype=tf.float32, name=("state_in_" + myname))

    self.state_in= tf.placeholder(shape=[None,INP_SHAPE],dtype=tf.float32, name=self.name_var("state_in"))

    hid_1 = slim.fully_connected(self.state_in,HP.HID_1_SHAPE,
                                  biases_initializer=None,activation_fn=tf.nn.relu, scope=self.name_var("hidden1"))
    hid_2 = slim.fully_connected(hid_1,HP.HID_2_SHAPE,
                                  biases_initializer=None,activation_fn=tf.nn.relu, scope=self.name_var("hidden2"))
    # TODO: split into separate advantage & action streams.

    self.q_out = slim.fully_connected(hid_2,OUT_SHAPE,
                                      activation_fn=tf.nn.tanh,biases_initializer=None,
                                      # activation_fn=tf.nn.softmax,biases_initializer=None,
                                      scope=self.name_var("q_out"))
    self.action_out = tf.argmax(self.q_out,1, name=self.name_var("action_out"))

    # self.boltzmann_denom = tf.placeholder(shape=None,dtype=tf.float32)
    self.boltzmann_denom = tf.placeholder(dtype=tf.float32)
    self.boltzmann_out = tf.nn.softmax(self.q_out/self.boltzmann_denom)


    # TODO: Take the log of the output values, to allow Positive and Negative rewards/advantages.
    # Taking a Log I think will stop divergence.
    # https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149#.kbxmx7lfm

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    self.target_q_holder = tf.placeholder(shape=[None],dtype=tf.float32, name=self.name_var("target_q_holder"))
    self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name=self.name_var("action_holder"))
    actions_onehot = tf.one_hot(self.action_holder,OUT_SHAPE,dtype=tf.float32, name=self.name_var("actions_onehot"))

    # Get the Q values from the network for the specified actions in action_holder.
    relevant_q = tf.reduce_sum(tf.mul(self.q_out, actions_onehot), reduction_indices=1)

    td_error = tf.square(self.target_q_holder - relevant_q)
    tf.summary.histogram(self.name_var('td_error'), td_error)
    loss = tf.reduce_mean(td_error, name=self.name_var("loss"))
    tf.summary.scalar(self.name_var('loss'), loss)

    trainer = tf.train.AdamOptimizer(learning_rate=HP.LEARNING_RATE, name=self.name_var("trainer"))
    self.update_model = trainer.minimize(loss)

    self.summaries = tf.summary.merge_all()

    # Get trainable vars so we can update Target network from Main network.
    self.trainable_variables = []
    updated_tvars = tf.trainable_variables()
    for var in updated_tvars:
      if var not in existing_tvars:
        self.trainable_variables.append(var)


  def name_var(self, prefix):
    return prefix + "_" + self.myname + "_" + str(self.uid)


  # You prob want to call argmax(get_output())
  # The output is an array of probability that the ith action should be taken.
  # Example usage:
  #
  # agent_out = Agent.get_output(state)
  # tmp = np.random.choice(agent_out,p=agent_out) # pick i based on probabilities
  # action = np.argmax(tmp == tmp)
  def get_q_out(self, states):
    return SESS.run(self.q_out,feed_dict={self.state_in:states})


  def get_actions(self, states):
    return SESS.run(self.action_out,feed_dict={self.state_in:states})


  def get_boltzmann_action(self, state):
    t = 0.5
    q_probs = SESS.run(self.boltzmann_out,feed_dict={self.state_in:[state],self.boltzmann_denom:t})
    action_value = np.random.choice(q_probs[0],p=q_probs[0])
    action = np.argmax(q_probs[0] == action_value)
    return action


  @staticmethod
  def train_batch(main_network, target_network, train_batch):
    # Get actions from Main, but Q-values for those actions from Target
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.odnj51rop
    # Q[s,a] = Q[s,a] + lr*(r + FUTURE_Q_DISCOUNT*np.max(Q[s1,:]) - Q[s,a])
    # Q-Target = r + FUTURE_Q_DISCOUNT*Q(s',argmax(Q(s',a,\theta),'\theta')

    # Get stuff for Bellman, i.e. what is the Q-value for the best move in the subsequent state?
    actions_2 = SESS.run(main_network.action_out,feed_dict={main_network.state_in:np.vstack(train_batch.states2())})
    q_2 = SESS.run(target_network.q_out,feed_dict={target_network.state_in:np.vstack(train_batch.states2())})

    end_multiplier = 1 - train_batch.dones() # Set the predicted future reward to 0 if it's the end state.
    # Chose the Q value from target_network for each action taken by main_network
    q_2_for_actions = q_2[range(len(train_batch.buffer)),actions_2]

    # RHS of Bellman equation: Q(s, a) = r + gamma * max(Q(s1, a1))
    # end_multiplier sets Q(s1, a1) = 0 when there's no future (and then reward is not 0)
    target_q = train_batch.rewards() + (HP.FUTURE_Q_DISCOUNT * q_2_for_actions * end_multiplier)

    if verbose():
      print ""
      print "TRAIN BATCH"
      if verbose(30): print "states_2 = " + str(train_batch.states2())
      if verbose(30):print "q_2 = " + str(q_2)
      if verbose(30): print "states = " + str(train_batch.states())
      print "q_2_for_actions = " + str(q_2_for_actions)
      print "end_multiplier = " + str(end_multiplier)
      print "actions = " + str(train_batch.actions())
      print "rewards = " + str(train_batch.rewards())
      print "actions_2 = " + str(actions_2)
      print "target_q = " + str(target_q)

    # Train the network with Q values from bellman equation, i.e. Assign RHS to LHS
    summary, _ = SESS.run([main_network.summaries, main_network.update_model],
        feed_dict={main_network.state_in:np.vstack(train_batch.states()),
                   main_network.target_q_holder:target_q,
                   main_network.action_holder:train_batch.actions()})

    main_network.step += 1
    # if (STEP % 10 == 0):
    TF_WRITER.add_summary(summary, main_network.step)

  def update_from_main_graph(self, main_graph):
    # ASSUME that .trainable_vars as in-step
    if self.update_from_main_ops is None:
      self.update_from_main_ops = []
      for i in range(0, len(self.trainable_variables)):
        var = self.trainable_variables[i]
        main_var = main_graph.trainable_variables[i]
        # weighted average with weight tau on main_var.
        # main_var = tau*main_var + (1-tau)*var
        self.update_from_main_ops.append(
            var.assign((HP.TAU*main_var.value()) + ((1-HP.TAU)*var.value())))
    for op in self.update_from_main_ops:
      SESS.run(op)


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

  trained = False

  def __init__(self, hyperparameters=None):
    # Set hyperparams in the global vars in this file.
    # TODO: make these instance vars so we can run multiple bots simultaneously.
    process_hyperparameters(hyperparameters)

    global SESS, TF_WRITER
    tf.reset_default_graph()
    SESS = tf.InteractiveSession()

    # Can only be one Bot instance at one time cause TF session graphs.
    # TODO: put some token in the node names so multiple bots can have their own graphs
    # in a single TF session.

    self.main_network = DNQNetwork('main')
    self.target_network = DNQNetwork('target')
    TF_WRITER = tf.summary.FileWriter('/tmp/tfgraph', SESS.graph)
    SESS.run(tf.global_variables_initializer())
    # Init the target network to be equal to the primary network.
    self.target_network.update_from_main_graph(self.main_network)
    self.experience_buffer = ExperienceBuffer()

    self.total_reward = 0
    self.total_reward_p = 0
    self.total_reward_n = 0
    self.n = 0
    self.explore = HP.START_E
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


  def get_commands(self, game_state, settings):
    global SETTINGS
    SETTINGS = settings
    commands = []

    # Skip every second frame.
    # This is because marine attacks take 2 frames, and so it can finish an attack started in a prev frame.
    # Need to make the current order an input into the NN so it can learn to return order-0 (no new order)
    # if already performing a good attack.):
    self.n += 1
    if self.n % FRAMES_PER_ACTION == 1: return []

    stage = Stage(game_state)
    self.update_battle(stage)
    stage.inp = Bot.battle_to_input(self.current_battle)

    if not self.current_battle.is_end:
      self.total_steps += 1

      # Figure out what action to take next.
      if verbose(): print "inp = " + str(stage.inp)

      agent_q = self.main_network.get_q_out([stage.inp])[0]
      # Best action, we'll use this if we aren't training/exploring.
      action = np.argmax(agent_q)
      if verbose(): print "agent_q = " + str(agent_q)
      if verbose(): print "best_action = " + str(action)

      if SETTINGS.mode == Mode.train:
        if self.total_steps <= HP.PRE_TRAIN_STEPS:
          # Still pre-training, always random actions.
          action = np.random.randint(0, OUT_SHAPE)

        elif Act.Boltzmann <= HP.ACTION_STRATEGY and HP.ACTION_STRATEGY <= Act.Boltzmann_C:
          # Always use boltzman action.
          if HP.ACTION_STRATEGY == Act.Boltzmann:
            action = self.main_network.get_boltzmann_action(stage.inp)

          # when explore, use boltzman action.
          elif HP.ACTION_STRATEGY == Act.Boltzmann_B:
            if np.random.rand(1) < self.explore:
              action = self.main_network.get_boltzmann_action(stage.inp)

          # when explore, use random action - otherwise use boltzman action.
          elif HP.ACTION_STRATEGY == Act.Boltzmann_C:
            if np.random.rand(1) < self.explore:
              action = np.random.randint(0, OUT_SHAPE)
            else
              action = self.main_network.get_boltzmann_action(stage.inp)

        elif HP.ACTION_STRATEGY == Act.Greedy:
          if np.random.rand(1) < self.explore:
            if verbose(): print "Explore!"
            action = np.random.randint(0, OUT_SHAPE)
          else:
            if verbose(): print "Dont Explore."

        if self.total_steps >  HP.PRE_TRAIN_STEPS and self.explore > HP.END_E:
          self.explore -= E_STEP

        else:
          raise Exception("Unknown action strategy")

      if verbose(): print "chosen_action = " + str(action)

      stage.action = action
      commands += Bot.output_to_command(action, game_state)


    if self.current_battle.is_end and not self.current_battle.trained:
      # Calculate rewards, and add experiences to buffer.
      self.add_battle_to_buffer(self.current_battle, self.experience_buffer)


      print "total_steps = " + str(self.total_steps) + ", PRE_TRAIN_STEPS = " + str(HP.PRE_TRAIN_STEPS)
      print "explore = " + str(self.explore)
      print ("win_buffer = " + str(len(self.experience_buffer.win_buffer)) +
             ", lose_buffer = " + str(len(self.experience_buffer.lose_buffer)))
      print ("total_reward = " + str(self.total_reward) +
             ", total_reward_p = " + str(self.total_reward_p) +
             ", total_reward_n = " + str(self.total_reward_n))
      print "battle.size() = " + str(self.current_battle.size())
      print "total_battles = " + str(self.total_battles)
      print "total_wins = " + str(self.total_wins)
      print "win_ratio = " + str(self.total_wins / float(self.total_battles))
      print "last_10 = " + str(sum(self.last_10_results)) + " / " + str(len(self.last_10_results))


    # Only train when not at the end of the battle, so training time is fairly proportioned over time spent fighting.
    # So bots that end battles slowly don't see less battles because of it.
    # See similar logic in exercise.py
    if (not self.current_battle.is_end and
        self.total_steps > HP.PRE_TRAIN_STEPS and
        self.total_steps % (HP.UPDATE_FREQ) == 0):
      # print "train WTF"
      DNQNetwork.train_batch(self.main_network, self.target_network, self.experience_buffer.sample(HP.BATCH_SIZE))
      # Set the target network to be equal to the primary network, with factor TAU = 0.001
      self.target_network.update_from_main_graph(self.main_network)

    # Outputs
    return commands


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
      reward = 1
    else:
      reward = -1

    rewards[-2] = reward

    self.total_reward += reward
    if reward > 0:
      self.total_reward_p += reward
    else:
      self.total_reward_n += reward

    # NOTE: We discount future rewards at training time, instead of here.

    for i in range(0, battle.size()-1):
      # def append(self, state, action, reward, new_state, done):
      buffer.append(battle[i].inp, battle[i].action, rewards[i], battle[i+1].inp,i == battle.size()-2, battle.is_won)


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
      return np.zeros(INP_SHAPE) # This state should never be used for training, because the prev action has done=true.
      # raise Exception("No input from last battle frame")

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

    return ([Bot.norm(battle.size(), ([0,64]))] +
            Bot.unit_to_vector(f0, True) + Bot.unit_to_vector(f1, True) +
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



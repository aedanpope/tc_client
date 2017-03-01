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
import agent
from experience import ExperienceBuffer
from agent import Battle
from agent import Stage
from my_logging import log
from my_logging import should_log

Mode = Map(
    train = 0, # Explore and train the network.
    test = 1, # Test the current optimal performance of the network.
)

class Settings:
  mode = Mode.train
  hyperparameters = None


FRAMES_PER_ACTION = 1

MAX_FRIENDLY_LIFE = None
MAX_ENEMY_LIFE = None


Act = Map(
  # highest Q value, else randomly explore.
  Greedy = 0,

  # Boltzmann: interpret softmax(q_values/0.5) as relative probabilities that we should take each action.
  Boltzmann = 1, # Always use boltzmann action.
  Boltzmann_B = 2, # Use boltzmann action when exploring.
  Boltzmann_C = 3, # Use random action when exploring, otherwise use boltzmann action.
)

# For harder learning, increase these params:




# Hyper Parameters
# TODO move params into this as they need to be set for experiments.
HP = Map(
ACTION_STRATEGY = Act.Boltzmann,

# Configurable topology
HID_1_SHAPE = 300,
HID_2_SHAPE = 200,

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
PRE_TRAIN_STEPS = 10000, # 10000#How many steps of random actions before training begins.
ANNEALING_STEPS = 50000, # 10000#How many steps of training to reduce startE to endE.
)
# Derived hyperparameters
def E_STEP():
  return (HP.START_E - HP.END_E)/HP.ANNEALING_STEPS


# TF Session
SESS = None
SETTINGS = Map({})
TF_WRITER = None

## Generally, try and keep everything in the [-1,1] range.
## TODO: consider trying [0,1] range for some stuff e.g. HP.


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
    if HP[param] is None:
      raise Exception("param " + param + " was set to none. Check for typos in flag value of hyperparameters")


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

    self.state_in= tf.placeholder(shape=[None,agent.INP_SHAPE],dtype=tf.float32, name=self.name_var("state_in"))

    hid_1 = slim.fully_connected(self.state_in,HP.HID_1_SHAPE,
                                  biases_initializer=None,activation_fn=tf.nn.relu, scope=self.name_var("hidden1"))
    hid_2 = slim.fully_connected(hid_1,HP.HID_2_SHAPE,
                                  biases_initializer=None,activation_fn=tf.nn.relu, scope=self.name_var("hidden2"))
    # TODO: split into separate advantage & action streams.

    self.q_out = slim.fully_connected(hid_2,agent.OUT_SHAPE,
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
    actions_onehot = tf.one_hot(self.action_holder,agent.OUT_SHAPE,dtype=tf.float32, name=self.name_var("actions_onehot"))

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
    """
      train_batch is an experience.ExperienceTable
    """
    # Get actions from Main, but Q-values for those actions from Target
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.odnj51rop
    # Q[s,a] = Q[s,a] + lr*(r + FUTURE_Q_DISCOUNT*np.max(Q[s1,:]) - Q[s,a])
    # Q-Target = r + FUTURE_Q_DISCOUNT*Q(s',argmax(Q(s',a,\theta),'\theta')

    # Get stuff for Bellman, i.e. what is the Q-value for the best move in the subsequent state?
    actions_2 = SESS.run(main_network.action_out,feed_dict={main_network.state_in:np.vstack(train_batch.states2())})
    q_2 = SESS.run(target_network.q_out,feed_dict={target_network.state_in:np.vstack(train_batch.states2())})

    end_multiplier = 1 - train_batch.dones() # Set the predicted future reward to 0 if it's the end state.
    # Chose the Q value from target_network for each action taken by main_network
    q_2_for_actions = q_2[range(train_batch.len()),actions_2]

    # RHS of Bellman equation: Q(s, a) = r + gamma * max(Q(s1, a1))
    # end_multiplier sets Q(s1, a1) = 0 when there's no future (and then reward is not 0)
    target_q = train_batch.rewards() + (HP.FUTURE_Q_DISCOUNT * q_2_for_actions * end_multiplier)

    if should_log():
      print ""
      print "TRAIN BATCH"
      if should_log(30): print "states_2 = " + str(train_batch.states2())
      if should_log(30):print "q_2 = " + str(q_2)
      if should_log(30): print "states = " + str(train_batch.states())
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

  frame = None
  war = None

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
    self.frame = 0
    self.explore = HP.START_E
    self.total_steps = 0
    self.war = agent.War()


  def get_commands(self, game_state, settings):
    global SETTINGS
    SETTINGS = settings
    commands = []

    self.frame += 1
    # Maybe want to skip frames or something, we don't need this anymore since prev state is an input
    # into Network so it can learn velocity & not to cancel current order.


    stage = Stage(game_state)
    self.war.update_current_battle(stage)
    stage.inp = agent.battle_to_input(self.war.current_battle)

    if not self.war.current_battle.is_end:
      self.total_steps += 1

      # Figure out what action to take next.
      log("inp = " + str(stage.inp))

      agent_q = self.main_network.get_q_out([stage.inp])[0]
      # Best action, we'll use this if we aren't training/exploring.
      action = np.argmax(agent_q)
      log("agent_q = " + str(agent_q))
      log("best_action = " + str(action))

      if SETTINGS.mode == Mode.train:
        if self.total_steps <= HP.PRE_TRAIN_STEPS:
          # Still pre-training, always random actions.
          action = np.random.randint(0, agent.OUT_SHAPE)

        # Always use boltzman action.
        elif HP.ACTION_STRATEGY == Act.Boltzmann:
            action = self.main_network.get_boltzmann_action(stage.inp)

        # When explore, use boltzman action.
        elif HP.ACTION_STRATEGY == Act.Boltzmann_B:
          if np.random.rand(1) < self.explore:
            action = self.main_network.get_boltzmann_action(stage.inp)

        # When explore, use random action - otherwise use boltzman action.
        elif HP.ACTION_STRATEGY == Act.Boltzmann_C:
          if np.random.rand(1) < self.explore:
            action = np.random.randint(0, agent.OUT_SHAPE)
          else:
            action = self.main_network.get_boltzmann_action(stage.inp)

        # When explore, use random action.
        elif HP.ACTION_STRATEGY == Act.Greedy:
          if np.random.rand(1) < self.explore:
            log("Explore!")
            action = np.random.randint(0, agent.OUT_SHAPE)
          else:
            log("Dont Explore.")

        else:
          raise Exception("Unknown action strategy")

        if self.total_steps >  HP.PRE_TRAIN_STEPS and self.explore > HP.END_E:
          self.explore -= E_STEP()


      log("chosen_action = " + str(action))

      stage.action = action
      commands += agent.output_to_command(action, game_state)


    if self.war.current_battle.is_end and not self.war.current_battle.trained:
      # Calculate rewards, and add experiences to buffer.
      print "Store Battle"
      self.war.current_battle.trained = True

      agent.write_battle_to_experience_buffer(self.war.current_battle, self.experience_buffer)

      reward = 1 if self.war.current_battle else -1
      self.total_reward += reward
      if reward > 0:
        self.total_reward_p += reward
      else:
        self.total_reward_n += reward

      print "total_steps = " + str(self.total_steps) + ", PRE_TRAIN_STEPS = " + str(HP.PRE_TRAIN_STEPS)
      print "explore = " + str(self.explore)
      print ("win_buffer = " + str(len(self.experience_buffer.win_buffer)) +
             ", lose_buffer = " + str(len(self.experience_buffer.lose_buffer)))
      print ("total_reward = " + str(self.total_reward) +
             ", total_reward_p = " + str(self.total_reward_p) +
             ", total_reward_n = " + str(self.total_reward_n))
      self.war.print_summary()


    # Only train when not at the end of the battle, so training time is fairly proportioned over time spent fighting.
    # So bots that end battles slowly don't see less battles because of it.
    # See similar logic in exercise.py
    if (not self.war.current_battle.is_end and
        self.total_steps > HP.PRE_TRAIN_STEPS and
        self.total_steps % (HP.UPDATE_FREQ) == 0):
      # print "train WTF"
      DNQNetwork.train_batch(self.main_network, self.target_network, self.experience_buffer.sample(HP.BATCH_SIZE))
      # Set the target network to be equal to the primary network, with factor TAU = 0.001
      self.target_network.update_from_main_graph(self.main_network)

    # Outputs
    return commands




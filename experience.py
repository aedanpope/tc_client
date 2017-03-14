
import experience_pb2
import sys
import random
import agent
import numpy as np


class ExperienceTable:

  # Nx5 array
  table = None

  def __init__(self, table=None):
    self.table = table

  def states(self):
    return np.array(self.table)[:,0]
  def actions(self):
    return np.array(self.table)[:,1]
  def rewards(self):
    return np.array(self.table)[:,2]
  def states2(self):
    return np.array(np.array(self.table)[:,3])
  def dones(self):
    return np.array(self.table)[:,4]
  def len(self):
    return len(self.table)


class ExperienceBuffer():

  separate_buffers = None
  buffer = None
  win_buffer = None
  lose_buffer = None
  buffer_size = None

  def __init__(self, buffer_size=sys.maxint, init_file_path=None, separate_buffers=True):
    """
    Args:
      init_file_path: If specified, initialize the experience buffer with the data in this file.
    """
    self.separate_buffers = separate_buffers
    if separate_buffers:
      self.win_buffer = []
      self.lose_buffer = []
    else:
      self.buffer = []

    self.buffer_size = buffer_size

    if init_file_path:
      experience_list = experience_pb2.ExperienceList()
      f = open(init_file_path, "rb")
      experience_list.ParseFromString(f.read())
      f.close()
      for exp_proto in experience_list.experience:
        self.append(exp_proto.state,
                    exp_proto.action,
                    exp_proto.reward,
                    exp_proto.new_state,
                    exp_proto.done,
                    exp_proto.is_won)


  def append(self, state, action, reward, new_state, done, is_won):
    row = [state, action, reward, new_state, done]
    if None in row:
      raise Exception("Can't have a none in experience " + str(vec))

    if self.separate_buffers:
      if is_won:
        buf = self.win_buffer
      else:
        buf = self.lose_buffer
    else:
      buf = self.buffer

    buf.append(row)
    # Maybe slow, consider using a boolean mask to change in-place
    # numpy.delete(self.buffer, (0), axis=0)
    if len(buf) >= self.buffer_size:
      buf.pop(0)


  def enough_to_sample(self, size):
    if self.separate_buffers:
      return size/2 < len(self.win_buffer) and size/2 < len(self.lose_buffer)
    else:
      return size < len(self.buffer)


  def sample(self, size):
    result = ExperienceTable()

    if self.separate_buffers:
      wins = random.sample(self.win_buffer, min(size/2, len(self.win_buffer)))
      losses = random.sample(self.lose_buffer , min(size/2, len(self.lose_buffer)))
      result.table = wins + losses
    else:
      result.table = random.sample(self.buffer, size)

    random.shuffle(result.table)
    return result


  def all(self):
    if self.separate_buffers:
      return ExperienceTable(self.win_buffer + self.lose_buffer)
    else:
      return ExperienceTable(self.buffer)


def add_experience_to_list(experience_list, row, is_won):
  exp = experience_list.experience.add()
  exp.state.extend(row[0])
  exp.action = row[1]
  exp.reward = row[2]
  exp.new_state.extend(row[3])
  exp.done = row[4]
  exp.is_won = is_won


def add_experiences_to_list(experience_list, table, is_won):
  for row in table:
    add_experience_to_list(experience_list, row, is_won)


# Usage:
# p exercise.py -s 1 -k 2 -t 1 -hp "foo={'PRE_TRAIN_STEPS':1000, 'ANNEALING_STEPS':0}" --test_battles=0  --record=data/1000steps.pb
class ExperienceRecordingBot:
  war = None
  experience_buffer = None
  file_path = None
  def __init__(self, file_path):
    self.war = agent.War()
    self.experience_buffer = ExperienceBuffer()
    self.file_path = file_path


  def get_commands(self, game_state, settings):
    commands = []
    stage = agent.Stage(game_state)
    self.war.update_current_battle(stage)
    stage.inp = agent.battle_to_input(self.war.current_battle)

    if not self.war.current_battle.is_end:
      action = np.random.randint(0, agent.OUT_SHAPE)
      stage.action = action
      commands += agent.output_to_command(action, game_state)

    if self.war.current_battle.is_end and not self.war.current_battle.trained:
      self.war.current_battle.trained = True
      agent.write_battle_to_experience_buffer(self.war.current_battle, self.experience_buffer)
      print ("win_buffer = " + str(len(self.experience_buffer.win_buffer)) +
             ", lose_buffer = " + str(len(self.experience_buffer.lose_buffer)))
      self.war.print_summary()

    return commands


  def close(self):
    experience_list = experience_pb2.ExperienceList()
    add_experiences_to_list(experience_list, self.experience_buffer.win_buffer, True)
    add_experiences_to_list(experience_list, self.experience_buffer.lose_buffer, False)
    f = open(self.file_path, 'wb')
    f.write(experience_list.SerializeToString())
    f.close()

    pass
    # TODO write experiences to file.


class ExperienceIO:
  experience_list = None


  def __init__(self):
    self.experience_list = experience_pb2.ExperienceList()


  def append(self, state, action, reward, new_state, done, is_won):
    # exp = experience_pb2.Experience()
    exp = self.experience_list.experience.add()
    exp.state = state
    exp.action = action
    exp.reward = reward
    exp.new_state = new_state
    exp.done = done
    exp.is_won = is_won


  def write(self, path):
    f = open(path, 'wb')
    f.write(experience_list.SerializeToString())
    f.close()


  def read(self, path):
    f = open(path, "rb")
    experience_list.ParseFromString(f.read())
    f.close()


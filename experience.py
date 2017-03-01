
import experience_pb2
import sys
import random
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

  win_buffer = None
  lose_buffer = None
  buffer_size = None

  def __init__(self, buffer_size=sys.maxint):
    self.win_buffer = []
    self.lose_buffer = []
    self.buffer_size = buffer_size

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
    if len(buf) >= self.buffer_size:
      buf.pop(0)

  def sample(self, size):
    # return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
    wins = random.sample(self.win_buffer, min(size/2, len(self.win_buffer)))
    losses = random.sample(self.lose_buffer , min(size/2, len(self.lose_buffer)))
    result = ExperienceTable()
    result.table = wins + losses
    random.shuffle(result.table)
    return result

  def all(self):
    return ExperienceTable(self.win_buffer + self.lose_buffer)


class ExperienceRecordingBot:
  war = None


  def __init__(self):
    self.war = agent.War()


  def get_commands(self, game_state, settings):
    commands = []
    stage = Stage(game_state)
    self.war.update_current_battle(stage)
    stage.inp = agent.battle_to_input(self.war.current_battle)

    if not self.war.current_battle.is_end:
      action = np.random.randint(0, agent.OUT_SHAPE)
      stage.action = action
      commands += agent.output_to_command(action, game_state)

    return commands


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


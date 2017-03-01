
import experience_pb2

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


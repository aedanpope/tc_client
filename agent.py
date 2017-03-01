
from map import Map
from logging import log

# MOVES = [(6,0), (-6,0), (0,6), (0,-6), (4,4), (4,-4), (-4,4), (-4,-4)]
MOVES = [(6,0), (-6,0), (0,6), (0,-6)]
# MOVES = [(10,0), (-10,0), (0,10), (0,-10), (7,7), (7,-7), (-7,7), (-7,-7)]
# friendly starts at (70,140), enemy starts at (100,140)
X_MOVE_RANGE = (60,120) # X_MOVE_RANGE and Y_MOVE_RANGE should be the same magnitude.
Y_MOVE_RANGE = (110,190)


# Represents a whole mini-battle.
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


# Represents a single frame in a battle.
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

  log(50, "f1 = " + str(f1))
  log(50, "e1 = " + str(e1))

  return ([Bot.norm(battle.size(), ([0,64]))] +
          Bot.unit_to_vector(f0, True) + Bot.unit_to_vector(f1, True) +
          Bot.unit_to_vector(e0, False) + Bot.unit_to_vector(e1, False))


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


def norm(x, min_max):
  # Map to range [-1.0,1.0]
  # return (2.0*(x - min_max[0]) / (min_max[1] - min_max[0])) - 1
  # Map to range [0,1.0]
  return (float(x) - min_max[0]) / (min_max[1] - min_max[0])


def constrain(x, min_max):
  # truncate X to fit in [x_min,x_max]
  return min(max(x,min_max[0]),min_max[1])



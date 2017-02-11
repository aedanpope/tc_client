# This is a reimplementation of replayer::{Order,Unit,Frame}
# See
# https://github.com/TorchCraft/TorchCraft/blob/master/replayer/frame.h
# https://github.com/TorchCraft/TorchCraft/blob/master/replayer/frame.cpp


from map import Map
from scanner import Scanner

V = False

""" See BWAPI::UnitTypes::Enum::Enum """
UNIT_TYPE_Terran_Marine = 0
UNIT_TYPE_Special_Map_Revealer = 101

class Order:
  first_frame = None # first frame number where order appeared
  type = None # see BWAPI::Orders::Enum
  targetId = None
  targetX = None
  targetY = None


  @staticmethod
  def next_order_from_scanner(sc):
    o = Order()
    o.first_frame = sc.next_int()
    o.type = sc.next_int()
    o.targetId = sc.next_int()
    o.targetX = sc.next_int()
    o.targetY = sc.next_int()
    return o

  def to_str(self):
    return ("type: " + str(self.type) +
            ", targetId: " + str(self.targetId) +
            ", x: " + str(self.targetX) +
            ", y: " + str(self.targetY) +
            ", first_frame: " + str(self.first_frame))
  def __str__(self):
    return self.to_str()
  def __repr__(self):
    return self.to_str()


class Unit:
  id = None
  type = None # see BWAPI::UnitTypes::Enum::Enum
  playerId = None

  health = None
  max_health = None
  armor = None
  shield = None
  max_shield = None
  shieldArmor = None

  energy = None


  idle = None
  visible = None

  x = None
  y = None
  pixel_x = None
  pixel_y = None
  pixel_size_x = None
  pixel_size_y = None
  velocityX = None
  velocityY = None
  size = None

  groundATK = None
  groundDmgType = None
  groundRange = None
  # https://bwapi.github.io/class_b_w_a_p_i_1_1_unit_interface.html#a5023cedeb7a1393b454cdbf172ae57a1
  groundCD = None

  airATK = None
  airDmgType = None
  airRange = None
  airCD = None

  maxCD = None # Seems to apply to both ground and air attack?

  orders = None

  resources = 0

  def to_str(self):
    return ("id: " + str(self.id) +
            ", type: " + str(self.type) +
            ", x: " + str(self.x) +
            ", y: " + str(self.y) +
            ", health: " + str(self.health) +
            ", max_health: " + str(self.max_health) +
            ", pixel_x: " + str(self.pixel_x) +
            ", pixel_y: " + str(self.pixel_y))
  def __str__(self):
    return self.to_str()
  def __repr__(self):
    return self.to_str()


  @staticmethod
  def next_unit_from_scanner(sc):
    # Property vals are read from SC code here:
    # https://github.com/TorchCraft/TorchCraft/blob/master/BWEnv/src/controller.cc#L786
    # Controller::addUnit()

    u = Unit()
    u.id = sc.next_int()
    u.x = sc.next_int()
    u.y = sc.next_int()
    u.health = sc.next_int()
    u.max_health = sc.next_int()
    u.shield = sc.next_int()
    u.max_shield = sc.next_int()
    u.energy = sc.next_int()
    u.maxCD = sc.next_int()
    u.groundCD = sc.next_int()
    u.airCD = sc.next_int()
    u.idle = sc.next_int()
    u.visible = sc.next_int()
    u.type = sc.next_int()
    u.armor = sc.next_int()
    u.shieldArmor = sc.next_int()
    u.size = sc.next_int()
    u.pixel_x = sc.next_int()
    u.pixel_y = sc.next_int()
    u.pixel_size_x = sc.next_int()
    u.pixel_size_y = sc.next_int()
    u.groundATK = sc.next_int()
    u.airATK = sc.next_int()
    u.groundDmgType = sc.next_int()
    u.airDmgType = sc.next_int()
    u.groundRange = sc.next_int()
    u.airRange = sc.next_int()

    u.orders = []
    num_orders = sc.next_int()
    for i in range(0, num_orders):
      u.orders.append(Order.next_order_from_scanner(sc))

    u.velocityX = sc.next_float()
    u.velocityY = sc.next_float()
    u.playerId = sc.next_int()
    u.resources = sc.next_int()

    return u

class Resource:
  ore = None
  gas = None
  used_psi = None
  total_psi = None


class State():

  state_map = None

  #   """ Map of units keyed by ID """
  friendly_units = None
  #   """ Map of units keyed by ID """
  enemy_units = None

  battle_ended = None
  battle_won = None

  def __init__(self, state_map):
    self.state_map = state_map
    if self.state_map.frame != None:
      self._parse_frame() # Get units etc.


  def update(self, new_state):
    # Assumes new_state contains values for all time-dependent fields.
    # We keep the existing non-time-dependenf tiels (e.g. map data)
    for (k,v) in new_state.items():
      self.state_map[k] = v
    if self.state_map.frame != None:
      self._parse_frame() # Refresh units etc.
    self._check_battle_ended()


  def _parse_frame(self):
    # See
    # https://github.com/TorchCraft/TorchCraft/blob/master/replayer/frame.cpp#L119
    # And frame packing here:
    # https://github.com/TorchCraft/TorchCraft/blob/master/BWEnv/src/controller.cc#L555

    if V: print "Parsing frame string:"
    sc = Scanner(self.state_map.frame)

    # Units
    self.units = {}
    num_players = sc.next_int()
    for i in range(0, num_players):
      player_id = sc.next_int()
      if V: print "player_id = " + str(player_id)
      # self.units[player_id] = []
      num_units = sc.next_int()

      # self.units[player_id] = {}
      units = {}

      for j in range(0, num_units):
        unit = Unit.next_unit_from_scanner(sc)
        if (unit.type == UNIT_TYPE_Special_Map_Revealer): continue
        if (unit.id in self.state_map.deaths): continue
        units[unit.id] = unit

      if player_id == self.state_map.player_id:
        self.friendly_units = units
      elif num_players == 2:
        self.enemy_units = units
      else:
        raise Exception("If there's more than 2 players of units, we NFI who enemy is.")

    # TODO Actions

    # TODO Resources

    # TODO Bullets

    # TODO Reward & Terminal

  def _check_battle_ended(self):
    # TODO only check if micro_battles mode.
    # Compute if the battle's over and whos won.
    self.battle_just_ended = False
    friendly_count = 0 if not self.friendly_units else len(self.friendly_units)
    enemy_count = 0 if not self.enemy_units else len(self.enemy_units)
    if self.state_map.deaths != None:
      for uid in self.state_map.deaths:
        if self.friendly_units and uid in self.friendly_units:
          friendly_count -= 1
        elif self.enemy_units and uid in self.enemy_units:
          enemy_count -= 1
        # sometimes the dead unit ID is already removed from the lists,
        # then we have to look in prev state to discern who died.
    if friendly_count == 0 or enemy_count == 0:
      self.battle_just_ended = True
      self.battle_won = enemy_count == 0

  def pretty_print(self):
    print "State:"
    for (k,v) in self.state_map.items():
      if (k in ['frame', 'friendly_units', 'enemy_units']): continue
      print str(k) + ": " + str(v)

    print "friendly_units:"
    if self.friendly_units:
      print '\n'.join(map(str,self.friendly_units.values()))

    print "enemy_units:"
    if self.enemy_units:
      print '\n'.join(map(str,self.enemy_units.values()))
    print ""


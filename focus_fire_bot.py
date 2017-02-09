import tc_client

class FocusFireBot:

  focus_enemy_id = -1

  def get_commands(self, state):
    # Find a non-dead enemy.
    enemy_units = state.enemy_units
    if (enemy_units and not self.focus_enemy_id in enemy_units):
      self.focus_enemy_id = enemy_units.keys()[0]
      print "new focus_enemy_id"

    print "focus_enemy_id: " + str(self.focus_enemy_id)

    commands = []

    if (self.focus_enemy_id != -1):
      for uid in state.friendly_units.keys():
        # Commands are: (unit ID, command, target id, target x, target y, extra)
        # (x, y) are walktiles instead of pixels
        # otherwise this corresponds exactly to BWAPI::UnitCommand
        commands.append([uid, tc_client.UNIT_CMD.Attack_Unit, self.focus_enemy_id])

    return commands




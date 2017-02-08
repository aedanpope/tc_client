import time
import tc_client
from tc_client import TorchClient
from tf_bot import TFBot
from focus_fire_bot import FocusFireBot


if __name__ == '__main__':

  port = 11111
  hostname = "127.0.0.1"

  tc = TorchClient(hostname, port)

  tc.initial_map = 'Maps/BroodWar/micro/m5v5_c_far.scm'
  tc.window_pos = [100, 100]
  tc.window_size = [640, 480]
  tc.mode.micro_battles = True
  tc.mode.replay = False
  tc.mode.replay = False

  tc.connect()

  # TODO  implement tc:set_variables() here.

  # We prob want to store a local version of the gamestate as well in another class,
  # Since we get back some crazy shit.

  tc.send([
      [tc_client.CMD.set_speed, 5],
      [tc_client.CMD.set_gui, 1],
      [tc_client.CMD.set_frameskip, 5],
      [tc_client.CMD.set_cmd_optim, 1],
      [tc_client.CMD.set_combine_frames, 5],
    ])

  x = 0
  focus_uid = -1

  bot = FocusFireBot()

  # bot = TFBot()

  update = tc.receive() # Get the first state so we can see our starting units.
  tc.send([])
  total_battles = 0
  battles_won = 0
  while True:
    update = tc.receive()

    # Inspect tc.state and figure out what to do.
    tc.state.pretty_print()

    if tc.state.battle_just_ended:
      if tc.state.battle_won:
        battles_won += 1
      total_battles += 1
      print "\nBATTLE ENDED"
      print "total_battles = " + str(total_battles)
      print "battles_won = " + str(battles_won)
      print ""

    # Populate commands.
    unit_commands = bot.get_commands(tc.state);
    commands = [[tc_client.CMD.command_unit_protected] + unit_command for unit_command in unit_commands]
    print "commands = " + str(commands)

    tc_client.CMD.command_unit_protected

    # Send the orders.
    tc.send(commands)
    x += 1
    time.sleep(0.5)

    # if x == 2: break

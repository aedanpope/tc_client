### USAGE:
# p exercise.py -s 1 -k 2 -t 3 -hp "greedy={'ACTION_STRATEGY':Act.Greed}, boltzmann={'ACTION_STRATEGY':Act.Boltzmann}" -f results/dnq_action_strategy_10k20k.txt

import time
import tc_client
import sys
import select
from tc_client import TorchClient
import tf_bot
import bot_q_learner_simple_a
import policy_bot
import advantage_bot
from agent import Settings
from agent import Mode
from map import Map
import dnq_bot
from focus_fire_bot import FocusFireBot
from scanner import Scanner
import argparse

V = False


out_file = None


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Run a StarCraft torch client.')
  parser.add_argument('-s', '--speed', type=int, default=1,
      help='The speed to run starcraft in, 1-13 with 13 being slowest')
  parser.add_argument('-k', '--kite', type=int, default=2, choices=[2, 3, 4],
      help='Which kite map to load, options are 2 and 4')
  parser.add_argument('-f', '--out_file', help='File to log results of trials to.')
  parser.add_argument('-t', '--trials', type=int, default=5,
      help='Number of trials to run for each config')
  parser.add_argument('-hp', '--hyperparameter_sets',
      help='Different sets of hyperparameters to evaluate num --trials times.')
  parser.add_argument('-v', '--forever', default=False, action='store_true',
      help='Run the first trial forever.')


  args = parser.parse_args()

  print 'Speed = ', str(args.speed)
  speed = args.speed

  settings = Settings()
  settings.verbosity = speed
  settings.mode = Mode.train


  # bot = FocusFireBot()
  # bot = tf_bot.Bot()
  # bot = advantage_bot.Bot()
  # bot = policy_bot.Bot()
  # bot = bot_q_learner_simple_a.Bot()

  port = 11111
  hostname = "127.0.0.1"

  tc = TorchClient(hostname, port)

  # tc.initial_map = 'Maps/BroodWar/micro/m5v5_c_far.scm'
  # # tc.initial_map = 'Maps/BroodWar/micro/Marine1vZergling1_c_far.scm'
  # tc.initial_map = 'Maps/BroodWar/micro/Marine1vZergling1_20s.scm'
  # tc.initial_map = 'Maps/BroodWar/micro/kite.scm'
  # tc.initial_map = 'Maps/BroodWar/micro/kite_2hit.scm'
  tc.initial_map = "Maps/BroodWar/micro/kite_" + str(args.kite) + ".scm"
  # tc.initial_map = 'Maps/BroodWar/micro/Vulture_v_Zealot.scm'
  tc.window_pos = [100, 100]
  tc.window_size = [640, 480]
  tc.mode.micro_battles = True
  tc.mode.replay = False

  tc.connect()

  # TODO  implement tc:set_variables() here.

  # We prob want to store a local version of the gamestate as well in another class,
  # Since we get back some crazy shit.

  # tc.send([
  #     [tc_client.CMD.restart, 1],
  #   ])
  # update = tc.receive()

  tc.send([
      # [tc_client.CMD.set_speed, 13],
      [tc_client.CMD.set_speed, speed],
      # [tc_client.CMD.set_gui, 0],
      # [tc_client.CMD.set_gui, 1],
      [tc_client.CMD.set_frameskip, 5],
      [tc_client.CMD.set_cmd_optim, 1],
      [tc_client.CMD.set_combine_frames, 5],
    ])


  update = tc.receive() # Get the first state so we can see our starting units.
  tc.send([])

  # Make sure each Trial gets the same number of steps to train, not battles.
  # So that sneaky agents don't get extra training time except an epsilon in the last training battle.
  training_steps = dnq_bot.HP.PRE_TRAIN_STEPS + int(1.5*dnq_bot.HP.ANNEALING_STEPS)
  if (args.forever):
    training_steps = 99999999
  test_battles = 100

  if args.out_file:
    out_file = open(args.out_file, 'a')
    print >>out_file, ""
    print >>out_file, "Command:"
    print >>out_file, str(' '.join(sys.argv))

  hyperparameter_sets = Map({'default_params': {}})
  if args.hyperparameter_sets:
    hyperparameter_sets = dnq_bot.parse_hyperparameter_sets(args.hyperparameter_sets)
    # eval("Map("+args.hyperparameter_sets+")")
  print "hyperparameter_sets = " + str(hyperparameter_sets)
  print ""

  if out_file:
    print >>out_file, "default hyperparameters: " + str(dnq_bot.HP)
  print "default hyperparameters: " + str(dnq_bot.HP)

  for (case,hyperparameters) in hyperparameter_sets.items():
    # Just check they all parse.
    dnq_bot.process_hyperparameters(hyperparameters)

  for (case,hyperparameters) in hyperparameter_sets.items():
    if out_file:
      print >>out_file, "case: " + str(case)
      print >>out_file, "hyperparameters: " + str(hyperparameters)
      file.flush(out_file)

    print "case: " + str(case)
    print "hyperparameters: " + str(hyperparameters)

    for trial in range(0,args.trials):
      bot = dnq_bot.Bot(hyperparameters)

      settings.mode = Mode.train
      steps = 0
      train_battles_fought = 0
      train_battles_won = 0
      test_battles_fought = 0
      test_battles_won = 0
      while test_battles_fought < test_battles:
        update = tc.receive()
        commands = []

        if tc.state.battle_just_ended:
          print "\nBATTLE ENDED"
          print ""

          won = int(tc.state.battle_won)
          if (settings.mode == Mode.train):
            train_battles_fought += 1
            train_battles_won += won
          else:
            test_battles_fought += 1
            test_battles_won += won

          if (steps >= training_steps):
            settings.mode = Mode.test

          print "case = " + str(case)
          print "trial = " + str(trial)
          print "train_battles_fought = " + str(train_battles_fought)
          print "train_battles_won = " + str(train_battles_won)
          print "test_battles_fought = " + str(test_battles_fought) + "/" + str(test_battles)
          print "test_battles_won = " + str(test_battles_won)
          print "settings.mode = " + str(settings.mode)
          print "steps = " + str(steps) + "/" + str(training_steps)

        # Don't send repeated frames of the same ended battle state.
        if not tc.state.battle_ended or tc.state.battle_just_ended:
          if not tc.state.battle_ended:
            steps += 1
          # Populate commands.
          unit_commands = bot.get_commands(tc.state, settings);
          commands = [[tc_client.CMD.command_unit_protected] + unit_command for unit_command in unit_commands]

        # http://stackoverflow.com/questions/3762881/how-do-i-check-if-stdin-has-some-data
        if select.select([sys.stdin,],[],[],0.0)[0]:
            line = sys.stdin.readline()
            print "Parsing User Input " + line + ""
            sc = Scanner(line)
            speed = sc.next_int()
            print "Setting speed to " + str(speed)
            settings.verbosity = speed
            commands.append([tc_client.CMD.set_speed, speed])
            time.sleep(1)


        if V: print "commands = " + str(commands)

        # Send the orders.
        tc.send(commands)

      print "trial " + str(trial)
      print "train win rate: " + str(float(train_battles_won) / train_battles_fought)
      print "test win rate: " + str(float(test_battles_won) / test_battles_fought)
      print ""
      print ""
      print "**********************"
      print ""
      print ""

      if out_file:
        print >>out_file, "trial " + str(trial)
        print >>out_file, "train win rate: " + str(float(train_battles_won) / train_battles_fought)
        print >>out_file, "test win rate: " + str(float(test_battles_won) / test_battles_fought)
        file.flush(out_file)

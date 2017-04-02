### USAGE:
# Run some experiment with trials:
# p exercise.py -s 1 -k 2 -t 3 -hp "greedy={'ACTION_STRATEGY':Act.Greedy}, boltzmann={'ACTION_STRATEGY':Act.Boltzmann}" -f results/dqn_action_strategy_10k20k.txt
#
# Quick run forever:
# p exercise.py -s 1 -k 2 -t 1 -hp "foo={'PRE_TRAIN_STEPS':100, 'ANNEALING_STEPS':1000}" -v
#
# Record experience
# p exercise.py -s 1 -k 2 -t 1 -hp "foo={'PRE_TRAIN_STEPS':1000, 'ANNEALING_STEPS':0}" --test_battles=0  --record=data/1000steps.pb
#
# Run forever and read some existing experience.
# p exercise.py -s 1 -k 2 -t 1 -hp "foo={'PRE_TRAIN_STEPS':100, 'ANNEALING_STEPS':1000}" -v --experience=data/1000steps.pb
#
# Read existing experience and run exercise
# p exercise.py -s 1 -k 2 --trials=3 -hp \
# "greedy={'ACTION_STRATEGY':Act.Greedy, 'PRE_TRAIN_STEPS':0, 'ANNEALING_STEPS':20000}, \
# boltzmann={'ACTION_STRATEGY':Act.Boltzmann_B, 'PRE_TRAIN_STEPS':0, 'ANNEALING_STEPS':20000}" \
#  --experience=data/20000steps_28w_2518l.pb
#  --out_file=results/2017_02_01_1822.txt

import time
import tc_client
import sys
import select
from tc_client import TorchClient
import tf_bot
import bot_q_learner_simple_a
import policy_bot
import advantage_bot
import experience
from dqn_bot import Mode
from map import Map
import my_logging
from my_logging import log
import dqn_bot
from focus_fire_bot import FocusFireBot
from scanner import Scanner
import argparse


out_file = None

def output(s):
  print s
  if out_file:
    print >>out_file, s
    file.flush(out_file)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Run a StarCraft torch client.')
  parser.add_argument('-s', '--speed', type=int, default=1,
      help='The speed to run starcraft in, 1-13 with 13 being slowest')
  parser.add_argument('-k', '--kite', default='2', choices=['2', '3', '4', '2o3', '2o3o4'],
      help='Which kite map to load.')
  parser.add_argument('-f', '--out_file', help='File to log results of trials to.')
  parser.add_argument('-t', '--trials', type=int, default=1,
      help='Number of trials to run for each config')
  parser.add_argument('--test_battles', type=int, default=100,
      help='Number of test battles to run to evaluate a trial.')
  parser.add_argument('--test_period', type=int, default=5000,
      help='Number of timesteps between tests')
  parser.add_argument('-hp', '--hyperparameter_sets',
      help='Different sets of hyperparameters to evaluate num --trials times.')
  parser.add_argument('-v', '--forever', default=False, action='store_true',
      help='Run the first trial forever.')
  parser.add_argument('--record',
      help='Record random experiences to this filename.')
  parser.add_argument('--experience',
      help='Filename containing experience to pre-fill the buffer with.')


  args = parser.parse_args()

  print 'Speed = ', str(args.speed)
  speed = args.speed

  my_logging.VERBOSITY = speed


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
  tc.window_pos = [0, 0]
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
      [tc_client.CMD.set_speed, speed],
      # [tc_client.CMD.set_gui, 1],
      [tc_client.CMD.set_frameskip, 5],
      [tc_client.CMD.set_cmd_optim, 1],
      [tc_client.CMD.set_combine_frames, 5],
    ])


  update = tc.receive() # Get the first state so we can see our starting units.
  tc.send([])

  if args.out_file:
    out_file = open(args.out_file, 'a')
    print >>out_file, ""
    print >>out_file, "Command:"
    print >>out_file, str(' '.join(sys.argv))

  hyperparameter_sets = Map({'default_params': {}})
  if args.hyperparameter_sets:
    hyperparameter_sets = dqn_bot.parse_hyperparameter_sets(args.hyperparameter_sets)
  print "hyperparameter_sets = " + str(hyperparameter_sets)
  print ""

  if out_file:
    print >>out_file, "default hyperparameters: " + str(dqn_bot.HP)
  print "default hyperparameters: " + str(dqn_bot.HP)

  first_keys = set(hyperparameter_sets.values()[0].keys())
  for (case,hyperparameters) in hyperparameter_sets.items():
    # Just check they all parse.
    dqn_bot.process_hyperparameters(hyperparameters)
    if (first_keys != set(hyperparameters.keys())):
      raise Exception("not all hyperparameter sets have the same keys. All keys must be specified so that order doesn't matter.")

  for (case,hyperparameters) in hyperparameter_sets.items():
    if out_file:
      print >>out_file, "case: " + str(case)
      print >>out_file, "hyperparameters: " + str(hyperparameters)
      file.flush(out_file)

    print "case: " + str(case)
    print "hyperparameters: " + str(hyperparameters)
    # Process now so they apply to num trainging steps.
    dqn_bot.process_hyperparameters(hyperparameters)

    # Make sure each Trial gets the same number of steps to train, not battles.
    # So that sneaky agents don't get extra training time except an epsilon in the last training battle.
    train_for_steps = dqn_bot.HP.PRE_TRAIN_STEPS + dqn_bot.HP.ANNEALING_STEPS + dqn_bot.HP.POST_ANNEALING_STEPS
    if (args.forever):
      train_for_steps = 99999999

    for trial in range(0,args.trials):
      if args.record:
        print "ExperienceRecordingBot"
        bot = experience.ExperienceRecordingBot(args.record)
      else:
        print "dqn_bot.Bot"
        bot = dqn_bot.Bot(hyperparameters, args.experience)


      output("START case: " + str(case) + ", " + "trial: " + str(trial))

      steps_trained = 0
      train_battles_fought = 0
      train_battles_won = 0

      test_battles_per_kite = Map()
      test_battles_fought = 0
      test_battles_won = 0
      test_battles = None
      last_test_result = None

      kite_n = None
      test_battles_fought_per_kite = {}
      test_battles_won_per_kite = {}

      # Start with a test:
      step_last_test_start = 0
      test_battles = args.test_battles
      if test_battles > 0:
        mode = Mode.test
      else:
        mode = Mode.train

      # train_for_steps+100 not +0 , so that we can finish the last training battle.
      while steps_trained <= train_for_steps+100 or (mode == Mode.test and test_battles_fought < test_battles):
        update = tc.receive()
        commands = []



        # Don't send repeated frames of the same ended battle state.
        if not tc.state.battle_ended or tc.state.battle_just_ended:
          if not tc.state.battle_ended and mode == Mode.train:
            steps_trained += 1
          if kite_n is None:
            kite_n = 1+((tc.state.enemy_units.values()[0].get_life()-1) / 20)
            if not kite_n in test_battles_fought_per_kite:
              test_battles_fought_per_kite[kite_n] = 0
              test_battles_won_per_kite[kite_n] = 0
          # Populate commands.
          unit_commands = bot.get_commands(tc.state, mode);
          commands = [[tc_client.CMD.command_unit_protected] + unit_command for unit_command in unit_commands]

        # http://stackoverflow.com/questions/3762881/how-do-i-check-if-stdin-has-some-data
        if select.select([sys.stdin,],[],[],0.0)[0]:
          line = sys.stdin.readline()
          print "Parsing UserInput '" + line + "'"
          sc = Scanner(line)
          if sc.has_next_int():
            speed = sc.next_int()
            print "Setting speed to " + str(speed)
            my_logging.VERBOSITY = speed
            commands.append([tc_client.CMD.set_speed, speed])
          elif sc.has_next_word():
            cmd = sc.next_word()
            if cmd == "test" and sc.has_next_int():
              test_battles = sc.next_int()
              test_battles_fought = 0
              test_battles_won = 0
              mode = Mode.test
              print "Manually testing for " + str(input_test_battles) + " battles"
            else:
              print "Invalid UserInput"
          time.sleep(1)

        # Send the orders.
        log("commands = " + str(commands), 30)
        tc.send(commands)

        if tc.state.battle_just_ended:
          print "\nBATTLE ENDED"
          print ""

          print "kite_n = " + str(kite_n)
          won = int(tc.state.battle_won)

          if (mode == Mode.train):
            train_battles_fought += 1
            train_battles_won += won
          else:
            test_battles_fought += 1
            test_battles_won += won
            test_battles_fought_per_kite[kite_n] += 1
            test_battles_won_per_kite[kite_n] += won

            test_battles -= 1
            if test_battles == 0:
              test_battles = None
              print "\n\n\n**********************\n\n\n"
              output("Test Finished")
              output("steps_trained = " + str(steps_trained))
              last_test_result = float(test_battles_won) / test_battles_fought
              output("Results = " + str(test_battles_won) + "/" + str(test_battles_fought) +
                     " = " + str(last_test_result))
              test_battles_fought = 0
              test_battles_won = 0
              for n in test_battles_fought_per_kite.keys():
                kite_n_last_test_result = float(test_battles_won_per_kite[n]) / test_battles_fought_per_kite[n]
                output("kite_" + str(n) + " Results = " +
                       str(test_battles_won_per_kite[n]) + "/" + str(test_battles_fought_per_kite[n]) +
                       " = " + str(kite_n_last_test_result))
                test_battles_fought_per_kite[n] = 0
                test_battles_won_per_kite[n] = 0
              time.sleep(1)
              mode = Mode.train
              if (steps_trained >= train_for_steps):
                # Finished training and testing, finished with this trial.
                break
          kite_n = None

          print "case = " + str(case)
          print "trial = " + str(trial)
          print "train_battles_fought = " + str(train_battles_fought)
          print "train_battles_won = " + str(train_battles_won)
          print "step_last_test_start = " + str(step_last_test_start)
          print "last_test_results = " + str(last_test_result)
          print "test_battles_fought = " + str(test_battles_fought)
          print "test_battles_won = " + str(test_battles_won)

          for n in test_battles_fought_per_kite.keys():
            print "kite_" + str(n) + " test_battles_fought = " + str(test_battles_fought_per_kite[n])
            print "kite_" + str(n) + " test_battles_won = " + str(test_battles_won_per_kite[n])

          print "current_mode = " + str(mode)
          print "steps_trained = " + str(steps_trained) + "/" + str(train_for_steps)
          print "check_for_test = " + str((steps_trained - step_last_test_start) / args.test_period)

          # Start testing if we've finished training, or periodically.
          if mode == Mode.train and args.test_battles > 0 and (
             steps_trained >= train_for_steps or ((steps_trained - step_last_test_start) / args.test_period >= 1)):
            mode = Mode.test
            step_last_test_start = steps_trained
            test_battles = args.test_battles
            print "\n\n\n**********************\n\n\n"
            print "starting test for " + str(test_battles) + " battles"
            time.sleep(1)


      bot.close()
      output("END case: " + str(case) + ", " + "trial: " + str(trial))
      output("train win rate: " + str(float(train_battles_won) / train_battles_fought))
      output("test win rate: " + str(last_test_result))
      print "\n\n\n**********************\n\n\n"

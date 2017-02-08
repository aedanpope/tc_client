import time
import zmq
import torch
import re
from map import Map
from state import State
from scanner import Scanner


###################
# GLOBAL CONSTANTS.
###################

# TorchCraft API version
PROTOCOL_VERSION = "16"

# All available commands to the server.
CMD = Map(
    # no arguments
    quit = 0,                  # leave the game
    restart = 1,               # resetart the game. Much faster, but doesn't
                               #   work in multiplayer.
    map_hack = 2,              # remove fog of war
    request_image = 3,
    exit_process = 4,
    noop = 5,                  # do nothing

    # one argument
    set_speed = 6,             # sets the game speed (integer)
    set_log = 7,               # activates logging (boolean)
    set_gui = 8,               # activates drawing and text in SC (boolean)
    set_frameskip = 9,         # number of frames to skip (integer)
    set_cmd_optim = 10,        # reduce bot APM (0-6)
    set_combine_frames = 11,   # combine n frames before sending (integer)

    # Sets the map with BWAPI->setMap and by writing to the config. Is not
    #   thread-safe. However, as long as the next connect finishes after
    #   set_map, you are guaranteed the map will be what you want.
    set_map = 12,
    set_multi = 13,

    # arguments: (unit ID, command, target id, target x, target y, extra)
    # (x, y) are walktiles instead of pixels
    # otherwise this corresponds exactly to BWAPI::UnitCommand
    command_unit = 14,
    command_unit_protected = 15,

    # arguments: (command, args)
    # For documentation about args, see usercommandtypes
    command_user = 16,

    #
    MAX_ACTION = 17
)

# corresponds to BWAPI::UnitCommandTypes::Enum
UNIT_CMD = Map(
    Attack_Move = 0,
    Attack_Unit = 1,
    Build = 2,
    Build_Addon = 3,
    Train = 4,
    Morph = 5,
    Research = 6,
    Upgrade = 7,
    Set_Rally_Position = 8,
    Set_Rally_Unit = 9,
    Move = 10,
    Patrol = 11,
    Hold_Position = 12,
    Stop = 13,
    Follow = 14,
    Gather = 15,
    Return_Cargo = 16,
    Repair = 17,
    Burrow = 18,
    Unburrow = 19,
    Cloak = 20,
    Decloak = 21,
    Siege = 22,
    Unsiege = 23,
    Lift = 24,
    Land = 25,
    Load = 26,
    Unload = 27,
    Unload_All = 28,
    Unload_All_Position = 29,
    Right_Click_Position = 30,
    Right_Click_Unit = 31,
    Halt_Construction = 32,
    Cancel_Construction = 33,
    Cancel_Addon = 34,
    Cancel_Train = 35,
    Cancel_Train_Slot = 36,
    Cancel_Morph = 37,
    Cancel_Research = 38,
    Cancel_Upgrade = 39,
    Use_Tech = 40,
    Use_Tech_Position = 41,
    Use_Tech_Unit = 42,
    Place_COP = 43,
    None_Command = 44,
    Unknown = 45,
    MAX = 46
)

# TODO: implement user commands and stuff.

######################
# END GLOBAL CONSTANTS.
######################

class Mode:
  micro_battles = True
  replay = False


class TorchClient:
  'Client for talking to a TorchServer'

  # Networking layer.
  hostname = ""
  port = 0
  socket = None
  zcontext = None

  # Alternate 1:1 send/receive.
  sent_message = False

  # TorchCraft state.
  state = None
  initial_map = None
  window_size = None
  window_pos = None
  mode = Mode()


  def __init__(self, hostname, port):
    """ Constructor """
    # TODO add local VM host support.
    self.hostname = hostname
    self.port = port
    print 'host:' + self.hostname + ':' + str(self.port)
    # we always need to make sure we alternate 1:1 send/receive, thus:
    self.sent_message = False


  def connect(self):
    """ Connect should be called at the beginning of every game. """
    addr = 'tcp://' + self.hostname + ':' + str(self.port)
    print "connecting to " + addr

    # Init socket connection.
    self.zcontext = zmq.Context()
    self.socket= self.zcontext.socket(zmq.REQ)
    res = self.socket.connect(addr)


    if (self.socket == None):
      print "Socket error connecting to '" + addr + "'"

    hello = 'protocol=' + PROTOCOL_VERSION

    if self.initial_map:
        hello += ",map=" + self.initial_map

    if self.window_size:
        hello += ",window_size=" + str(self.window_size[0]) + " " + str(self.window_size[1])

    if self.window_pos:
        hello += ",window_pos=" + str(self.window_pos[0]) + " " + str(self.window_pos[1])

    hello += ",micro_mode=" + str(self.mode.micro_battles)

    self.socket.send(hello)

    print "getting setup message"
    setup_msg = self.socket.recv()
    print "got a setup msg"
    self.state = State(TorchClient._unpack_message(setup_msg))
    print "SETUP STATE:"# + str(self.state)[:20] + "..." + str(self.state)[-20:]
    self.state.pretty_print()
    self.sent_message = False

  def send(self, commands):
    """ Sends the table to the StarCraft Server

        Table is a [list] of lists [CMD, valuea, valueb...] where keys are ints from CMD

        Send a string in the form "CMD1,valuea,valueb:CMD2,value2..."
        e.g. "6,15:8,1:9,5:10,1:11,30"
    """
    if self.sent_message:
      raise Exception("can't send another message without receiving one first")

    msg = ':'.join([','.join([str(c) for c in cmd]) for cmd in commands])
    print "Sending message: '" + msg + "'"

    self.socket.send(msg)
    self.sent_message = True


  def receive(self):
    if not self.sent_message:
      raise Exception("can't receive a message without sending one first")

    if not self.socket.poll(30000):
        # Timed out after 30s, starcraft.exe probably crashed.
        self.close()
        raise Exception("starcraft.exe crashed")

    msg = self.socket.recv()
    self.sent_message = False
    self.state.update(TorchClient._unpack_message(msg))
    print "Got State Update."


  def close(self):
    self.socket.close()


  @staticmethod
  def _unpack_message(msg):
    """ Parses the strings generated by

        Controller::setupHandshake()
        https://github.com/TorchCraft/TorchCraft/blob/master/BWEnv/src/controller.cc#L195

        Utils::mapToTensorStr()
        https://github.com/TorchCraft/TorchCraft/blob/master/BWEnv/src/utils.cc#L181

        Replayer::Frame::<<
        https://github.com/TorchCraft/TorchCraft/blob/master/replayer/frame.cpp#L90
        https://github.com/TorchCraft/TorchCraft/blob/master/BWEnv/src/controller.cc#L617
        [[ a_units unit_1_x unit_1_y ... unit_a_y
           b_actions action_1 ... action_b
           c_resources ... d_bullets
           some_reward is_terminal
        ]]

        e.g.
        "{lag_frames = 2, map_name = 'm5v5_c_far.scm',
          map_data = torch.ByteTensor({{2,2,2,2,-1,-1}},
          frame=[[$FRAME_BYTES]])}"


        Doesn't support response when CMD.request_image = True yet, frames have key TCIMAGEDATA
    """

    # Reformat some lua syntax, and drop wrapping {}.
    msg = msg[1:-1]
    msg = msg.replace(",}", "}")
    msg = msg.replace("}", "]")
    msg = msg.replace("{", "[")
    msg = msg.replace("false", "False")
    msg = msg.replace("true", "True")

    # Fix "frame = [[2 0 -4 0.041 224 160 ... 10 0 0 0 0]]"
    # to  "frame = '2 0 -4 0.041 224 160 ... 10 0 0 0 0'"
    msg = re.sub(r"frame\s*=\s*\[\[([-?\d+\.?\d*\s]*)\]\]",
                 "frame = \'\\1\'", msg)

    # Eval it into a map.
    msg = "Map("+msg+")"
    return eval(msg)

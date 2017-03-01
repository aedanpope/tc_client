# TensorFlow and StarCraft

## Background

#### StarCraft
- [StarCraft](https://en.wikipedia.org/wiki/StarCraft_(video_game)): A 1998 Real-Time strategy game. Used in lots of AI research, and also has a [strong professional competetive following](https://en.wikipedia.org/wiki/Professional_StarCraft_competition) in Korea worth ~millions of USD.
- [Brood War](https://en.wikipedia.org/wiki/StarCraft:_Brood_War): The expansion pack for StarCraft.

All current professional competition and research uses StarCraft: Brood War.

#### AI Research with starcraft:
Recent examples:
- [Usunier et al, 2016](https://arxiv.org/abs/1609.02993), Episodic Exploration for Deep Deterministic Policies:
An Application to StarCraft Micromanagement Tasks
- [Foerster et al, 2017](https://arxiv.org/abs/1702.08887), Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning

#### BWAPI
The Brood War Application Programming Interface ([BWAPI](https://github.com/bwapi/bwapi)) allows interaction with a StarCraft client through C++ code.

It is implemented as a C++ DLL injected into a running StarCraft client. The user of the API adds code interacting with the BWAPI library, builds a DLL, and runs a tool to inject that into a StarCraft client.

#### Torch

[Torch](http://torch.ch/) is a Lua framework for implementing machine learning algorithms, with GPU support.

Torch is open-source, and maintained primarily by Facebook.

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) is an open-source library for machine learning like Torch, but has support in more languages (Python, C++, Java, GO) and more features (e.g. run on a mobile device.).

TensorFlow is maintained by Google's Brain team. TensorFlow left Beta and V1  was published on 15th of Feb, 2017 ([github](https://github.com/tensorflow/tensorflow/releases/tag/v1.0.0), (announcement video)[https://www.youtube.com/watch?v=mWl45NkFBOc])

Tensor flow's [github](https://github.com/tensorflow/tensorflow) has ~10x more commits than [Torch's](https://github.com/torch/torch7).


#### TorchCraft:

[TorchCraft](https://github.com/TorchCraft/TorchCraft) is A bridge between Torch and BWAPI, consisting of:
- A socket server implemented in a C++ BWAPI DLL, that serves the StarCraft game state and takes commands.
- A Lua client for talking to the server.
- Examples of Lua code that implement a StarCraft agent using the Torch machine learning library.

The main advantage of TorchCraft is that one can build a StarCraft agent in a Unix environment.


## This Project

1. A Python client for the BWAPI server from TorchCraft.
2. Examples of StarCraft agents implemented using the TensorFlow python libraries.
3. A Deep Q-network to learn the ["kiting"](http://wiki.teamliquid.net/starcraft2/Kiting) mechanic in StarCraft.

### Python Client for BWAPI

Why? So that we can use the very nice TensorFlow python api in a native Unix environment for building StarCraft AIs.

We've written a python client for the TorchCraft C++ server:
- Network IO is generally in [tc_client.py](tc_client.py).
- The TorchCraft server returns the game-state as a Lua object string, which we transform into a python object string and ```eval()``` [here](https://github.com/aedanpope/tc_client/blob/427aafc9aa5dce7561325e74c64f4e8a13905e5e/tc_client.py#L254).
- [state.py](state.py) is responsible for parsing the responses from the server and turning them into typed obejects.


### TensorFlow Examples

The existing best-class environment for StarCraft research is TorchCraft (used in [[1](https://arxiv.org/abs/1609.02993), [2](https://arxiv.org/abs/1702.08887)]).

TensorFlow support for BWAPI makes writing machine learning agents more accessible, and is a solid long-term investment as TensorFlow grows in popularity and functionality.

For example, [Juliani's Q-Learing Part 0 blog post](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.icolg93n8) cointains sample TensorFlow code for a simple Q-network, which we've implemented to control a starcraft agent in [bot_q_learner_simple_a.py](bot_q_learner_simple_a.py) ([Relevant TensorFlow code here](https://github.com/aedanpope/tc_client/blob/728ac6b889b1aa702ecea65a7a49bdb99d2625cd/bot_q_learner_simple_a.py#L127).

### A DQN for Kiting

#### Intro
Recent AI research using Q-networks has looked primarily at symmetrical battles of groups of Marines. In particular, Marine 5v5 where the opponent just singly attack-moves the AI controlled 5 marines was studied in [Usunier et al, 2016](https://arxiv.org/abs/1609.02993) and [Foerster et al, 2017](https://arxiv.org/abs/1702.08887).

This is a good environment for exploring the multi-agent problem, and extending models to the stochasitc high-dimensional StarCraft space.

There are many challenges to StarCraft that this environment does not expose us to:
- Exploration complexity: In (Foerster, 2017), they measure that giving _no_ orders to the friendly controlled marines leads to a win rate of 84%, this means that learning agents have somewhere decent to iteratively learn and improve from.
- Asymmetry: One of the interesting AI challenges of StarCraft is the asymmetry in micro battles and races. There are 3 distint races with very different units and technologies available, leading to 9 different roles a completely general StarCraft AI has to learn (i.e. controlling {Terran, Zerg, Protos} vs enemy {Terran, Zerg, Protoss} are 9 distinct problems to solve with many common features).

Consider a battle a [Terran Vulture](http://wiki.teamliquid.net/starcraft/Vulture) and a [Protoss Zealot](http://wiki.teamliquid.net/starcraft/Zealot).

The vulture is a fast, fragile unit with a ranged attack. The Zealot is a strong slow-moving unit with a melee attack. If a Vulture and a Zealot simply move directly to attack each other - the Zealot will win.

In professional StarCraft, it is commonly accepted that Vultures beat Zealots - because expert players will [micro](http://wiki.teamliquid.net/starcraft2/Micro_(StarCraft)#Battle_micro) the vultures to hit the zealot once from range, then dance back before the zealot can attack and hit it again. This techniqe of "dancing back" is called [kiting](http://wiki.teamliquid.net/starcraft2/Kiting).

A kiting micro battle is one that is both:
- hard to win randomly
- asymmetrical

There has been [some previous research](https://scholar.google.co.uk/scholar?hl=en&q=starcraft+kiting&btnG=&as_sdt=1%2C5&as_sdtp=) into kiting in StarCraft using machine learning. Notably [Szlam 2016](https://arxiv.org/abs/1511.07401) where a generalized gaming agent was able to develope highly successful kiting strategies with reinforcement learning.

In this project we consider the Kiting problem as a exercise to:
- Demonstrating the usefulness of the TensorFlow integration into BWAPI.
- See if a relatively generic Deep Q-learning network can solve the kiting problem (that is, a network largely like that described in [(Lillicrap and Hunt, 2016)](https://arxiv.org/pdf/1509.02971.pdf))

#### Kiting Environemt

Our environment consits of a 1v1 battle between a Vulture and a Zealot.

To simplify the game state, we have modified the hit points and damage of the Vulture and Zealot such that one attack from the Zealot kills the vulture, and the number of attacks for the Vulture to kill the zealot is a parameter "n-kite" where n is in [2,3,4]

A human winning the 4-kite exercise: https://www.youtube.com/watch?v=PnEhLxpL29U


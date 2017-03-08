# TensorFlow and StarCraft

* [Background](#background)
      * [StarCraft](#starcraft)
      * [AI research with StarCraft](#ai-research-with-starcraft)
      * [BWAPI](#bwapi)
      * [Torch](#torch)
      * [TensorFlow](#tensorflow)
      * [TorchCraft](#torchcraft)
* [This project](#this-project)
   * [1. Python Client for BWAPI](#1-python-client-for-bwapi)
   * [2. StarCraft TensorFlow agent examples](#2-starcraft-tensorflow-agent-examples)
   * [3. A DQN for solving a difficult binary-success problem in StarCraft (that is, a DQN for a kiting micro-battle).](#3-a-dqn-for-solving-a-difficult-binary-success-problem-in-starcraft-that-is-a-dqn-for-a-kiting-micro-battle)
      * [Abstract](#abstract)
      * [Introduction](#introduction)
      * [Environment and Parameterization](#environment-and-parameterization)
      * [Algorithm](#algorithm)
      * [Results](#results)
         * [Rewards](#rewards)
         * [Experience Buffers and Exploration](#experience-buffers-and-exploration)
      * [Conclusions](#conclusions)
      * [Future Work](#future-work)
      * [Implementation](#implementation)
      * [References](#references)

## Background

#### StarCraft

- [StarCraft](https://en.wikipedia.org/wiki/StarCraft_(video_game)): A 1998 Real-Time strategy game. Used in lots of AI research, and also has a [strong professional competetive following](https://en.wikipedia.org/wiki/Professional_StarCraft_competition) in Korea worth ~millions of USD.
- [Brood War](https://en.wikipedia.org/wiki/StarCraft:_Brood_War): The expansion pack for StarCraft.

All professional competition and research with StarCraft these days uses the Brood War expansion.

#### AI research with StarCraft

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

[TensorFlow](https://www.tensorflow.org/) is an open-source library for machine learning like Torch, but has support in more languages (Python, C++, Java, GO) and more features (e.g. run on a mobile device, Google Cloud Platform).

TensorFlow is maintained by Google's Brain team. TensorFlow left Beta and V1  was published on 15th of Feb, 2017 ([github](https://github.com/tensorflow/tensorflow/releases/tag/v1.0.0), (announcement video)[https://www.youtube.com/watch?v=mWl45NkFBOc])

#### TorchCraft

[TorchCraft](https://github.com/TorchCraft/TorchCraft) is A bridge between Torch and BWAPI, consisting of:
- A socket server implemented in a C++ BWAPI DLL, that serves the StarCraft game state and takes commands.
- A Lua client for talking to the server.
- Examples of Lua code that implement a StarCraft agent using the Torch machine learning library.

The main advantage of TorchCraft is that one can build a StarCraft agent in a Unix environment.


## This project

1. A Python client for the BWAPI server from TorchCraft.
2. Examples of StarCraft agents implemented using the TensorFlow python libraries.
3. A Deep Q-network to learn the ["kiting"](http://wiki.teamliquid.net/starcraft2/Kiting) mechanic in StarCraft.

### 1. Python Client for BWAPI

Why? So that we can use the very nice TensorFlow python api in a native Unix environment for building StarCraft AIs.

We've written a python client for the TorchCraft C++ server:
- Network IO is generally in [tc_client.py](tc_client.py).
- The TorchCraft server returns the game-state as a Lua object string, which we transform into a python object string and ```eval()``` [here](https://github.com/aedanpope/tc_client/blob/427aafc9aa5dce7561325e74c64f4e8a13905e5e/tc_client.py#L254).
- [state.py](state.py) is responsible for parsing the responses from the server and turning them into typed obejects.


### 2. StarCraft TensorFlow agent examples

The existing best-class environment for StarCraft research is TorchCraft (used in [[1](https://arxiv.org/abs/1609.02993), [2](https://arxiv.org/abs/1702.08887)]).

TensorFlow support for BWAPI makes writing machine learning agents accessible to everyone who knows TensorFlow, and is a solid long-term investment as TensorFlow grows in popularity and functionality. For example, TensorFlow's [github](https://github.com/tensorflow/tensorflow) has ~10x more commits than [Torch's](https://github.com/torch/torch7).

For example, [Juliani's Q-Learing Part 0 blog post](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.icolg93n8) cointains sample TensorFlow code for a simple Q-network, which we've implemented to control a starcraft agent in [bot_q_learner_simple_a.py](bot_q_learner_simple_a.py) (the relevant TensorFlow code is [here](https://github.com/aedanpope/tc_client/blob/728ac6b889b1aa702ecea65a7a49bdb99d2625cd/bot_q_learner_simple_a.py#L127)).

### 3. A DQN for solving a difficult binary-success problem in StarCraft (that is, a DQN for a kiting micro-battle).

#### Abstract

Recent research into Reinforcement learning has used _micromanagement battles_ in the the real-time strategy game StarCraft as benchmarks for reinforcement learning algorithms.

Historically, the battles have been between symmetrical forces. Giving no orders in these battles (and thus defaulting to self-defence) can still result in a win 80%+ of the time. Many micro problems in StarCraft require very specific actions to have any chance of success.

When pitting a fast ranged unit vs. a slow melee one in StarCraft, the optimal control strategy for the ranged unit is to _Kite_ the melee one (fire from range, dance backwards before the melee unit can attack, and fire again). Giving no orders in this Battle to the ranged unit is a guaranteed loss.

We construct a simplified battle where kiting is the optimal stragey, and where randomly generating orders from a small but sufficient set results in nominal win rates (~1%). Then we show that a relatively generic DQN is able to learn to solve this battle with two key modifications:

- A much shorter experience buffer size of recent actions to re-train.
- Separate experience buffers for experience in battles which were won or lost.


Human Expert winning the micro battle: https://www.youtube.com/watch?v=PnEhLxpL29U

The DQN learning an optimal strategy over time: https://www.youtube.com/watch?v=UHgK2RxLCKM


#### Introduction
Recent AI research using Q-networks for StarCraft micro has looked primarily at symmetrical battles of groups of Marines. In particular, Marine 5v5 where the opponent just attack-moves the AI controlled 5 marines was studied in [Usunier et al, 2016](https://arxiv.org/abs/1609.02993) and [Foerster et al, 2017](https://arxiv.org/abs/1702.08887). This is a good environment for exploring the multi-agent problem, and extending models to the stochasitc high-dimensional StarCraft space.

There are many challenges to the StarCraft environment that marine 5v5 does not expose us to:
- Exploration complexity: Foerster et al. measure that giving _no_ orders to the friendly controlled marines leads to a win rate of 84% (Foerster, 2017), this means that agents have a decent default policy to iteratively learn and improve from.
- Asymmetry: One of the interesting AI challenges of StarCraft is the asymmetry in micro battles and races. There are 3 distint races with very different units and technologies available, leading to 9 different roles a completely general StarCraft AI has to learn (i.e. controlling {Terran, Zerg, Protos} vs enemy {Terran, Zerg, Protoss} is 9 distinct problems to solve with many common features).
- Planning: In marine 5v5, short term gains are a close proxy for long term victory. Doing some extra damage to the opponent this frame is probably good. In both (Foerster et al., 2017) and (Usunier et al., 2016), the agents are rewarded for dealing more damage to the opponent than they take in a timestep. However, In StarCraft it's common to either:
  - Make short term sacrifices to realise a longer term advantage.
  - Realise a short term advantage, without it being clear you've incurred a longer term disadvantage (you have dealt damage to an opponent but moved yourself into a position where you will take a lot more damage shortly).

Consider a battle a [Terran Vulture](http://wiki.teamliquid.net/starcraft/Vulture) and a [Protoss Zealot](http://wiki.teamliquid.net/starcraft/Zealot).

The vulture is a fast, fragile unit with a ranged attack. The zealot is a strong slow-moving unit with a melee attack. If a vulture and a zealot simply move directly to attack each other - the zealot will win.

In professional StarCraft, it is commonly accepted that vultures beat zealots - because expert players will [micro](http://wiki.teamliquid.net/starcraft2/Micro_(StarCraft)#Battle_micro) the vultures to hit the zealot once from range, then dance back before the zealot can attack and hit it again. This techniqe of "dancing back" is called [kiting](http://wiki.teamliquid.net/starcraft2/Kiting). Here is a video of a human controlling a Vulture to consistently beat a Zealot by kiting: https://www.youtube.com/watch?v=PnEhLxpL29U

A kiting micro battle is:
- hard to win with null or random actions
- asymmetrical
- has a slightly longer planning horizion than Marine 5v5

There has been [some previous research](https://scholar.google.co.uk/scholar?hl=en&q=starcraft+kiting&btnG=&as_sdt=1%2C5&as_sdtp=) into kiting in StarCraft using machine learning. Notably (Sukhbaatar et al., 2016), where a generalized gaming agent was able to develope highly successful kiting strategies with reinforcement learning.

In this project we consider the Kiting problem as a exercise to:
- Demonstrating the usefulness of the TensorFlow integration into BWAPI.
- See if a generic Deep Q-learning network can solve the kiting problem (that is, a network largely like that described in (Mnih et al, 2015).


#### Environment and Parameterization

Our environment consits of a simplified 1v1 battle between a vulture and a zealot.

- The hit points and damage of the units are modified such that the zealot kills the vulture in one attack, and the vulture kills the zealot in _n_ attacks - where _n_ is a parameter in ℤ<sup>+</sup>. For a particular _n_, we call the environment "_n_-kite". We chose n=2 going forward.

- Zealot: the enemy unit controlled by the environment, ordered to attack directly at the vulture.
- Vulture: the unit controlled by our agent.
- Starting Positions: The two opposing units start close enough that if the vulture also simply attacks the zealot, the zealot will reach it and attack at melee range after the vulture fires one shot. The vulture must move to survive and win the battle.

- Our vulture has 20 seconds to win the battle, otherwise the zealot wins by default.

We parameterise the enivornment for input into the network as follows (see [agent.py](agent.py)):

- Each timestep for the agent consists of 5 frames of the game.

- For both units
  - x and y positional coordinates (normalized in a bounding box)
  - current life of the unit
  - [cooldown](http://wiki.teamliquid.net/starcraft/Game_Speed#Cooldown) of the unit's attack, that is how long until it's attack is ready.

- For the agent controlled vulture, we also pass a one-hot vector of what order type the unit is currently following from {Guard, Move, Attack}.
- All parameters are normalized to the range [0,1]. x and y co-ordinates are normalized within a bounding box.
- We pass the values of the above for the current & previous timesteps (so that the network can determine velocity of both units).

The output of the agent is a one-hot vector representing 6 possible orders:
- Give no order, so the vulture continues with its current order.
- Give an order to move 6 tiles in 4 possible directions {up, down, left right}. It will take more than one timestep to finish moving the 6 tiles.
- Give an order to attack the enemy zealot. The vulture will move towards the zealot until it is in range, then fire, and pursue indefinitely.

If the vulture is currently standing still, then it defaults to the "guard" order, meaning that it will attack and pursue any enemy that comes within range.


#### Algorithm

We use a deep neural network to approximate the optimal action-value function.

_Q(s,a)_ = reward for taking action _a_ in state _s_ plus all discounted future reward (so the best action a in a state s is the one such that Q(s,a) is maximum)

Network Topology:
- 23 input nodes
- fully connected hidden layer of size 200, activation function ReLU
- fully connected hidden layer of size 300, activation function ReLU
- 6 output nodes, activation function TanH


Pseudocode:

```
Initialize win experience buffer W to capacity N_w
Initialize lose experience buffer L to capacity N_l
Initialize action-value function Q with random weights θ
Initialize target action-value function Q_2 with random weights θ_2

For battle 1, ... do
  Initialise battle buffer E
  while battle_not_over do
    read state s from battle
    if (battle <= K)
      select random action a
    else
      with probability ε
        select a = random choice of a_i with probability_weight softmax(Q(s, a_i, θ)/0.5)
      otherwise select a = argmax_a_i Q(s, a_i, θ)
    execute a on battle
    read state s1 from battle
    set r to 1 if battle is won, -1 if lost, 0 otherwise.
    store transition (s, a, r, s1) in E
    if (battle > K)
      select B = T/2 random samples each from W and L
      For each experience (s, a, r, s1) in B
        set a1 = argmax_a_i Q(s1, a_i, θ)
        set q1 = Q_2(s1, a1, θ_2)
        set y = r + γ * q1
        Update θ for loss (y - Q(s, a, θ))^2 with learning rate λ using Adam gradient-descent algorithm
        Set θ_2 = τ*θ + (1-τ)*θ_2
  End For
  if battle was won
    append E to W
  else
    append E to L
End For
```

Hyperparameters

| Variable   | Hyperparameter | value |
|---- | ------------------- | --- |
| T   | training batch size | 100 |
| λ | learning rate | 0.01 |
| N<sub>W</sub> | win experience buffer size | 5000 |
| N<sub>L</sub> | lose experience buffer size | 5000 |
| K | replay start size | 100000 |
| ε | exploration rate (see below) | 0.2 |
| γ | future reward discount | 0.99 |
| τ | target network update rate | 0.001 |

We use the Adam gradient-descent algorithm for training the network ([Kingma et. al., 2014](https://arxiv.org/abs/1412.6980)). Also see TensorFlow [tf.train.AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer).


#### Results

After pre-training for 100k timesteps, the network is generally able to learn an optimal strategy in 20-25k steps:

https://www.youtube.com/watch?v=UHgK2RxLCKM


Win rates by testing the best guess of the network from 100 battles * 5 trials after a given number of training steps:

| Training Steps | Win rate |
| --- | --- |
| 5k | 0.405 |
| 10k | 0.362 |
| 15k | 0.334 |
| 20k | 0.434 |

TODO run this experiment again with more trials, results are super noisy wiht occasional zeros 20k = (0.27, 0.7, 0.0, 0.69, 0.51) but the sample we recorded (in a single trial) for the video was 1.0 here.


##### Rewards

We give the agent a binary +1 or -1 reward at the end of the battle depending on if it wins or loses. This is the only reward the agent gets, so it is encouraged to plan ahead for long term reward.

Unlike (Foerster et al., 2017) and (Usunier et al., 2016), we do not give the agent any partial rewards for dealing more damage to the opponent than they take in a given timestep. We originally experimented with this, but found that the agent would get stuck in greedy local optima, where it would fire at the zealot but not run away in time - happy with the small reward it had incurred. Attempting to explicitly reward incurring damage in one timestep and not suffering damage within a set number of future timesteps trained the agent to be too risk adverse, not moving in to kill off the opponent.


##### Experience Buffers and Exploration

Randomly choosing actions from the 6 possible commands results in a win-rate of ~1.1%. This means that the traditional DQN with a single experience buffer is training examples with -ve reward the vast majority of the time. In experiments with this strategy, the agents would never learn how to win - just how to delay defeat for as long as possible.

*Two Buffers*

By storing experience from won and loss battles in separate buffers, and choosing half of the training sample from each, we ensure that the agent is able to learn some positive behaviours as well as negative.

*Shorter experience buffer*

The DQNs from (Mnih et al., 2015) and (Lillicrap et al., 2015) use an experience buffer of 10^6 time steps. We experimented with a buffer of size 10^5 but found the agent unable to develop a strong winning strategy.

Consider that in our case, the successful strategies in battles won in random exploration are likely to involve a lot of chaotic action and large idle time periods of the agent not firing (e.g. ~50 timesteps), whereas the optimal strategy of a human expert is to fire as soon as there is enough distance from the enemy and the weapon is off of cooldown (e.g. finishing a battle in 10 timesteps).

Intuitively, a shorter buffer (10k in our case) allows the agent to focus on learning recent successful wins, and learning recent mistakes made. This allows the agent to quickly learn to improve once it does find a successful strategy. It can try variants of that strategy, and with the short buffer then is likely to train the variants that failed a lot. So it is able to make small adjustments to a winning strategy until it is optimised.

We set the initial replay dataset to 10^5 timesteps, this generally resulted in about 4000 timesteps of positive reward - which would almost fill the Win Buffer of length 5000, which was a health start to then begin training from.

It seems like a good choice of Win+Loss buffer size for this technique is one that holds ~100-500 episodes of data (each of our battles is ~10-50 timesteps).

*Fixed ε=0.2 value*

Other DQN examples in the literature anneal ε down from 1 to a small value, commonly 0.1. In our case there's not much point at exploring with high ε values once we have the initial random-action replay dataset collected - instead it's faster to just train the positive examples we have from the initial exploration, and then start learning what variants of those examples fail.

Conversely, ε=0.2 is a good choice for this particular environment since the vulture's weapon cools down in about 5 timesteps - and so it makes sense to try one deviation from the known strategy with this cadence. If ε is smaller then convergence to a winning strategy takes longer, if ε is larger then the bot struggles to develop a winning strategy at all as it's always losing from the chaos (The bot requires at least 10 timesteps to win, and can instantly lose if 1-2 of the actions in these timesteps are bad).


#### Conclusions

These modifications to a generic DQN:
- separate Win+Loss experience buffers
- buffer size approx 100-500 episodes (battles)
- fixed ε=0.2 (or likely more generally, a non-trivial value for ε that we don't waste time annealing down to)
- 10^5 random exploration before training begins (enough to fill the Win experience buffer)

Mean that our algorithm is able to repeatably and successfully solve a problem with a 1% random success rate.


#### Future Work

- See if we can extend the algorithm to solve a harder kiting battle. Recall that _n_ is the environment parameter of the number of hits that the vulture requires to kill the zealot (and the zealot always kills the vulture in one hit). Perhapse the same algorithm can solve 3-kite or 4-kite. It's unlikely it will succeed for high values of n, as the amount of random exploration required to fill the win buffer will exponentially increase (it takes ~5 timesteps to fire once at the zealot and dance back, so the number of good random choices required for victory is O(n), and the probability of making enough good choices to win is then O(p^n)). Maybe: we can learn the beginnings of a successful strategy for, say n=4, by training an agent against n=2 and then slowly introducing n=3 and n=4 episodes into the conflict. Or by evenly cycling between battles of n=2,3,4 (only resulting in linear growth of the amount of initial exploration required to fill the win buffer).

- Find other 1%-random-success problems and see if these modifications can solve problems of this nature in general.

- Combine this work with (Foerster et al., 2017)'s on multi-unit micro battles in StarCraft to see if we can win multi-unit kiting battles (e.g. 5 Vultures vs. 5 Zealots).


#### Implementation


- [dqn_bot.py](dqn_bot.py): The DQN network implementation in TensorFlow.
- [exercise.py](exercise.py): ```main()``` function to run multiple trials on multiple hyperparameter configurations, and interface between the agent and [tc_client.py](tc_client.py).
- [agent.py](agent.py): Generic StarCraft micro-battle agent code for recording the game state and parameterising it.


#### References

Foerster, Nardelli, Farquhar, H. S. Torr, Kohli, and Whiteson. _Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning_. arXiv preprint [arXiv:1702.08887](https://arxiv.org/abs/1702.08887), 2017.

Mnih, Kavukcuoglu, Silver, Rusu, Veness, Bellemare, Graves, Riedmiller, Fidjeland, Ostrovski, Petersen, Beattie, Sadik, Antonoglou, King, Kumaran, Wierstra, Legg, Hassabis. _Human level control through deep reinforcement learning_. Nature, 518(7540):529–533, 2015.

Pritzel, Uria, Srinivasan, Puigdomènech, Vinyals, Hassabis, Wierstra, and Charles Blundell. _Neural Episodic Control_. arXiv preprint [arXiv:1703.01988](https://arxiv.org/abs/1703.01988), 2017.

Usunier, Synnaeve, Lin, and Chintala. _Episodic Exploration for Deep Deterministic Policies: An Application to StarCraft Micromanagement Tasks_. arXiv preprint [arXiv:1609.02993](https://arxiv.org/abs/1609.02993), 2016.

Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver, Wierstra. _Continuous control with deep reinforcement learning_. arXiv preprint [arXiv:1509.02971](https://arxiv.org/abs/1509.02971), 2015.

Sukhbaatar, Szlam, Synnaeve, Chintala, and Fergus. _MazeBase: A Sandbox for Learning from Games_. arXiv preprint [arXiv:1511.07401](https://arxiv.org/abs/1511.07401), 2016.

([Kingma et. al., 2014](https://arxiv.org/abs/1412.6980)) ([tensorflow](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer))
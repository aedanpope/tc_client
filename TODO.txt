To do:

randomize starting location

build a single map with kite 2/3/4 in it,
- maybe by using hero units
- randomly/serially rotate through them
- make the heros just have more HP, so input is the same and we don't need 3 unit types.
- can make 1..N kite just by setting enemy hp

build a single map which uses different kite unit types, e.g. zerglings, firebats, that have different movement speeds.

Build a harness for experimentally evaluating hyperparameter changes:
- try these 5 values for some hyperparameter, e.g. Hidden layer sizes.
- single py script runs say, 2000 battles so that E = end_e, then runs 100 battles to evaluate performance with e=0

Keep reading https://arxiv.org/pdf/1509.02971.pdf,
- try rectifier non-linearity for the hidden networks instead of ReLU
- Actor network separate from critic.
- try differnet learning rates for Actor and Critic (e.g. e-4 and e-3 as the paper does)
- figure out what "actions were not included until the 2nd hidden layer" means for us.
- try intializing n [−3 × 10−3, 3 × 10−3]
- add separate Value & Advantage networks.
- Try Ornstein-Uhlenbeck process for Noize (like Sec 7), since our env is kind of physical too.

Other things to try:
- decay training examples.
- dropout (if we use large hidden layers). Also see: Bayesian
- Recurrent DNQ https://arxiv.org/abs/1605.06676
- multi agent: https://arxiv.org/abs/1702.08887
- various exploration methods (e.g. Boltzmann, Bayesian)
- a separate network to predict next state (maybe of just opponent)
- apply negative reward for more time taken, so that it optimizes to win fast.
- try randomly rotating and reflecting the input training data , maybe just 4 directions or random degrees.

from map import Map

Mode = Map(
    train = 0, # Explore and train the network.
    test = 1, # Test the current optimal performance of the network.
)

class Settings:
  # How noisy to print debug output. 0 = none, 30 = max
  verbosity = None

  mode = Mode.train
  hyperparameters = None

# How noisy to print debug output. Higher = more noisy
VERBOSITY = 1

def log(v=10, msg):
  if VERBOSITY >= v:
    print msg

def should_log(v=10):
  return VERBOSITY >= v
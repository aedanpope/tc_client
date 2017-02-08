# Fakes for lua Torch objects.

class ByteTensor:

  foo = None

  def __init__(self, foo):
    self.foo = foo
    print "new ByteTensor"

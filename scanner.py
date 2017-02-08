import re

class Scanner:

  def __init__(self, s):
    self.s = s
    self.offset = 0

  def end(self):
    return self.offset == len(self.s)

  def _assert_not_end(self):
    if self.end():
      raise Exception("At end of scanner.")

  def peek_char(self):
    return self.next_char(advance=False)

  def next_char(self, advance=True):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    c = self.s[self.offset]
    if advance:
      self.offset += 1
    return c

  def next_word(self, advance=True):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    return self.next_pattern(r"\w+", advance=advance)

  def peek_word(self):
    return self.next_word(advance=False)

  def next_int(self):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    return int(self.next_pattern(r"-?\d+"))

  def next_float(self):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    return float(self.next_pattern(r"-?\d+\.?\d*"))

  def next_space(self):
    self._assert_not_end()
    return self.next_pattern(r"\s+")

  def consume_space(self):
    self._assert_not_end()

    while not self.end() and self.s[self.offset].isspace():
      self.offset += 1

  def next_pattern(self, pattern, flags=0, advance=True):
    self._assert_not_end()

    if isinstance(pattern, basestring):
      pattern = re.compile(pattern, flags)
    match = pattern.match(self.s, self.offset)
    if match is not None:
      if self.offset != match.start():
        raise Exception("Next token in scanner does not match pattern'" + pattern + "'.")

      if advance:
        self.offset = match.end()
      return match.group(0)
    return None


import re

class Scanner:

  def __init__(self, str):
    self.str = str
    self.offset = 0


  def end(self):
    return self.offset == len(self.str)


  def _assert_not_end(self):
    if self.end():
      raise Exception("At end of scanner.")


  def peek_char(self):
    return self.next_char(advance=False)


  def next_char(self, advance=True):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    c = self.str[self.offset]
    if advance:
      self.offset += 1
    return c


  def next_word(self, advance=True):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    return self.next_pattern(r"\w+", advance=advance)


  def has_next_word(self):
    return self.next_word(advance=False) is not None


  def peek_word(self):
    return self.next_word(advance=False)


  def next_int(self, advance=True):
    """ Consumes preceding whitespace. """
    self.consume_space()
    self._assert_not_end()
    token = self.next_pattern(r"-?\d+", advance=advance)
    if token is None: return None
    return int(token)


  def has_next_int(self):
    return self.next_int(advance=False) is not None


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

    while not self.end() and self.str[self.offset].isspace():
      self.offset += 1


  def next_pattern(self, pattern, flags=0, advance=True, strict=False):
    self._assert_not_end()

    if isinstance(pattern, basestring):
      pattern = re.compile(pattern, flags)
    match = pattern.match(self.str, self.offset)
    if match is None or match.start() != self.offset:
      if strict:
        raise Exception("Next token in scanner does not match pattern'" + pattern + "'.")
      else:
        return None
    if advance:
      self.offset = match.end()
    return match.group(0)



import unittest
from tf_bot import TFBot
import re
import pytest

class TFBotTest(unittest.TestCase):

  def test_norm(self):
    self.assertAlmostEqual(TFBot.norm(98, 70, 150), -0.30)
    self.assertAlmostEqual(TFBot.norm(40, 0, 40), 1.0)
    self.assertAlmostEqual(TFBot.norm(33, 0, 40), 0.65)

if __name__ == '__main__':
    unittest.main()
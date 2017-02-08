import unittest
from scanner import Scanner
import re

class ScannerTest(unittest.TestCase):


  def test_scanner(self):
    scanner = Scanner("asd 3434 = [] -0.31 -422.31")
    self.assertEqual(scanner.next_char(), "a")
    self.assertEqual(scanner.next_word(), "sd")
    self.assertEqual(scanner.next_int(), 3434)
    self.assertEqual(scanner.peek_char(), "=")
    self.assertEqual(scanner.next_char(), "=")
    self.assertEqual(scanner.next_char(), "[")
    self.assertEqual(scanner.next_char(), "]")
    self.assertEqual(scanner.next_float(), -0.31)
    self.assertEqual(scanner.next_float(), -422.31)
    self.assertEqual(scanner.end(), True)

  def test_re(self):
    msg = ("{lag_frames = 2,map_data = torch.ByteTensor({{2,2,2,2,-1,-1,},}),"
           "map_name = 'm5v5_c_far.scm',"
           "frame = [[1 2 0 0 33 13]],"
           "is_replay = false,player_id = 0,"
           "neutral_id = -1,}")

    msg = msg[1:-1]
    msg = msg.replace(",}", "}")
    msg = msg.replace("}", "]")
    msg = msg.replace("{", "[")
    msg = msg.replace("false", "False")
    msg = msg.replace("true", "True")

    rx = r"frame\s*=\s*\[\[([-?\d\.?\d*\s]*)\]\]"

    foo = "frame= [[11 -0.203125 22 -1 00]], cats=3"
    print "foo = '" + re.sub(rx, "frame=\'\\1\'", foo) + "'"

    self.assertEqual(re.sub(rx, "frame = \'\\1\'", foo),
                     "frame = '11 -0.203125 22 -1 00', cats=3")

    long_foo = "frame=[[2 0 19 0 224 -0.203125 160 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1792 1280 27 31 0 0 5 5 0 0 1 121 23 -1 224 160 0 0 0 0 1 32 32 1 1 0 0 0 0 0 0 1 1 101 0 0 0 256 256 27 31 0 0 5 5 0 0 1 121 23 -1 32 32 0 0 0 0 2 224 224 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1792 1792 27 31 0 0 5 5 0 0 1 121 23 -1 224 224 0 0 0 0 3 32 224 1 1 0 0 0 0 0 0 1 1 101 0 0 0 256 1792 27 31 0 0 5 5 0 0 1 121 23 -1 32 224 0 0 0 0 4 160 224 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1280 1792 27 31 0 0 5 5 0 0 1 121 23 -1 160 224 0 0 0 0 5 96 224 1 1 0 0 0 0 0 0 1 1 101 0 0 0 768 1792 27 31 0 0 5 5 0 0 1 121 23 -1 96 224 0 0 0 0 6 160 160 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1280 1280 27 31 0 0 5 5 0 0 1 121 23 -1 160 160 0 0 0 0 7 96 160 1 1 0 0 0 0 0 0 1 1 101 0 0 0 768 1280 27 31 0 0 5 5 0 0 1 121 23 -1 96 160 0 0 0 0 8 32 160 1 1 0 0 0 0 0 0 1 1 101 0 0 0 256 1280 27 31 0 0 5 5 0 0 1 121 23 -1 32 160 0 0 0 0 9 224 96 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1792 768 27 31 0 0 5 5 0 0 1 121 23 -1 224 96 0 0 0 0 10 160 96 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1280 768 27 31 0 0 5 5 0 0 1 121 23 -1 160 96 0 0 0 0 17 80 138 40 40 0 0 0 15 1 1 0 1 0 0 0 1 640 1104 17 20 6 6 3 3 16 16 1 121 10 21 80 138 0 0 0 0 11 96 96 1 1 0 0 0 0 0 0 1 1 101 0 0 0 768 768 27 31 0 0 5 5 0 0 1 121 23 -1 96 96 0 0 0 0 12 32 96 1 1 0 0 0 0 0 0 1 1 101 0 0 0 256 768 27 31 0 0 5 5 0 0 1 121 23 -1 32 96 0 0 0 0 13 224 32 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1792 256 27 31 0 0 5 5 0 0 1 121 23 -1 224 32 0 0 0 0 14 160 32 1 1 0 0 0 0 0 0 1 1 101 0 0 0 1280 256 27 31 0 0 5 5 0 0 1 121 23 -1 160 32 0 0 0 0 15 96 32 1 1 0 0 0 0 0 0 1 1 101 0 0 0 768 256 27 31 0 0 5 5 0 0 1 121 23 -1 96 32 0 0 0 0 23 83 138 40 40 0 0 0 15 13 13 0 1 0 0 0 1 664 1104 17 20 6 6 3 3 16 16 1 121 10 20 83 138 0 0 0 0 24 83 141 10 40 0 0 0 15 7 7 0 1 0 0 0 1 664 1128 17 20 6 6 3 3 16 16 1 121 10 20 83 141 0 0 0 0 1 3 19 96 136 40 40 0 0 0 15 8 8 0 1 0 0 0 1 772 1094 17 20 6 6 3 3 16 16 1 121 10 24 96 136 0 0 1 0 20 97 142 16 40 0 0 0 15 9 9 0 1 0 0 0 1 782 1140 17 20 6 6 3 3 16 16 1 121 10 24 97 142 0 0 1 0 21 91 137 28 40 0 0 0 15 6 6 0 1 0 0 0 1 728 1099 17 20 6 6 3 3 16 16 1 121 10 24 91 137 0 0 1 0 0 1 0 0 0 6 0 7 142 93 135 142 97 142 142 97 142 142 83 140 142 83 141 142 97 142 142 83 140 0 0]],deaths=[],frame_from_bwapi = 125,battle_frame_count = 5,"

    print "long_foo = '" + re.sub(rx, "frame=\'\\1\'", long_foo) + "'"

    # line = re.sub(r"</?\[\d+>", "", line)

    # scanner = Scanner("asd 3434 = []")
    # self.assertEqual(scanner.next_char(), "a")
    # self.assertEqual(scanner.next_word(), "sd")
    # self.assertEqual(scanner.next_int(), 3434)
    # self.assertEqual(scanner.peek_char(), "=")
    # self.assertEqual(scanner.next_char(), "=")
    # self.assertEqual(scanner.next_char(), "[")
    # self.assertEqual(scanner.next_char(), "]")
    # self.assertEqual(scanner.end(), True)


if __name__ == '__main__':
    unittest.main()
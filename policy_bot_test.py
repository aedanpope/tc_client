import unittest
import state
import copy
from policy_bot import Bot
from policy_bot import Battle
from policy_bot import Stage
import re
from map import Map

class PolicyBotTest(unittest.TestCase):

  def test_battle_to_input(self):
    f0 = state.Unit()
    f0.id = 11
    f0.health = 10
    f0.max_health = 40
    f0.shield = 0
    f0.max_shield = 0
    f0.x = 50
    f0.y = 102
    f0.groundCD = 50
    f0.maxCD = 100
    f0.orders = [state.Order()]
    f1 = copy.copy(f0)
    f1.health = 5
    f1.x += 10
    f1.y += 24
    f1.groundCD = 40

    e0 = state.Unit()
    e0.id = 22
    e0.health = 8
    e0.max_health = 64
    e0.shield = 0
    e0.max_shield = 0
    e0.x = 110
    e0.y = 150
    e0.groundCD = 20
    e0.maxCD = 100
    e1 = copy.copy(e0)
    e1.health = 16
    e1.x += 5
    e1.y += 12
    e1.groundCD = 10

    s0 = state.State(Map())
    s0.friendly_units = {f0.id : f0}
    s0.enemy_units = {e0.id : e0}
    s1 = state.State(Map())
    s1.friendly_units = {f1.id : f1}
    s1.enemy_units = {e1.id : e1}

    battle = Battle()
    battle.add_stage(Stage(s0))
    battle.add_stage(Stage(s1))

    self.assertEqual(Bot.battle_to_input(battle),
        [[0.1, 0.1, 0.25, 0.5, 0, 0, 0, 0.2, 0.3, 0.125, 0.4, 0, 0, 0, 0.7, 0.5, 0.125, 0.2, 0.75, 0.6, 0.25, 0.1]])


if __name__ == '__main__':
    unittest.main()
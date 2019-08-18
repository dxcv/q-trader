#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:59:24 2019

@author: igor
"""

import unittest

import knight as k

class TestKnight(unittest.TestCase):

    def test_get_best_path(self):
        t = k.get_best_path('A1', 'H8')

        self.assertEqual(t.get_path_str(), 'A1 C2 E3 G4 E5 G6 H8')


if __name__ == '__main__':
    unittest.main()
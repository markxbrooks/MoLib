"""Tests for Coordinates."""

import numpy as np
from unittest import TestCase

from molib.pdb.coordinate.coordinate import Coordinates


class CoordinatesTests(TestCase):
    def test_from_array_numpy(self):
        coord = Coordinates.from_array(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(coord, Coordinates(1.0, 2.0, 3.0))

    def test_from_array_tuple(self):
        coord = Coordinates.from_array((4.0, 5.0, 6.0))
        self.assertEqual(coord.as_tuple(), (4.0, 5.0, 6.0))

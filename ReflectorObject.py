# -*- coding:utf-8 -*-
import copy

import numpy as np

from functions import rotate_vector, unit_normal


class Reflector:
    """An ideal specular reflector in a single medium.
    """

    def __init__(self, normal=(0, 0, 1)):
        self.normal = normal

    def __repr__(self):
        return f"Reflector(normal={self.normal})"

    def calc_reflect_k(self, ray):
        """Reflect ray (K vector) in a plane, preserving the magnitude.
        """
        K = ray.K
        normal = unit_normal(self.normal)
        projection = np.sum(K * normal, axis=-1, keepdims=True)
        return K - 2 * projection * normal

    def reflect(self, ray):
        return ray.copy_with(K=self.calc_reflect_k(ray))

    def at_wavelength(self, wavelength=None):
        """An ideal mirror is achromatic; return an independent copy."""
        return copy.deepcopy(self)

    def calc_interact_k(self, ray, n_out=None, reverse=False):
        """Alias for calc_reflect_k."""
        return self.calc_reflect_k(ray)

    def interact(self, ray, n_out=None, reverse=False):
        """Alias for reflect."""
        return self.reflect(ray)

    @classmethod
    def from_rotation(cls, angles, base_normal=(0, 0, 1)):
        """Rotate the base normal about fixed x, then y, then z axes (degrees)."""
        return cls(normal=rotate_vector(unit_normal(base_normal), angles))


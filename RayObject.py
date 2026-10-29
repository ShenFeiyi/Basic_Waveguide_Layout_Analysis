# -*- coding:utf-8 -*-
import numpy as np

from functions import unit_normal

class Ray:
    """
    A class representing a ray in 3D space, defined by its wave vector K or field angle.
    Ray properties:
    - K: Wave vector (kx, ky, kz)
    - theta: Elevation angle in degrees
    - phi: Azimuth angle in degrees
    - field_angle: Field angle (theta_x, theta_y) in degrees, relative to the surface normal, catesian coordinates;
      measured from the +z or -z axis the ray travels along, so its sign follows (kx, ky)
    - n: Refractive index of the medium
    - x, y, z: Position of the ray in 3D space, mm
    """
    def __init__(self, **kwargs):
        self.x = kwargs.get('x', 0) # position x, mm
        self.y = kwargs.get('y', 0) # position y, mm
        self.z = kwargs.get('z', 0) # position z, mm
        self.n = kwargs.get('n', 1.0) # refractive index

        if not "K" in kwargs:
            # If K is not provided, calculate it from the field angle and refractive index
            self.field_angle = kwargs.get('field_angle', np.array([0, 0])) # field angle, cartesian, relative to surface normal

            self.theta_x, self.theta_y = self.field_angle[0], self.field_angle[1]
            self.theta = self.calc_elevation_angle() # elevation angle, degrees
            self.phi = self.calc_azimuth_angle() # azimuth angle, degrees
            self.kx = self.n * np.sin(np.radians(self.theta)) * np.cos(np.radians(self.phi))
            self.ky = self.n * np.sin(np.radians(self.theta)) * np.sin(np.radians(self.phi))
            self.kz = self.n * np.cos(np.radians(self.theta))
            self.K = np.array([self.kx, self.ky, self.kz]) # wave vector

        else:
            # If K is provided, calculate the field angle from it
            self.K = kwargs.get('K', np.array([0, 0, 1])) # wave vector

            self.kx, self.ky, self.kz = self.K[0], self.K[1], self.K[2]
            assert not self.kz == 0, "kz is zero, evanescent wave"
            # |kz| keeps the sign of a -z ray's field angle equal to the sign of its tilt.
            self.theta_x = np.degrees(np.arctan(self.kx / abs(self.kz)))
            self.theta_y = np.degrees(np.arctan(self.ky / abs(self.kz)))
            self.field_angle = np.array([self.theta_x, self.theta_y]) # field angle, cartesian, relative to surface normal
            self.theta = self.calc_elevation_angle() # elevation angle, degrees
            self.phi = self.calc_azimuth_angle() # azimuth angle, degrees

    def __repr__(self):
        return (
            f"Ray(K={self.K}, theta={self.theta:.2f}, phi={self.phi:.2f},"
            f"field_angle={self.field_angle}, n={self.n},"
            f"position=({self.x}, {self.y}, {self.z}))"
            )

    @property
    def direction(self):
        """Unit propagation direction, including its signed z component."""
        return self.K / np.linalg.norm(self.K)

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    def copy_with(self, K=None, n=None, position=None):
        """Return a new ray with selected properties replaced.
        """
        new_n = self.n if n is None else n
        if K is None:
            K = self.K.copy() if n is None else self.direction * new_n
        x, y, z = self.position if position is None else position
        return Ray(K=K, n=new_n, x=x, y=y, z=z)

    def calc_elevation_angle(self):
        # Calculate the elevation angle of the ray in degrees
        _theta = np.sqrt(np.tan(np.radians(self.theta_x))**2 + np.tan(np.radians(self.theta_y))**2)
        _theta *= 1 if self.theta_x >= 0 else -1
        return np.degrees(np.arctan(_theta))

    def calc_azimuth_angle(self):
        # Calculate the azimuth angle of the ray in degrees
        if self.theta_x == 0:
            _phi = np.pi/2 if self.theta_y > 0 else -np.pi/2
        else:
            _phi = np.arctan(np.tan(np.radians(self.theta_y)) / np.tan(np.radians(self.theta_x)))
        return np.degrees(_phi)


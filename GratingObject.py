# -*- coding:utf-8 -*-
import numpy as np

from RayObject import Ray


class Grating:
    def __init__(self, **kwargs):
        self.wavelength = kwargs.get('wavelength', 550) # wavelength, nm
        self.grating_period = kwargs.get('grating_period', 1000) # grating period, nm
        self.diffraction_order = kwargs.get('diffraction_order', 1) # diffraction order
        self.grating_orientation = kwargs.get('grating_orientation', 0) # grating vector orientation angles, degrees

    def __repr__(self):
        return (
            f"Grating(wavelength={self.wavelength},"
            f"grating_period={self.grating_period},"
            f"diffraction_order={self.diffraction_order},"
            f"grating_orientation={self.grating_orientation})"
        )

    @property
    def M(self):
        return self.diffraction_order

    @property
    def PHI(self):
        return self.grating_orientation

    @property
    def K(self):
        # Grating vector, (kx, ky)
        return self.M * self.wavelength / self.grating_period * np.array([np.cos(np.radians(self.PHI)), np.sin(np.radians(self.PHI))])

    def calc_diffract_k(self, ray, n_out, reverse=False):
        """Return the candidate wave vector after this grating.
        reverse=True subtracts the grating vector, as used when tracing backward from the output.
        """
        grating_k = self.K
        direction = -1 if reverse else 1
        kx = ray.K[0] + direction * grating_k[0]
        ky = ray.K[1] + direction * grating_k[1]
        kz_squared = n_out**2 - kx**2 - ky**2
        kz = np.sqrt(kz_squared) if kz_squared >= 0 else 0
        return np.array([kx, ky, kz])

    def diffract(self, ray, n_out, reverse=False):
        """Create a diffracted Ray, retaining its position.
        Non-propagating grating candidates are rejected here.
        """
        K = self.calc_diffract_k(ray, n_out, reverse=reverse)
        self.validate_candidate(K)
        return Ray(K=K, n=n_out, x=ray.x, y=ray.y, z=ray.z)

    @staticmethod
    def validate_candidate(K):
        if K[2] == 0:
            raise AssertionError("Ray not propagating after diffraction.")

    def calc_interact_k(self, ray, n_out, reverse=False):
        """Alias for calc_diffract_k."""
        return self.calc_diffract_k(ray, n_out, reverse=reverse)

    def interact(self, ray, n_out, reverse=False):
        """Alias for diffract."""
        return self.diffract(ray, n_out, reverse=reverse)


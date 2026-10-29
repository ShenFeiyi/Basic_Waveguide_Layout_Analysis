# -*- coding:utf-8 -*-
import warnings

import numpy as np


class Material:
    """Refractive index as a function of vacuum wavelength in nm.

    Build one with:
    - Material(1.5): constant (non-dispersive) index
    - Material(callable): any n(wavelength_nm) function
    - Material.from_table({450: 1.971, 550: 1.9310, 650: 1.9107}): measured or catalog values
    - Material.sellmeier(B, C): Sellmeier coefficients, wavelength in um, C in um^2
    - Material.cauchy(A, B, C): n = A + B/lambda^2 + C/lambda^4, wavelength in um
    - Material.from_abbe(nd, vd): two-term Cauchy estimate from nd and Abbe number
    """

    def __init__(self, index=1.0, name=None):
        if callable(index):
            self._function = index
            self.constant = None
        else:
            value = float(index)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Refractive index must be positive and finite, got {index!r}.")
            self._function = None
            self.constant = value
        self.name = name

    def __repr__(self):
        if self.constant is not None:
            return f"Material(n={self.constant})"
        return f"Material({self.name or 'dispersive'})"

    @property
    def is_dispersive(self):
        return self.constant is None

    def __call__(self, wavelength):
        """Return the index at a vacuum wavelength in nm."""
        if self.constant is not None:
            return self.constant
        if wavelength is None:
            raise ValueError(f"A wavelength is required to evaluate the dispersive index {self!r}.")
        if not wavelength > 0:
            raise ValueError(f"Wavelength must be positive, got {wavelength!r} nm.")
        n = float(self._function(float(wavelength)))
        if not np.isfinite(n) or n <= 0:
            raise ValueError(f"{self!r} gives an invalid index {n} at {wavelength} nm.")
        return n

    @classmethod
    def create(cls, spec):
        """Accept a Material, a number, a {wavelength_nm: index} mapping, or a callable."""
        if isinstance(spec, Material):
            return spec
        if isinstance(spec, dict):
            return cls.from_table(spec)
        return cls(spec)

    @classmethod
    def from_table(cls, table, name=None):
        """Interpolate tabulated indices linearly in 1/wavelength^2.

        Between two table points this is an exact two-term Cauchy fit. Outside the
        table the end segments are extended, with a warning.
        """
        if len(table) == 0:
            raise ValueError("A dispersion table needs at least one wavelength.")
        pairs = sorted((float(w), float(n)) for w, n in table.items())
        wavelengths, indices = (np.array(column) for column in zip(*pairs))
        if np.any(np.diff(wavelengths) == 0):
            raise ValueError(f"Duplicate wavelengths in the dispersion table {table!r}.")
        if np.any(wavelengths <= 0) or np.any(~np.isfinite(indices)) or np.any(indices <= 0):
            raise ValueError(f"Invalid dispersion table {table!r}.")
        label = name or "table " + ", ".join(f"{w:g} nm: {n:g}" for w, n in zip(wavelengths, indices))
        if len(wavelengths) == 1:
            return cls(indices[0], name=label)
        x = 1 / wavelengths[::-1]**2 # increasing
        n_x = indices[::-1]

        def index(wavelength):
            if not wavelengths[0] <= wavelength <= wavelengths[-1]:
                warnings.warn(
                    f"Extrapolating the dispersion table ({wavelengths[0]:g}-{wavelengths[-1]:g} nm) "
                    f"to {wavelength:g} nm.", stacklevel=3,
                )
            xq = 1 / wavelength**2
            j = int(np.clip(np.searchsorted(x, xq), 1, len(x) - 1))
            return n_x[j-1] + (n_x[j] - n_x[j-1]) * (xq - x[j-1]) / (x[j] - x[j-1])

        return cls(index, name=label)

    @classmethod
    def sellmeier(cls, B, C, name=None):
        """n^2 = 1 + sum(B_i * lambda^2 / (lambda^2 - C_i)), lambda in um, C_i in um^2."""
        B, C = np.asarray(B, dtype=float), np.asarray(C, dtype=float)
        if B.shape != C.shape:
            raise ValueError("Sellmeier B and C need the same length.")

        def index(wavelength):
            lam2 = (wavelength / 1000)**2
            return np.sqrt(1 + np.sum(B * lam2 / (lam2 - C)))

        return cls(index, name=name or "Sellmeier")

    @classmethod
    def cauchy(cls, A, B=0.0, C=0.0, name=None):
        """n = A + B/lambda^2 + C/lambda^4, lambda in um."""
        def index(wavelength):
            lam2 = (wavelength / 1000)**2
            return A + B / lam2 + C / lam2**2

        return cls(index, name=name or f"Cauchy A={A:g}, B={B:g}, C={C:g}")

    @classmethod
    def from_abbe(cls, nd, vd, name=None):
        """Two-term Cauchy estimate from nd (587.56 nm) and the Abbe number vd.

        Uses nF - nC = (nd - 1)/vd. Adequate for a first look; prefer catalog
        Sellmeier data for design work.
        """
        lam_d, lam_F, lam_C = 0.5875618, 0.4861327, 0.6562725 # um
        B = (nd - 1) / vd / (1 / lam_F**2 - 1 / lam_C**2)
        A = nd - B / lam_d**2
        return cls.cauchy(A, B, name=name or f"Abbe nd={nd:g}, vd={vd:g}")

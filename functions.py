# -*- coding:utf-8 -*-
import numpy as np


def unit_normal(normal):
    """Return a normalized real surface normal; either orientation is valid."""
    return normal / np.linalg.norm(normal)


def rotation_matrix_xyz(angles, degrees=True):
    """Right-handed extrinsic XYZ rotation: apply Rx, then Ry, then Rz."""
    alpha, beta, gamma = np.radians(angles) if degrees else angles
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate_vector(vector, angles, degrees=True):
    """Rotate vectors of shape (3,) or (..., 3), retaining their magnitudes."""
    return vector @ rotation_matrix_xyz(angles, degrees=degrees).T


class Line2D:
    """An infinite line through two distinct planar points."""

    def __init__(self, point1, point2):
        length = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        assert length > 0, "The two points must be distinct to define a line."
        self.point1 = point1
        self.point2 = point2
        self.x0, self.y0 = point1
        self.x1, self.y1 = point2
        self.x_step = self.x1 - self.x0
        self.y_step = self.y1 - self.y0
        # ax + by + c = 0; retain the original coefficient arithmetic.
        self.params = (
            self.y_step, -self.x_step,
            self.x_step * self.y0 - self.y_step * self.x0,
        )

    def __repr__(self):
        return "Line2D: {:.6f}X + {:.6f}Y + {:.6f} = 0".format(*self.params)

    def intersect(self, other, thres=1e-12):
        """Intersect with another line; reject parallel/ill-conditioned pairs."""
        a1, b1, c1 = self.params
        a2, b2, c2 = other.params
        det = a1 * b2 - a2 * b1
        scale = np.hypot(a1, b1) * np.hypot(a2, b2)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Intersection requires two valid lines.")
        if not np.isfinite(det) or abs(det) <= thres * scale:
            raise ValueError("Lines are parallel, coincident, or too nearly parallel for a stable intersection.")
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return float(x), float(y)


def intersection(line1, line2, thres=1e-12):
    """Intersect two Line2D objects; reject parallel/ill-conditioned pairs."""
    return line1.intersect(line2, thres=thres)




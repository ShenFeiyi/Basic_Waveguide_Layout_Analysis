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


def convex_hull(points):
    """Counter-clockwise convex hull of 2D points (Andrew's monotone chain)."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float).reshape(-1, 2))))
    if len(points) <= 2:
        return np.array(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(vertices):
    """Shoelace area of a simple polygon given in boundary order."""
    x, y = np.asarray(vertices, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def wavelength_to_rgb(wavelength):
    """Approximate plot color (sRGB, 0-1) for a vacuum wavelength in nm.

    Hue follows a piecewise-linear visible spectrum; luminance follows a fixed
    profile (greens lighter, blues and reds darker) so typical RGB sets stay
    distinguishable under red-green color blindness and keep >= 3:1 contrast on
    white. Closely spaced wavelengths still look alike. Outside 380-780 nm: gray.
    """
    w = float(wavelength)
    if not 380 <= w <= 780:
        return (0.5, 0.5, 0.5)
    if w < 440:
        rgb = [(440 - w) / 60, 0.0, 1.0]
    elif w < 490:
        rgb = [0.0, (w - 440) / 50, 1.0]
    elif w < 510:
        rgb = [0.0, 1.0, (510 - w) / 20]
    elif w < 580:
        rgb = [(w - 510) / 70, 1.0, 0.0]
    elif w < 645:
        rgb = [1.0, (645 - w) / 65, 0.0]
    else:
        rgb = [1.0, 0.0, 0.0]
    rgb = np.array(rgb)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)**2.4)
    luminance = linear @ [0.2126, 0.7152, 0.0722]
    target = np.interp(w, [380, 450, 500, 540, 580, 620, 650, 780],
                       [0.05, 0.10, 0.24, 0.29, 0.20, 0.11, 0.10, 0.05])
    linear = linear * min(target / luminance, 1 / linear.max())
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0)**(1/2.4) - 0.055)
    return tuple(float(c) for c in srgb)




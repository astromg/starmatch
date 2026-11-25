"""Second-order polynomial coordinate transformation.

x' = px[0] + px[1]*x + px[2]*y + px[3]*x*y + px[4]*x*x + px[5]*y*y
y' = py[0] + py[1]*x + py[2]*y + py[3]*x*y + py[4]*x*x + py[5]*y*y
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_COEFFS = 6


def design_matrix(x, y) -> np.ndarray:
    """Return the (N, 6) matrix of polynomial terms [1, x, y, x*y, x^2, y^2]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])


def fit_poly(x, y, target, order: int = 2) -> np.ndarray:
    """Least-squares fit of ``target = poly(x, y)``; returns 6 coefficients.

    ``order=1`` fits only ``[1, x, y]`` (affine), the 2nd-order coefficients are then 0.
    """
    a = design_matrix(x, y)[:, : 3 if order == 1 else N_COEFFS]
    norm = np.linalg.norm(a, axis=0)
    norm[norm == 0] = 1
    coeffs, *_ = np.linalg.lstsq(a / norm, np.asarray(target, dtype=float), rcond=None)
    out = np.zeros(N_COEFFS)
    out[: a.shape[1]] = coeffs / norm
    return out


def fit_poly2(x, y, target) -> np.ndarray:
    """Least-squares fit of ``target = poly2(x, y)``; returns the 6 coefficients."""
    return fit_poly(x, y, target, order=2)


def coo_trans(x, y, px, py):
    """Apply the polynomial transformation to coordinates ``x, y``.

    ``x, y`` may be scalars or array-like; returns the same kind.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xt = px[0] + px[1] * x + px[2] * y + px[3] * x * y + px[4] * x * x + px[5] * y * y
    yt = py[0] + py[1] * x + py[2] * y + py[3] * x * y + py[4] * x * x + py[5] * y * y
    if scalar:
        return float(xt), float(yt)
    return xt, yt


@dataclass
class XYTransform:
    """Field <-> reference transformation coefficients (as stored in ``.xytr`` files)."""

    fr_x: np.ndarray  # x field -> reference
    fr_y: np.ndarray  # y field -> reference
    rf_x: np.ndarray  # x reference -> field
    rf_y: np.ndarray  # y reference -> field
    reference: str | None = None
    field: str | None = None

    def field_to_ref(self, x, y):
        """Transform field coordinates into the reference frame."""
        return coo_trans(x, y, self.fr_x, self.fr_y)

    def ref_to_field(self, x, y):
        """Transform reference coordinates into the field frame."""
        return coo_trans(x, y, self.rf_x, self.rf_y)

    @property
    def scale(self) -> float:
        """Reference pixels per field pixel (linear part, = pixscale field/reference)."""
        a, b, c, d = self.fr_x[1], self.fr_x[2], self.fr_y[1], self.fr_y[2]
        return float(np.sqrt(abs(a * d - b * c)))

    @property
    def rotation(self) -> float:
        """Rotation field -> reference [deg] (linear part)."""
        a, b, c, d = self.fr_x[1], self.fr_x[2], self.fr_y[1], self.fr_y[2]
        if a * d - b * c < 0:  # mirrored: measure rotation of the x axis
            return float(np.degrees(np.arctan2(c, a)))
        return float(np.degrees(np.arctan2(c - b, a + d)))

    @property
    def flipped(self) -> bool:
        """True if field and reference are mirror images."""
        return bool(self.fr_x[1] * self.fr_y[2] - self.fr_x[2] * self.fr_y[1] < 0)

    def __repr__(self):
        flip = ", flipped" if self.flipped else ""
        return f"XYTransform(scale={self.scale:.5f}, rotation={self.rotation:.3f} deg{flip})"

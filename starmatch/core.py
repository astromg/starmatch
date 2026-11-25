"""Star list matching: find the 2nd-order transformation between a field and a reference list.

Authors: Marek Gorski, Dominika Switlik

Pipeline (:func:`match`):

1. **Local features** - for the brightest stars, the neighbours within a radius are projected on a
   local frame (axis1 = direction to the brightest neighbour, axis2 = its normal), scaled by the
   distance to that neighbour. These projection arrays do not depend on scale or rotation.
   Every reference star gets the field star with the most similar feature as a candidate.
2. **Consensus** - every pair of candidates defines a similarity transform (shift, rotation, scale).
   Models with a scale inconsistent with ``pixscale`` are dropped; the model that agrees with the
   most candidates wins. Exhaustive, deterministic (no random sampling).
3. **Refinement** - all stars are transformed and matched with their nearest neighbour; the
   2nd-order polynomial is fitted with sigma clipping, iterated. The result is accepted only if the
   number of close pairs is far above the number of chance coincidences.

If it fails, it is repeated with twice as many stars (``nb_use``).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from os import PathLike

import numpy as np

from .io import Catalog, load_catalog, save_xytr
from .transform import XYTransform, coo_trans, design_matrix, fit_poly

log = logging.getLogger(__name__)

MIN_PAIRS = 6  # minimal number of close pairs to accept a solution
MIN_PAIRS_ORDER2 = 30  # below this the transformation is fitted as linear (affine)
MIN_SIGNIFICANCE = 5.0  # (close pairs - chance pairs) / sqrt(chance pairs)
_OFFSETS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)])


class MatchError(RuntimeError):
    """Raised when the two star lists could not be matched."""


# ------------------------------------------------------------------------- result


@dataclass
class MatchResult:
    """Result of :func:`match`.

    ``ref_idx[k]`` and ``field_idx[k]`` are indices (into the input catalogues) of the k-th matched pair.
    """

    transform: XYTransform
    ref_idx: np.ndarray
    field_idx: np.ndarray
    rms: float  # residual rms of matched pairs, field -> reference [reference px]
    ref: Catalog
    field: Catalog
    n_candidates: int = 0  # pairs from local feature matching
    n_consensus: int = 0  # candidates agreeing with the best similarity model
    nb_use: int = 0  # number of brightest stars finally used
    candidates: tuple = field(default=(), repr=False)  # (ref_xy, field_xy, consensus mask) - diagnostics
    order: int = 2  # polynomial order of the transformation
    method: str = "features"  # how the initial solution was found: features / triangles

    @property
    def n(self) -> int:
        """Number of matched pairs."""
        return len(self.ref_idx)

    @property
    def ref_xy(self) -> np.ndarray:
        """(n, 2) reference coordinates of matched stars."""
        return np.column_stack([self.ref.x[self.ref_idx], self.ref.y[self.ref_idx]])

    @property
    def field_xy(self) -> np.ndarray:
        """(n, 2) field coordinates of matched stars."""
        return np.column_stack([self.field.x[self.field_idx], self.field.y[self.field_idx]])

    @property
    def ref_id(self):
        """Star IDs of matched reference stars (if the catalogue has IDs)."""
        return None if self.ref.id is None else self.ref.id[self.ref_idx]

    @property
    def field_id(self):
        """Star IDs of matched field stars (if the catalogue has IDs)."""
        return None if self.field.id is None else self.field.id[self.field_idx]

    @property
    def residuals(self) -> np.ndarray:
        """(n, 2) field -> reference residuals of matched pairs [reference px]."""
        x, y = self.transform.field_to_ref(self.field.x[self.field_idx], self.field.y[self.field_idx])
        return np.column_stack([x - self.ref.x[self.ref_idx], y - self.ref.y[self.ref_idx]])

    def save(self, path):
        """Save the transformation to an ``.xytr`` file."""
        return save_xytr(path, self.transform)

    def save_pairs(self, path):
        """Save matched pairs: id_ref x_ref y_ref mag_ref id_field x_field y_field mag_field."""
        r, f = self.ref_idx, self.field_idx
        rid = self.ref.id[r] if self.ref.id is not None else r
        fid = self.field.id[f] if self.field.id is not None else f
        with open(path, "w") as fh:
            fh.write("# id_ref x_ref y_ref mag_ref id_field x_field y_field mag_field\n")
            for k in range(self.n):
                fh.write(
                    f"{rid[k]} {self.ref.x[r[k]]:.3f} {self.ref.y[r[k]]:.3f} {self.ref.mag[r[k]]:.4f} "
                    f"{fid[k]} {self.field.x[f[k]]:.3f} {self.field.y[f[k]]:.3f} {self.field.mag[f[k]]:.4f}\n"
                )
        return path

    def __repr__(self):
        t = self.transform
        return (
            f"MatchResult(n={self.n} pairs, rms={self.rms:.3f} px, scale={t.scale:.5f}, "
            f"rotation={t.rotation:.3f} deg{', flipped' if t.flipped else ''})"
        )


# ------------------------------------------------------------------------- inputs


def as_catalog(obj) -> Catalog:
    """Accept a file path, a :class:`Catalog`, ``(x, y, mag)``, an (N, 3) array or a table/dict
    with ``x``, ``y`` and ``mag`` (or ``m``) columns."""
    if isinstance(obj, Catalog):
        return obj
    if isinstance(obj, (str, PathLike)):
        return load_catalog(obj)
    if hasattr(obj, "keys") or getattr(getattr(obj, "dtype", None), "names", None):
        names = {str(k).lower(): k for k in (obj.keys() if hasattr(obj, "keys") else obj.dtype.names)}
        try:
            cols = [obj[names[n]] for n in ("x", "y")]
            cols.append(obj[names["mag"] if "mag" in names else names["m"]])
        except KeyError as e:
            raise ValueError("table must have columns x, y and mag (or m)") from e
        ids = obj[names["id"]] if "id" in names else None
        return Catalog(*(np.asarray(c, float) for c in cols), id=None if ids is None else np.asarray(ids))
    arr = obj if isinstance(obj, (tuple, list)) else np.asarray(obj, float)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] != 3:
        arr = arr.T
    if len(arr) != 3:
        raise ValueError("expected (x, y, mag)")
    return Catalog(*arr)


# ------------------------------------------------------------------- 1. features


def local_features(star_x, star_y, all_x, all_y, radius, axis=0, kmax=32):
    """Projection arrays for stars ``star_*`` with neighbours from ``all_*`` (sorted by brightness).

    axis1 points to the ``axis``-th brightest neighbour (0 = brightest), axis2 is its normal; the
    projections of the other neighbours (brightness order) are scaled by the distance to that star.
    Returns ``(p1, p2, n)``: (S, K) projections (NaN-padded) and the number of valid entries per star.
    """
    dx = star_x[:, None] - all_x[None, :]
    dy = star_y[:, None] - all_y[None, :]
    d2 = dx * dx + dy * dy
    mask = (d2 < radius * radius) & (d2 > 0)
    cnt = mask.sum(1)
    k = int(min(kmax + 1, max(cnt.max(initial=0), axis + 1)))
    idx = np.argsort(~mask, axis=1, kind="stable")[:, :k]  # neighbours in brightness order
    valid = np.take_along_axis(mask, idx, 1)
    ndx = np.take_along_axis(dx, idx, 1)
    ndy = np.take_along_axis(dy, idx, 1)
    rr = np.hypot(ndx[:, axis], ndy[:, axis])
    rr[rr == 0] = 1
    ux, uy = ndx[:, axis] / rr, ndy[:, axis] / rr
    p1 = (ux[:, None] * ndx + uy[:, None] * ndy) / rr[:, None]
    p2 = (-uy[:, None] * ndx + ux[:, None] * ndy) / rr[:, None]
    p1[~valid] = np.nan
    p2[~valid] = np.nan
    n = np.where(cnt > axis, np.minimum(cnt, k) - 1, 0)
    return np.delete(p1, axis, 1), np.delete(p2, axis, 1), n


def feature_scores(f1, f2, err):
    """(R, F) similarity scores in [0, 1] of reference and field features.

    For element k a point weight ``n - k`` is given if the projections agree within ``err`` with
    element k, k+1 or k-1 of the other array (n = shorter array length).
    """
    x1, y1, n1 = f1
    x2, y2, n2 = f2
    k = min(x1.shape[1], x2.shape[1])
    x1, y1, x2, y2 = x1[:, :k], y1[:, :k], x2[:, :k], y2[:, :k]
    nmax = np.minimum(np.minimum(n1[:, None], n2[None, :]), k)
    kk = np.arange(k)[None, None, :]
    X1, Y1 = x1[:, None, :], y1[:, None, :]

    def close(xb, yb):
        with np.errstate(invalid="ignore"):
            return (np.abs(X1 - xb[None]) < err) & (np.abs(Y1 - yb[None]) < err)

    pad = np.full((x2.shape[0], 1), np.nan)
    hit = close(x2, y2)
    hit |= close(np.hstack([x2[:, 1:], pad]), np.hstack([y2[:, 1:], pad])) & (kk < nmax[:, :, None] - 1)
    hit |= close(np.hstack([pad, x2[:, :-1]]), np.hstack([pad, y2[:, :-1]])) & (kk > 1)
    w = np.where(kk < nmax[:, :, None], nmax[:, :, None] - kk, 0)
    score = (hit * w).sum(2) / np.maximum(0.5 * nmax * (nmax + 1), 1)
    score[nmax == 0] = 0
    return score


# ------------------------------------------------------------------ 2. consensus


def similarity_consensus(ref_c, field_c, pixscale, tol, scale_tol=0.03, max_models=20000):
    """Best similarity transform ``field = k * ref + t`` (complex) from candidate pairs.

    Every pair of candidates defines a model; returns ``(k, t, inlier_mask)`` or ``None``.
    """
    i, j = np.triu_indices(len(ref_c), 1)
    dr = ref_c[j] - ref_c[i]
    df = field_c[j] - field_c[i]
    ok = np.abs(dr) > 0
    i, dr, df = i[ok], dr[ok], df[ok]
    k = df / dr
    good = np.abs(np.abs(k) * pixscale - 1) < scale_tol
    i, k = i[good], k[good]
    if len(k) == 0:
        return None
    if len(k) > max_models:
        sel = np.linspace(0, len(k) - 1, max_models).astype(int)
        i, k = i[sel], k[sel]
    t = field_c[i] - k * ref_c[i]
    best_n, best = -1, 0
    for s in range(0, len(k), 2000):
        pred = k[s : s + 2000, None] * ref_c[None, :] + t[s : s + 2000, None]
        nin = (np.abs(pred - field_c[None, :]) < tol).sum(1)
        b = int(np.argmax(nin))
        if nin[b] > best_n:
            best_n, best = int(nin[b]), s + b
    inl = np.abs(k[best] * ref_c + t[best] - field_c) < tol
    return k[best], t[best], inl


# ------------------------------------------------------------- 2b. triangles (few stars)


def triangles(x, y):
    """All triangles of the points ``x, y``.

    Vertices are ordered by the length of the opposite side (shortest first). Returns
    ``(vertices (T, 3), longest side (T,), ratios (T, 2) = shorter sides / longest, orientation (T,))``.
    """
    n = len(x)
    if n < 3:
        return np.zeros((0, 3), int), np.zeros(0), np.zeros((0, 2)), np.zeros(0)
    v = np.array(list(itertools.combinations(range(n), 3)))
    px, py = x[v], y[v]
    # side opposite vertex m connects the two other vertices
    opp = np.stack(
        [np.hypot(px[:, (m + 1) % 3] - px[:, (m + 2) % 3], py[:, (m + 1) % 3] - py[:, (m + 2) % 3]) for m in range(3)],
        axis=1,
    )
    o = np.argsort(opp, axis=1, kind="stable")
    v = np.take_along_axis(v, o, 1)
    sides = np.take_along_axis(opp, o, 1)
    longest = sides[:, 2]
    ratios = sides[:, :2] / np.maximum(longest, 1e-12)[:, None]
    x0, y0 = x[v[:, 0]], y[v[:, 0]]
    cross = (x[v[:, 1]] - x0) * (y[v[:, 2]] - y0) - (y[v[:, 1]] - y0) * (x[v[:, 2]] - x0)
    return v, longest, ratios, np.sign(cross)


def triangle_models(ax, ay, bx, by, pixscale, tol, scale_tol=0.03, chunk=64):
    """Similarity models ``field = k * ref + t`` from all pairs of matching triangles.

    Triangles match if their longest sides agree (field side x ``pixscale``, within ``scale_tol`` and
    ``tol``), their side ratios agree (within ~1.5 px relative to the side) and orientation is the same.
    """
    vr, lr, rr, sr = triangles(ax, ay)
    vf, lf, rf, sf = triangles(bx, by)
    if len(vr) == 0 or len(vf) == 0:
        return np.zeros(0, complex), np.zeros(0, complex)
    lf_ref = lf * pixscale  # field side in reference pixels
    # sort both lists by the longest side: each chunk of field triangles is compared only with the
    # narrow slice of reference triangles of similar size
    o_r = np.argsort(lr, kind="stable")
    vr, lr, rr, sr = vr[o_r], lr[o_r], rr[o_r], sr[o_r]
    o_f = np.argsort(lf_ref, kind="stable")
    vf, lf_ref, rf, sf = vf[o_f], lf_ref[o_f], rf[o_f], sf[o_f]
    ref_c, fld_c = ax + 1j * ay, bx + 1j * by
    ks, ts = [], []
    for s in range(0, len(vf), chunk):
        e = slice(s, s + chunk)
        lo = np.searchsorted(lr, lf_ref[e][0] * (1 - scale_tol) - tol, "left")
        hi = np.searchsorted(lr, lf_ref[e][-1] * (1 + scale_tol) + tol, "right")
        if hi <= lo:
            continue
        w = slice(lo, hi)
        eps = np.clip(1.5 / np.maximum(lf_ref[e], 1e-9), 0.005, 0.05)[:, None]
        ok = np.abs(lr[None, w] - lf_ref[e, None]) < tol + scale_tol * lf_ref[e, None]
        ok &= np.abs(rr[None, w, 0] - rf[e, None, 0]) < eps
        ok &= np.abs(rr[None, w, 1] - rf[e, None, 1]) < eps
        ok &= sr[None, w] == sf[e, None]
        fi, ri = np.nonzero(ok)
        if len(fi) == 0:
            continue
        r3 = ref_c[vr[lo + ri]]  # (M, 3)
        f3 = fld_c[vf[s + fi]]
        rc = r3 - r3.mean(1, keepdims=True)
        fc = f3 - f3.mean(1, keepdims=True)
        k = (fc * np.conj(rc)).sum(1) / np.maximum((np.abs(rc) ** 2).sum(1), 1e-12)
        good = np.abs(np.abs(k) * pixscale - 1) < scale_tol
        ks.append(k[good])
        ts.append((f3.mean(1) - k * r3.mean(1))[good])
    if not ks:
        return np.zeros(0, complex), np.zeros(0, complex)
    return np.concatenate(ks), np.concatenate(ts)


def _match_triangles(ref, fld, p, info):
    """Initial solution from triangles of the brightest stars, then the usual refinement."""
    ratio = p.starratio if p.starratio is not None else area_ratio(ref, fld, p.pixscale)
    n_fld = min(len(fld.x), p.triangle_stars)
    n_ref = int(min(len(ref.x), max(round(n_fld / ratio), 3), 2.5 * p.triangle_stars))
    ro, fo = _brightest(ref.mag, n_ref), _brightest(fld.mag, n_fld)
    ax, ay, bx, by = ref.x[ro], ref.y[ro], fld.x[fo], fld.y[fo]
    k, t = triangle_models(ax, ay, bx, by, p.pixscale, p.tol)
    info.update(method="triangles", n_triangle_models=len(k), n_candidates=len(k))
    if len(k) == 0:
        return None
    # support of each model: field stars with a reference star within tol (field px)
    ref_c, fld_c = ax + 1j * ay, bx + 1j * by
    tol_f = p.tol / p.pixscale
    support = np.zeros(len(k), int)
    for s in range(0, len(k), 512):
        pred = k[s : s + 512, None] * ref_c[None, :] + t[s : s + 512, None]  # (M, R)
        d = np.abs(pred[:, :, None] - fld_c[None, None, :]).min(1)  # (M, F)
        support[s : s + 512] = (d < tol_f).sum(1)
    best = int(np.argmax(support))
    info.update(n_consensus=int(support[best]))
    if support[best] < 4:  # 3 = the triangle itself
        return None
    kb, tb = k[best], t[best]
    pred = kb * ref_c + tb
    d = np.abs(pred[:, None] - fld_c[None, :])
    j = d.argmin(1)
    inl = d[np.arange(len(ref_c)), j] < tol_f
    info["candidates"] = (ref_c[inl], fld_c[j[inl]], np.ones(int(inl.sum()), bool))
    px = np.array([tb.real, kb.real, -kb.imag, 0, 0, 0])
    py = np.array([tb.imag, kb.imag, kb.real, 0, 0, 0])
    return refine(ref.x, ref.y, fld.x, fld.y, px, py, r0=tol_f, n_iter=p.n_iter, clip=p.clip, order=p.order)


# ----------------------------------------------------------------- 3. refinement


def nearest_neighbour(qx, qy, px, py, r):
    """Index of the nearest point ``p`` within ``r`` of each query point (-1 if none) and distance.

    Grid hashing with cell size ``r``; numpy only.
    """
    best_d = np.full(len(qx), np.inf)
    best_i = np.full(len(qx), -1)
    if len(px) == 0 or len(qx) == 0:
        return best_i, best_d
    cx = np.floor(px / r).astype(np.int64)
    cy = np.floor(py / r).astype(np.int64)
    x0, y0 = cx.min() - 2, cy.min() - 2  # margin: points occupy cells 2..w-3, queries 1..w-2
    w = cx.max() - x0 + 3
    h = cy.max() - y0 + 3
    key = (cy - y0) * w + (cx - x0)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    qcx = np.floor(qx / r).astype(np.int64) - x0
    qcy = np.floor(qy / r).astype(np.int64) - y0
    q_in = np.nonzero((qcx >= 1) & (qcx < w - 1) & (qcy >= 1) & (qcy < h - 1))[0]
    qkey = qcy[q_in] * w + qcx[q_in]
    q_in = q_in[np.argsort(qkey, kind="stable")]  # sorted queries: much faster searchsorted
    qkey = qcy[q_in] * w + qcx[q_in]
    for oy in (-1, 0, 1):
        row = qkey + oy * w
        lo = np.searchsorted(skey, row - 1, "left")  # cells (x-1, x, x+1) are contiguous keys
        cnt = np.searchsorted(skey, row + 1, "right") - lo
        for c in range(int(cnt.max(initial=0))):
            sel = np.nonzero(cnt > c)[0]
            q = q_in[sel]
            pi = order[lo[sel] + c]
            d = np.hypot(px[pi] - qx[q], py[pi] - qy[q])
            better = d < best_d[q]
            best_d[q[better]] = d[better]
            best_i[q[better]] = pi[better]
    best_i[best_d > r] = -1
    return best_i, best_d


def _pairs(rx, ry, fx, fy, px, py, r):
    """One-to-one nearest-neighbour pairs (ref index, field index, distance) under ref->field model."""
    tx, ty = coo_trans(rx, ry, px, py)
    j, d = nearest_neighbour(tx, ty, fx, fy, r)
    i = np.nonzero(j >= 0)[0]
    j, d = j[i], d[i]
    o = np.lexsort((d, j))  # keep the closest reference star for each field star
    first = np.r_[True, j[o][1:] != j[o][:-1]] if len(o) else np.zeros(0, bool)
    return i[o][first], j[o][first], d[o][first]


def poly_order(n_pairs, order=None):
    """Polynomial order for ``n_pairs`` pairs: ``order`` if given, else 2 from 30 pairs, otherwise 1."""
    if order is not None:
        return order
    return 2 if n_pairs >= MIN_PAIRS_ORDER2 else 1


def _leverage(x, y, order):
    """Diagonal of the hat matrix of the polynomial least-squares fit (``1 - h`` scales LOO residuals)."""
    a = design_matrix(x, y)[:, : 3 if order == 1 else 6]
    a = a / np.maximum(np.linalg.norm(a, axis=0), 1e-300)
    q, _ = np.linalg.qr(a)
    return np.minimum((q * q).sum(1), 0.99)


def refine(rx, ry, fx, fy, px, py, r0, n_iter=4, clip=3.0, min_r=0.3, order=None):
    """Iterative nearest-neighbour matching of all stars + polynomial fit ref -> field with clipping.

    ``r0`` is the initial matching radius in field pixels. Returns ``(px, py, i, j)`` or ``None``
    if the solution is not significant.
    """
    r = r0
    i = j = np.zeros(0, int)
    for _ in range(n_iter):
        i, j, _d = _pairs(rx, ry, fx, fy, px, py, r)
        if len(i) < MIN_PAIRS:
            return None
        o = poly_order(len(i), order)
        px, py = fit_poly(rx[i], ry[i], fx[j], o), fit_poly(rx[i], ry[i], fy[j], o)
        tx, ty = coo_trans(rx[i], ry[i], px, py)
        res = np.hypot(tx - fx[j], ty - fy[j])
        rms = np.sqrt(np.mean(res**2))
        keep = res < max(clip * rms, 1e-9)
        i, j = i[keep], j[keep]
        if len(i) < MIN_PAIRS:
            return None
        o = poly_order(len(i), order)
        px, py = fit_poly(rx[i], ry[i], fx[j], o), fit_poly(rx[i], ry[i], fy[j], o)
        r = min(r, max(clip * rms, min_r))

    # significance: close pairs vs. chance coincidences
    # - leave-one-out residuals (each pair against the fit without it): with few pairs the fit itself
    #   pulls the pairs together, which would make random pairs look "close"
    o = poly_order(len(i), order)
    tx, ty = coo_trans(rx[i], ry[i], px, py)
    res = np.hypot(tx - fx[j], ty - fy[j]) / (1 - _leverage(rx[i], ry[i], o))
    r_s = r0 / 3
    n_close = int((res < r_s).sum())
    # - chance level measured by shifting the solution; only stars that stay inside the field count
    step = max(1, len(rx) // 5000)  # a subsample is enough to estimate the background
    ax, ay = coo_trans(rx[::step], ry[::step], px, py)
    (x0, x1), (y0, y1) = np.percentile(fx, [1, 99]), np.percentile(fy, [1, 99])
    n_in = ((ax > x0) & (ax < x1) & (ay > y0) & (ay < y1)).sum() * step
    rates = []
    for dx, dy in _OFFSETS * 10 * r0:
        sx, sy = ax + dx, ay + dy
        inside = (sx > x0) & (sx < x1) & (sy > y0) & (sy < y1)
        if inside.sum() >= 5:
            rates.append((nearest_neighbour(sx[inside], sy[inside], fx, fy, r_s)[0] >= 0).mean())
    chance = n_in * (np.mean(rates) if rates else 1.0)
    if n_close < MIN_PAIRS or (n_close - chance) / np.sqrt(chance + 1) < MIN_SIGNIFICANCE:
        return None
    return px, py, i, j


# ----------------------------------------------------------------------- driver


def area_ratio(ref, fld, pixscale) -> float:
    """Expected field/reference ratio of star numbers = ratio of the areas covered by the lists.

    Areas from the bounding boxes (5-95 percentiles); the field area is converted to reference pixels.
    """

    def area(x, y):
        if len(x) < 3:
            return 1.0
        (x0, x1), (y0, y1) = np.percentile(x, [5, 95]), np.percentile(y, [5, 95])
        return max((x1 - x0) * (y1 - y0) / 0.81, 1e-9)

    r = area(fld.x, fld.y) * pixscale**2 / area(ref.x, ref.y)
    return float(np.clip(r, 0.05, 20))


def _brightest(mag, n):
    return np.argsort(mag, kind="stable")[:n]


@dataclass
class Params:
    """Tuning parameters of :func:`match` (see its docstring)."""

    pixscale: float = 1.0
    nb_use: int = 400
    max_nb_use: int = 1600
    starratio: float | None = None
    tol: float = 3.0
    nbPCent_match: float = 0.25
    min_feature_stars: int = 100
    nbStarsRadius: float = 10
    starlist_error: float = 0.05
    min_score: float = 0.3
    axis_stars: int = 1
    n_iter: int = 4
    clip: float = 3.0
    order: int | None = None
    method: str = "auto"
    triangle_stars: int = 25


def _match_once(ref, fld, p: Params, nb_use, info):
    # 1. brightest stars (same magnitude depth: numbers scaled by the field/reference star ratio)
    ratio = p.starratio if p.starratio is not None else area_ratio(ref, fld, p.pixscale)
    n_ref = int(min(len(ref.x), nb_use, len(fld.x) / ratio))
    n_fld = int(min(len(fld.x), max(round(n_ref * ratio), 3)))
    ro, fo = _brightest(ref.mag, n_ref), _brightest(fld.mag, n_fld)
    ax, ay, bx, by = ref.x[ro], ref.y[ro], fld.x[fo], fld.y[fo]
    if len(ax) < 3 or len(bx) < 3:
        return None
    area = max(np.ptp(ax) * np.ptp(ay), 1e-9)
    radius = 0.5 * (p.nbStarsRadius / (len(ax) / area)) ** 0.5
    ns_r = min(len(ax), max(int(len(ax) * p.nbPCent_match), p.min_feature_stars))
    ns_f = min(len(bx), max(int(len(bx) * p.nbPCent_match), p.min_feature_stars))
    f_fld = local_features(bx[:ns_f], by[:ns_f], bx, by, radius / p.pixscale)
    # axis on the brightest neighbour in the field and on one of the `axis_stars` brightest in the
    # reference: tolerates a different brightness order of neighbours (e.g. other band)
    score = np.max(
        [
            feature_scores(local_features(ax[:ns_r], ay[:ns_r], ax, ay, radius, axis=a), f_fld, p.starlist_error)
            for a in range(p.axis_stars)
        ],
        axis=0,
    )
    jbest = score.argmax(1)
    ci = np.nonzero(score[np.arange(len(jbest)), jbest] > p.min_score)[0]
    cj = jbest[ci]
    info.update(method="features", nb_use=nb_use, n_candidates=len(ci), radius=radius, n_feature=(ns_r, ns_f))
    if len(ci) < 3:
        return None

    # 2. consensus similarity model
    ref_c = ax[ci] + 1j * ay[ci]
    fld_c = bx[cj] + 1j * by[cj]
    cons = similarity_consensus(ref_c, fld_c, p.pixscale, p.tol / p.pixscale)
    if cons is None:
        return None
    k, t, inl = cons
    info.update(n_consensus=int(inl.sum()), candidates=(ref_c, fld_c, inl))
    if inl.sum() < 3:
        return None

    # 3. refinement on all stars
    px = np.array([t.real, k.real, -k.imag, 0, 0, 0])
    py = np.array([t.imag, k.imag, k.real, 0, 0, 0])
    return refine(
        ref.x, ref.y, fld.x, fld.y, px, py, r0=p.tol / p.pixscale, n_iter=p.n_iter, clip=p.clip, order=p.order
    )


def _match_parity(ref, fld, p: Params, mirrored):
    """Run the matching (with retries); ``mirrored`` matches against the reference with y -> -y."""
    if mirrored:
        ref = Catalog(ref.x, -ref.y, ref.mag, id=ref.id, path=ref.path)
    info = {}
    out = None
    if p.method in ("auto", "features"):
        nb_use = p.nb_use
        n_max = max(len(ref.x), len(fld.x))
        while True:
            out = _match_once(ref, fld, p, nb_use, info)
            log.debug(
                "mirrored=%s nb_use=%d: %s", mirrored, nb_use, {k: v for k, v in info.items() if k != "candidates"}
            )
            if out is not None or nb_use >= min(n_max, p.max_nb_use):
                break
            nb_use = min(2 * nb_use, p.max_nb_use)
    if out is None and p.method in ("auto", "triangles"):
        tinfo = {}
        out = _match_triangles(ref, fld, p, tinfo)
        log.debug("mirrored=%s triangles: %s", mirrored, {k: v for k, v in tinfo.items() if k != "candidates"})
        if out is not None or p.method == "triangles":
            info = tinfo
    if out is not None and mirrored:  # candidates back in the original reference frame
        ref_c, fld_c, inl = info["candidates"]
        info["candidates"] = (np.conj(ref_c), fld_c, inl)
    return out, info


def match(ref, field, pixscale: float = 1.0, *, flip: bool | None = False, **params) -> MatchResult:
    """Match a field star list to a reference star list (both in pixels).

    Parameters
    ----------
    ref, field
        File path (daophot ``.ap/.out/.coo/...`` or text ``ID X Y MAG``), :class:`Catalog`,
        ``(x, y, mag)``, an (N, 3) array or a table/dict with ``x, y, mag`` columns.
    pixscale
        Pixel scale ratio field/reference (e.g. HAWKI/SOFI = 0.368). Must be known to ~3%.
    flip
        ``False``: same parity; ``True``: field is a mirror image of the reference;
        ``None``: try both.
    nb_use (400), max_nb_use (1600)
        Brightest stars used to find the initial solution; doubled on failure up to ``max_nb_use``.
    starratio (None)
        Field/reference ratio of star numbers in the same magnitude range; ``None``: from the
        ratio of the areas covered by the two lists (assumes similar depth for the brightest stars).
    tol (3.0)
        Matching tolerance [reference px] of the initial (shift + rotation + scale) solution; must
        cover the distortion not described by it.
    nbPCent_match (0.25), min_feature_stars (100), nbStarsRadius (10), starlist_error (0.05), min_score (0.3)
        Local features: fraction (at least ``min_feature_stars``) of the ``nb_use`` stars for which
        features are computed, average number of neighbours in the feature radius, projection
        tolerance and minimal score.
    axis_stars (1)
        Number of brightest reference neighbours tried as the feature axis. Use 2-3 when the
        brightness order differs between the lists (different bands).
    n_iter (4), clip (3.0)
        Refinement iterations and sigma-clipping threshold.
    method ("auto")
        Initial solution: ``"features"`` (local features + consensus), ``"triangles"`` (triangles of
        the ``triangle_stars`` (25) brightest field stars, for few stars) or ``"auto"`` (features,
        triangles if they fail).
    order (None)
        Polynomial order 1 or 2; ``None``: 2 if at least 30 pairs, otherwise 1.

    Returns
    -------
    MatchResult
        ``.transform`` (field <-> reference), ``.ref_idx`` / ``.field_idx`` of matched stars, ``.rms``.

    Raises
    ------
    MatchError
        If no significant solution was found.
    """
    unknown = set(params) - set(Params.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
    if params.get("method", "auto") not in ("auto", "features", "triangles"):
        raise ValueError("method must be 'auto', 'features' or 'triangles'")
    p = Params(pixscale=float(pixscale), **params)
    p.starratio = None if p.starratio is None else float(p.starratio)
    ref = as_catalog(ref)
    fld = as_catalog(field)

    out, info = None, {}
    for mirrored in (False, True) if flip is None else (bool(flip),):
        out, info = _match_parity(ref, fld, p, mirrored)
        if out is not None:
            break
    if out is None:
        raise MatchError(
            f"no significant match (candidates={info.get('n_candidates', 0)}, "
            f"consensus={info.get('n_consensus', 0)}, nb_use={info.get('nb_use', p.nb_use)}) - check pixscale"
            + ("" if flip is None else " and flip")
        )
    _, _, i, j = out
    # final fit of both directions on the matched pairs, in the original coordinates
    o = poly_order(len(i), p.order)
    px, py = fit_poly(ref.x[i], ref.y[i], fld.x[j], o), fit_poly(ref.x[i], ref.y[i], fld.y[j], o)
    qx, qy = fit_poly(fld.x[j], fld.y[j], ref.x[i], o), fit_poly(fld.x[j], fld.y[j], ref.y[i], o)
    transform = XYTransform(
        qx,
        qy,
        px,
        py,
        reference=None if ref.path is None else str(ref.path),
        field=None if fld.path is None else str(fld.path),
    )
    tx, ty = coo_trans(fld.x[j], fld.y[j], qx, qy)
    rms = float(np.sqrt(np.mean((tx - ref.x[i]) ** 2 + (ty - ref.y[i]) ** 2)))
    result = MatchResult(
        transform,
        i,
        j,
        rms,
        ref,
        fld,
        n_candidates=info.get("n_candidates", 0),
        n_consensus=info.get("n_consensus", 0),
        nb_use=info.get("nb_use", p.nb_use),
        candidates=info.get("candidates", ()),
        order=o,
        method=info.get("method", "features"),
    )
    log.debug("%s", result)
    return result


# --------------------------------------------------------------- legacy class API


class StarMatch:
    """Attribute-style interface kept for compatibility with the original scripts.

    New code should use :func:`match`::

        sm = StarMatch(pixscale=0.368)
        sm.ref_xr, sm.ref_yr, sm.ref_mr = x_ref, y_ref, m_ref
        sm.field_xr, sm.field_yr, sm.field_mr = x_field, y_field, m_field
        sm.go()
        sm.p_fr_x, sm.p_fr_y   # field -> reference
        sm.p_rf_x, sm.p_rf_y   # reference -> field
        sm.result              # MatchResult
    """

    _PARAMS = {
        "nb_use": "nb_use",
        "nbPCent_match": "nbPCent_match",
        "nbStarsRadius": "nbStarsRadius",
        "fieldStarsRatio": "starratio",
        "pixscale": "pixscale",
        "starlist_error": "starlist_error",
        "min_score": "min_score",
        "tol": "tol",
    }

    def __init__(self, **params):
        self.loud = False
        self.nb_use = 400
        self.nbPCent_match = 0.25
        self.nbStarsRadius = 10
        self.fieldStarsRatio = None  # None: from the areas covered by the two lists
        self.pixscale = 1
        self.starlist_error = 0.05
        self.min_score = 0.3
        self.tol = 3.0
        self.seed = None  # no longer used: the algorithm is deterministic
        self.mssg = ""
        self.ref_xr = self.ref_yr = self.ref_mr = ()
        self.field_xr = self.field_yr = self.field_mr = ()
        self.p_fr_x = self.p_fr_y = self.p_rf_x = self.p_rf_y = None
        self.result = None
        for key, value in params.items():
            if not hasattr(self, key):
                raise TypeError(f"unknown StarMatch parameter: {key!r}")
            setattr(self, key, value)

    def set_reference(self, x, y, mag):
        self.ref_xr, self.ref_yr, self.ref_mr = x, y, mag
        return self

    def set_field(self, x, y, mag):
        self.field_xr, self.field_yr, self.field_mr = x, y, mag
        return self

    @property
    def transform(self) -> XYTransform:
        if self.result is None:
            raise MatchError("no transformation yet - run go() first")
        return self.result.transform

    def go(self):
        params = {new: getattr(self, old) for old, new in self._PARAMS.items()}
        r = match((self.ref_xr, self.ref_yr, self.ref_mr), (self.field_xr, self.field_yr, self.field_mr), **params)
        self.result = r
        t = r.transform
        self.p_fr_x, self.p_fr_y, self.p_rf_x, self.p_rf_y = t.fr_x, t.fr_y, t.rf_x, t.rf_y
        # candidates from local features / consensus (coordinates)
        ref_c, fld_c, inl = r.candidates
        self.ref_match_x, self.ref_match_y = list(ref_c.real), list(ref_c.imag)
        self.field_match_x, self.field_match_y = list(fld_c.real), list(fld_c.imag)
        # final pairs used for the fit (formerly: confirmed by triangles)
        self.trianglesMatch_ref_x, self.trianglesMatch_ref_y = map(list, r.ref_xy.T)
        self.trianglesMatch_field_x, self.trianglesMatch_field_y = map(list, r.field_xy.T)
        self.trianglesFail_ref_x, self.trianglesFail_ref_y = list(ref_c.real[~inl]), list(ref_c.imag[~inl])
        self.trianglesFail_field_x, self.trianglesFail_field_y = list(fld_c.real[~inl]), list(fld_c.imag[~inl])
        if self.loud:
            self.mssg += (
                f"\nstars used: {r.nb_use}"
                f"\ncandidates from local features: {r.n_candidates}"
                f"\ncandidates consistent with consensus: {r.n_consensus}"
                f"\nmatched pairs: {r.n}, rms = {r.rms:.3f} px"
            )
        return self

    _DEPRECATED = {
        f"traingles{kind}_{who}_{c}": f"triangles{kind}_{who}_{c}"
        for kind in ("Match", "Fail")
        for who in ("ref", "field")
        for c in ("x", "y")
    }

    def __getattr__(self, name):
        new = type(self)._DEPRECATED.get(name)
        if new is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        import warnings

        warnings.warn(f"{name} is deprecated, use {new}", DeprecationWarning, stacklevel=2)
        return getattr(self, new)

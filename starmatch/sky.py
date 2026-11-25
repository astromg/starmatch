"""Astrometry: match a pixel star list to a sky catalogue (RA, Dec), e.g. Gaia.

The catalogue is projected on the tangent plane (gnomonic / TAN projection) and matched to the
field with the same engine as two pixel lists (:func:`starmatch.match`). The result maps pixels to
the standard coordinates (xi, eta) with a polynomial (2nd order, or linear for few stars) and can
be exported as a FITS WCS header (TAN or TAN-SIP).

Pixel coordinates are assumed to follow the FITS/daophot convention (centre of the first pixel = 1).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from .core import Catalog, MatchError, as_catalog, match
from .transform import coo_trans, fit_poly

log = logging.getLogger(__name__)

GAIA_EPOCH = 2016.0  # Gaia DR3 reference epoch
MAS_PER_DEG = 3.6e6

# ----------------------------------------------------------------------- catalogue


class SkyCatalog(tuple):
    """Sky catalogue: behaves like ``(ra, dec, mag)`` [deg, deg, mag]; optional proper motions
    ``pmra`` (= mu_alpha * cos(dec)) and ``pmdec`` [mas/yr], ``id``, ``epoch`` [yr]."""

    def __new__(cls, ra, dec, mag, pmra=None, pmdec=None, id=None, epoch=GAIA_EPOCH, path=None):  # noqa: A002
        obj = super().__new__(cls, (np.asarray(ra, float), np.asarray(dec, float), np.asarray(mag, float)))
        n = len(obj[0])
        obj.pmra = np.zeros(n) if pmra is None else np.nan_to_num(np.asarray(pmra, float))
        obj.pmdec = np.zeros(n) if pmdec is None else np.nan_to_num(np.asarray(pmdec, float))
        obj.id = None if id is None else np.asarray(id)
        obj.epoch = epoch
        obj.path = path
        return obj

    ra = property(lambda self: self[0])
    dec = property(lambda self: self[1])
    mag = property(lambda self: self[2])

    def __repr__(self):
        src = f", path={os.fspath(self.path)!r}" if self.path is not None else ""
        return f"SkyCatalog({len(self.ra)} stars, epoch={self.epoch}{src})"

    def at_epoch(self, epoch) -> SkyCatalog:
        """Positions moved with proper motions to ``epoch`` (decimal year or ISO date)."""
        epoch = decimal_year(epoch)
        dt = epoch - self.epoch
        dec = self.dec + self.pmdec * dt / MAS_PER_DEG
        ra = self.ra + self.pmra * dt / MAS_PER_DEG / np.cos(np.radians(self.dec))
        return SkyCatalog(ra, dec, self.mag, self.pmra, self.pmdec, self.id, epoch, self.path)


_ALIASES = {
    "ra": ("ra", "ra_icrs", "raj2000", "_raj2000", "alpha"),
    "dec": ("dec", "de", "de_icrs", "dej2000", "_dej2000", "delta"),
    "mag": ("mag", "phot_g_mean_mag", "gmag", "g_mag", "g", "m"),
    "pmra": ("pmra",),
    "pmdec": ("pmdec", "pmde"),
    "id": ("id", "source_id", "source"),
}


def _to_float(col):
    return np.array([np.nan if v in ("--", "", "nan", "NaN", "null", "None") else float(v) for v in col])


def load_sky_catalog(path, epoch: float = GAIA_EPOCH) -> SkyCatalog:
    """Read a sky catalogue.

    * text with a header line (CSV or whitespace; Gaia archive / VizieR column names such as
      ``ra, dec, phot_g_mean_mag, pmra, pmdec, source_id`` or ``RA_ICRS, DE_ICRS, Gmag, pmRA, pmDE``),
    * or plain columns ``ra dec mag [pmra pmdec]`` without a header (``--`` = missing value).
    """
    text = Path(path).read_text().splitlines()
    lines = [ln for ln in text if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"no stars in {path}")
    sep = "," if "," in lines[0] else None
    first = [t.strip().strip('"') for t in lines[0].split(sep)]
    try:
        float(first[0])
        header = None
    except ValueError:
        header = [t.lower() for t in first]
        lines = lines[1:]
    rows = [[t.strip().strip('"') for t in ln.split(sep)] for ln in lines]
    cols = list(zip(*rows))
    if header is None:
        names = ["ra", "dec", "mag", "pmra", "pmdec"][: len(cols)]
        get = dict(zip(names, cols))
    else:
        get = {}
        for key, aliases in _ALIASES.items():
            for a in aliases:
                if a in header:
                    get[key] = cols[header.index(a)]
                    break
        missing = {"ra", "dec", "mag"} - set(get)
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)} in header {first}")
    return SkyCatalog(
        _to_float(get["ra"]),
        _to_float(get["dec"]),
        _to_float(get["mag"]),
        _to_float(get["pmra"]) if "pmra" in get else None,
        _to_float(get["pmdec"]) if "pmdec" in get else None,
        id=get.get("id"),
        epoch=epoch,
        path=path,
    )


def save_sky_catalog(path, cat: SkyCatalog):
    """Write ``ra dec mag pmra pmdec`` (with a header line)."""
    with open(path, "w") as f:
        f.write("ra dec mag pmra pmdec\n")
        for r, d, m, a, b in zip(cat.ra, cat.dec, cat.mag, cat.pmra, cat.pmdec):
            f.write(f"{r:.9f} {d:.9f} {m:.4f} {a:.4f} {b:.4f}\n")
    return path


def as_sky_catalog(obj) -> SkyCatalog:
    """File path, :class:`SkyCatalog`, ``(ra, dec, mag)`` or a table/dict with ra/dec/mag columns."""
    if isinstance(obj, SkyCatalog):
        return obj
    if isinstance(obj, (str, PathLike)):
        return load_sky_catalog(obj)
    if hasattr(obj, "keys") or getattr(getattr(obj, "dtype", None), "names", None):
        names = {str(k).lower(): k for k in (obj.keys() if hasattr(obj, "keys") else obj.dtype.names)}
        cols = {}
        for key, aliases in _ALIASES.items():
            for a in aliases:
                if a in names:
                    cols[key] = np.asarray(obj[names[a]])
                    break
        if not {"ra", "dec", "mag"} <= set(cols):
            raise ValueError("table must have ra, dec and mag columns")
        return SkyCatalog(
            cols["ra"],
            cols["dec"],
            np.asarray(cols["mag"], float),
            cols.get("pmra"),
            cols.get("pmdec"),
            id=cols.get("id"),
        )
    ra, dec, mag = obj
    return SkyCatalog(ra, dec, mag)


def decimal_year(t) -> float:
    """Decimal year from a number (year), ``numpy.datetime64`` or ISO date string."""
    if isinstance(t, (int, float, np.floating)):
        return float(t)
    try:
        return float(t)  # "2012.78"
    except ValueError:
        pass
    d = np.datetime64(str(t).replace("Z", ""), "s")
    year = d.astype("datetime64[Y]")
    start = year.astype("datetime64[s]")
    end = (year + np.timedelta64(1, "Y")).astype("datetime64[s]")
    return 1970 + year.astype(int) + float((d - start) / (end - start))


# ---------------------------------------------------------------------- projection


def radec_to_std(ra, dec, ra0, dec0):
    """Gnomonic (TAN) projection: standard coordinates ``xi`` (east), ``eta`` (north) [arcsec]."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)) / cosc
    return np.degrees(xi) * 3600, np.degrees(eta) * 3600


def std_to_radec(xi, eta, ra0, dec0):
    """Inverse gnomonic projection; ``xi, eta`` in arcsec, returns ``ra, dec`` [deg]."""
    x, y = np.radians(np.asarray(xi, float) / 3600), np.radians(np.asarray(eta, float) / 3600)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0))
    ra = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)


def _tangent_point(ra, dec):
    """Centre of a catalogue (circular mean of RA, mean Dec)."""
    a = np.radians(ra)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360), float(np.mean(dec))


# ------------------------------------------------------------------------- result


@dataclass
class SkyTransform:
    """Pixel <-> sky transformation.

    ``(xi, eta) = poly(x, y)`` with coefficients ``p_xi, p_eta`` [arcsec] (same layout as ``.xytr``),
    inverse ``(x, y) = poly(xi, eta)`` with ``p_x, p_y``; ``(xi, eta)`` are standard coordinates of
    the TAN projection around ``(ra0, dec0)``.
    """

    ra0: float
    dec0: float
    p_xi: np.ndarray
    p_eta: np.ndarray
    p_x: np.ndarray
    p_y: np.ndarray
    field: str | None = None
    catalog: str | None = None

    def pix_to_sky(self, x, y):
        """Pixel coordinates -> ``ra, dec`` [deg]."""
        xi, eta = coo_trans(x, y, self.p_xi, self.p_eta)
        return std_to_radec(xi, eta, self.ra0, self.dec0)

    def sky_to_pix(self, ra, dec):
        """``ra, dec`` [deg] -> pixel coordinates."""
        xi, eta = radec_to_std(ra, dec, self.ra0, self.dec0)
        return coo_trans(xi, eta, self.p_x, self.p_y)

    @property
    def _cd(self):
        return np.array([[self.p_xi[1], self.p_xi[2]], [self.p_eta[1], self.p_eta[2]]])  # arcsec/px

    @property
    def pixscale(self) -> float:
        """Pixel scale [arcsec/px] (linear part)."""
        return float(np.sqrt(abs(np.linalg.det(self._cd))))

    @property
    def flipped(self) -> bool:
        """``False`` for the usual sky orientation seen from the ground (north up -> east left)."""
        return bool(np.linalg.det(self._cd) > 0)

    @property
    def rotation(self) -> float:
        """Position angle of the +y pixel axis, east of north [deg] (0 = north up)."""
        return float(np.degrees(np.arctan2(self.p_xi[2], self.p_eta[2])) % 360)

    def wcs_header(self) -> dict:
        """FITS WCS keywords (``RA---TAN`` or ``RA---TAN-SIP`` with forward and inverse SIP).

        CRVAL is the tangent point, CRPIX the pixel at the tangent point.
        """
        a, b = np.asarray(self.p_xi, float) / 3600, np.asarray(self.p_eta, float) / 3600  # deg
        x0, y0 = coo_trans(0.0, 0.0, self.p_x, self.p_y)  # Newton: pixel where xi = eta = 0
        for _ in range(20):
            f = np.array(coo_trans(x0, y0, a, b))
            jac = np.array(
                [
                    [a[1] + a[3] * y0 + 2 * a[4] * x0, a[2] + a[3] * x0 + 2 * a[5] * y0],
                    [b[1] + b[3] * y0 + 2 * b[4] * x0, b[2] + b[3] * x0 + 2 * b[5] * y0],
                ]
            )
            dx, dy = np.linalg.solve(jac, -f)
            x0, y0 = x0 + dx, y0 + dy
        crval = (self.ra0, self.dec0)
        # re-expand around crpix: xi = CD (u, v) + quadratic(u, v)
        cd = np.array(
            [
                [a[1] + a[3] * y0 + 2 * a[4] * x0, a[2] + a[3] * x0 + 2 * a[5] * y0],
                [b[1] + b[3] * y0 + 2 * b[4] * x0, b[2] + b[3] * x0 + 2 * b[5] * y0],
            ]
        )
        hdr = {
            "WCSAXES": 2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "RADESYS": "ICRS",
            "CRPIX1": float(x0),
            "CRPIX2": float(y0),
            "CRVAL1": float(crval[0]),
            "CRVAL2": float(crval[1]),
            "CD1_1": float(cd[0, 0]),
            "CD1_2": float(cd[0, 1]),
            "CD2_1": float(cd[1, 0]),
            "CD2_2": float(cd[1, 1]),
        }
        quad = np.array([[a[4], a[3], a[5]], [b[4], b[3], b[5]]])  # u^2, uv, v^2
        if not np.any(quad):
            return hdr
        # SIP: CD @ (u + f, v + g) = CD @ (u, v) + quad  ->  (f, g) = CD^-1 quad
        sip = np.linalg.solve(cd, quad)
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
        hdr["A_ORDER"] = hdr["B_ORDER"] = 2
        for name, row in (("A", sip[0]), ("B", sip[1])):
            hdr[f"{name}_2_0"], hdr[f"{name}_1_1"], hdr[f"{name}_0_2"] = map(float, row)
        # inverse SIP (AP, BP) fitted on a grid around crpix
        span = 1.2 * max(abs(x0), abs(y0), 1000.0)
        g = np.linspace(-span, span, 41)
        u, v = (m.ravel() for m in np.meshgrid(g, g))
        f = sip[0, 0] * u * u + sip[0, 1] * u * v + sip[0, 2] * v * v
        h = sip[1, 0] * u * u + sip[1, 1] * u * v + sip[1, 2] * v * v
        U, V = u + f, v + h
        ap, bp = fit_poly(U, V, u - U), fit_poly(U, V, v - V)
        hdr["AP_ORDER"] = hdr["BP_ORDER"] = 2
        for name, c in (("AP", ap), ("BP", bp)):  # c: [1, U, V, UV, U^2, V^2]
            for (p_, q_), k in zip([(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)], range(6)):
                hdr[f"{name}_{p_}_{q_}"] = float(c[k])
        return hdr

    def wcs(self):
        """``astropy.wcs.WCS`` object (requires astropy)."""
        from astropy.wcs import WCS

        return WCS(self.wcs_header())

    def __repr__(self):
        flip = ", flipped" if self.flipped else ""
        return f'SkyTransform(pixscale={self.pixscale:.4f}"/px, PA={self.rotation:.3f} deg{flip})'


def save_sky_transform(path, t: SkyTransform) -> Path:
    """Save to a text file (``.radec``), same style as ``.xytr``."""

    def fmt(p):
        return "".join(repr(float(v)) + " " for v in p)

    txt = (
        f"# field={t.field} catalog={t.catalog}\n"
        "# (xi, eta) [arcsec] = gnomonic (TAN) standard coordinates around (RA0, DEC0) [deg]\n"
        "# xi  = pxi[0]  + pxi[1] * x + pxi[2] * y + pxi[3] * x*y + pxi[4] * x*x + pxi[5] * y*y\n"
        "# x   = px[0]   + px[1] * xi + px[2] * eta + px[3] * xi*eta + px[4] * xi*xi + px[5] * eta*eta\n"
        f"TAN RA0= {t.ra0!r} DEC0= {t.dec0!r}\n"
        "# Pixel to sky\n"
        f"P2S Pxi=  {fmt(t.p_xi)}\n"
        f"P2S Peta=  {fmt(t.p_eta)}\n"
        "# Sky to pixel\n"
        f"S2P Px=  {fmt(t.p_x)}\n"
        f"S2P Py=  {fmt(t.p_y)}\n"
    )
    path = Path(path)
    path.write_text(txt)
    return path


def load_sky_transform(path) -> SkyTransform:
    vals = {}
    field = catalog = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") and "field=" in line:
            field = line.split("field=")[1].split()[0]
            catalog = line.split("catalog=")[1].split()[0] if "catalog=" in line else None
        elif line.startswith("TAN"):
            parts = line.split()
            vals["ra0"], vals["dec0"] = float(parts[2]), float(parts[4])
        else:
            for key in ("P2S Pxi=", "P2S Peta=", "S2P Px=", "S2P Py="):
                if line.startswith(key):
                    vals[key] = np.array(line.split("=")[1].split(), dtype=float)
    return SkyTransform(
        vals["ra0"],
        vals["dec0"],
        vals["P2S Pxi="],
        vals["P2S Peta="],
        vals["S2P Px="],
        vals["S2P Py="],
        field=None if field in (None, "None") else field,
        catalog=None if catalog in (None, "None") else catalog,
    )


@dataclass
class SkyMatchResult:
    """Result of :func:`match_sky`. ``field_idx[k]`` and ``cat_idx[k]`` are the k-th matched pair."""

    transform: SkyTransform
    field_idx: np.ndarray
    cat_idx: np.ndarray
    rms: float  # residual rms [arcsec]
    field: Catalog
    catalog: SkyCatalog  # at the epoch of the observation
    order: int = 2
    pixel_match: object = None  # underlying MatchResult (diagnostics)

    @property
    def n(self) -> int:
        return len(self.field_idx)

    @property
    def field_id(self):
        return None if self.field.id is None else self.field.id[self.field_idx]

    @property
    def cat_id(self):
        return None if self.catalog.id is None else self.catalog.id[self.cat_idx]

    @property
    def residuals(self) -> np.ndarray:
        """(n, 2) residuals (catalogue - fitted) in (xi, eta) [arcsec]."""
        t = self.transform
        xi, eta = coo_trans(self.field.x[self.field_idx], self.field.y[self.field_idx], t.p_xi, t.p_eta)
        cxi, ceta = radec_to_std(self.catalog.ra[self.cat_idx], self.catalog.dec[self.cat_idx], t.ra0, t.dec0)
        return np.column_stack([cxi - xi, ceta - eta])

    def radec(self):
        """RA, Dec [deg] of all field stars."""
        return self.transform.pix_to_sky(self.field.x, self.field.y)

    def save(self, path):
        return save_sky_transform(path, self.transform)

    def wcs_header(self) -> dict:
        return self.transform.wcs_header()

    def save_wcs(self, path):
        """Write the WCS as FITS header cards (text, 80 characters per card)."""
        cards = []
        for k, v in self.wcs_header().items():
            if isinstance(v, str):
                card = f"{k:<8}= '{v:<8}'"
            elif isinstance(v, int):
                card = f"{k:<8}= {v:>20d}"
            else:
                card = f"{k:<8}= {v:>20.13G}"
            cards.append(card.ljust(80))
        Path(path).write_text("\n".join(cards) + "\n")
        return path

    def save_pairs(self, path):
        """id_field x y mag id_cat ra dec mag_cat dxi deta[arcsec]."""
        f, c = self.field_idx, self.cat_idx
        fid = self.field.id[f] if self.field.id is not None else f
        cid = self.catalog.id[c] if self.catalog.id is not None else c
        res = self.residuals
        with open(path, "w") as fh:
            fh.write("# id_field x y mag id_cat ra dec mag_cat dxi[arcsec] deta[arcsec]\n")
            for k in range(self.n):
                fh.write(
                    f"{fid[k]} {self.field.x[f[k]]:.3f} {self.field.y[f[k]]:.3f} {self.field.mag[f[k]]:.4f} "
                    f"{cid[k]} {self.catalog.ra[c[k]]:.8f} {self.catalog.dec[c[k]]:.8f} "
                    f"{self.catalog.mag[c[k]]:.4f} {res[k, 0]:.4f} {res[k, 1]:.4f}\n"
                )
        return path

    def __repr__(self):
        t = self.transform
        return (
            f'SkyMatchResult(n={self.n} stars, rms={self.rms:.3f}", pixscale={t.pixscale:.4f}"/px, '
            f"PA={t.rotation:.2f} deg{', flipped' if t.flipped else ''}, order={self.order})"
        )


# ----------------------------------------------------------------------- matching


def match_sky(
    field,
    catalog,
    pixscale: float,
    epoch=None,
    *,
    ra0: float | None = None,
    dec0: float | None = None,
    flip: bool | None = None,
    tol: float = 3.0,
    axis_stars: int = 2,
    **params,
) -> SkyMatchResult:
    """Astrometric solution: match a pixel star list to a sky catalogue.

    Parameters
    ----------
    field
        Pixel star list: file path (daophot / text ``ID X Y MAG``), :class:`Catalog`, ``(x, y, mag)``...
    catalog
        Sky catalogue: file path (see :func:`load_sky_catalog`), :class:`SkyCatalog`, ``(ra, dec, mag)``
        or a table with ra/dec/mag (and optionally pmra/pmdec) columns.
    pixscale
        Pixel scale [arcsec/px], must be known to ~3%.
    epoch
        Epoch of the observation (decimal year or ISO date, e.g. FITS ``DATE-OBS``); the catalogue is
        moved with proper motions from its epoch (Gaia DR3: 2016.0). ``None``: no correction.
    ra0, dec0
        Tangent point [deg]; default: centre of the catalogue.
    flip
        ``None`` (default): try both parities; ``False``: usual sky orientation (east is
        counter-clockwise from north, as seen from the ground); ``True``: mirrored.
    tol
        Matching tolerance of the initial solution [field px].
    axis_stars
        Number of brightest catalogue neighbours tried as the local feature axis (2: tolerates
        a different brightness order of neighbours between the catalogue band and the image band).
    **params
        Passed to :func:`starmatch.match` (``nb_use``, ``axis_stars``, ``order``, ...).
    """
    fld = as_catalog(field)
    cat = as_sky_catalog(catalog)
    if epoch is not None:
        cat = cat.at_epoch(epoch)
    if ra0 is None or dec0 is None:
        ra0, dec0 = _tangent_point(cat.ra, cat.dec)
    xi, eta = radec_to_std(cat.ra, cat.dec, ra0, dec0)
    pixscale = float(pixscale)
    # reference "pixels" with the field pixel scale; (xi, eta) frame is mirrored relative to the usual
    # image orientation, hence flip=False here means mirrored=True in the pixel matcher
    ref = Catalog(xi / pixscale, eta / pixscale, cat.mag, id=cat.id)
    order_parity = (True, False) if flip is None else ((not flip),)
    err = None
    for mirrored in order_parity:
        try:
            r = match(ref, fld, pixscale=1.0, flip=mirrored, tol=tol, axis_stars=axis_stars, **params)
            break
        except MatchError as e:
            err = e
    else:
        raise MatchError(f"astrometric match failed: {err}")

    i, j = r.ref_idx, r.field_idx
    o = r.order
    p_xi = fit_poly(fld.x[j], fld.y[j], xi[i], o)
    p_eta = fit_poly(fld.x[j], fld.y[j], eta[i], o)
    p_x = fit_poly(xi[i], eta[i], fld.x[j], o)
    p_y = fit_poly(xi[i], eta[i], fld.y[j], o)
    t = SkyTransform(
        ra0,
        dec0,
        p_xi,
        p_eta,
        p_x,
        p_y,
        field=None if fld.path is None else str(fld.path),
        catalog=None if cat.path is None else str(cat.path),
    )
    fxi, feta = coo_trans(fld.x[j], fld.y[j], p_xi, p_eta)
    rms = float(np.sqrt(np.mean((fxi - xi[i]) ** 2 + (feta - eta[i]) ** 2)))
    result = SkyMatchResult(t, j, i, rms, fld, cat, order=o, pixel_match=r)
    log.info("%s", result)
    return result

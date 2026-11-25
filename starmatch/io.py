"""Readers for star catalogues (daophot formats, plain text) and ``.xytr`` files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .transform import XYTransform

DAOPHOT_HEADER_LINES = 3
# daophot photometry files with one line per star: ID X Y MAG ...
DAOPHOT_SINGLE_LINE = {".out", ".coo", ".lst", ".als", ".nst", ".mag", ".pk"}


def _rows(path, skip: int = 0, comment: str = "#"):
    """Yield split, non-empty, non-comment lines after skipping ``skip`` header lines."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            parts = line.split()
            if parts and not parts[0].startswith(comment):
                yield parts


def read_daophot_ap(path):
    """Read a daophot ``.ap`` file (two lines per star: values, errors).

    Returns ``(data, errors)`` as lists of columns (strings), like the legacy ``loadap``.
    """
    rows = list(_rows(path, skip=DAOPHOT_HEADER_LINES))
    data = list(zip(*rows[0::2]))
    errors = list(zip(*rows[1::2]))
    return data, errors


def read_columns(path, skip: int = 0):
    """Read a whitespace-separated text file and return a list of columns (strings)."""
    return list(zip(*_rows(path, skip=skip)))


class Catalog(tuple):
    """Star list: behaves like the tuple ``(x, y, mag)``, plus optional ``id`` and ``path``.

    ``x, y, m = catalog`` works as for a plain tuple.
    """

    def __new__(cls, x, y, mag, id=None, path=None):  # noqa: A002
        obj = super().__new__(cls, (np.asarray(x, float), np.asarray(y, float), np.asarray(mag, float)))
        obj.id = None if id is None else np.asarray(id)
        obj.path = path
        return obj

    x = property(lambda self: self[0])
    y = property(lambda self: self[1])
    mag = property(lambda self: self[2])

    def __repr__(self):
        src = f", path={os.fspath(self.path)!r}" if self.path is not None else ""
        return f"Catalog({len(self.x)} stars{src})"


def _ids(col):
    try:
        return np.asarray(col, dtype=np.int64)
    except ValueError:
        return np.asarray(col)


def load_catalog(path, fmt: str | None = None, columns=(1, 2, 3), id_column: int | None = 0) -> Catalog:
    """Load star positions and magnitudes from a catalogue file.

    The format is chosen from the file extension (or ``fmt``):

    * ``.ap``  - daophot aperture photometry (two lines per star),
    * ``.out``, ``.coo``, ``.lst``, ``.als``, ... - daophot, one line per star,
    * anything else - plain text; ``#`` comments are skipped.

    ``columns`` are the 0-based indices of x, y and magnitude and ``id_column`` the index of the
    star ID (``None`` = no IDs). Daophot default: ``ID X Y MAG``.

    Returns a :class:`Catalog` - unpacks as ``x, y, mag`` (float arrays); ``.id`` holds star IDs.
    """
    ext = (fmt or Path(path).suffix).lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext == ".ap":
        cols, _ = read_daophot_ap(path)
    elif ext in DAOPHOT_SINGLE_LINE:
        cols = read_columns(path, skip=DAOPHOT_HEADER_LINES)
    else:
        cols = read_columns(path)
    if not cols:
        raise ValueError(f"no stars found in {path}")
    ix, iy, im = columns
    ids = _ids(cols[id_column]) if id_column is not None else None
    return Catalog(cols[ix], cols[iy], cols[im], id=ids, path=path)


# --------------------------------------------------------------------------- xytr


def save_xytr(path, transform: XYTransform) -> Path:
    """Save transformation coefficients to an ``.xytr`` text file."""

    def fmt(p):
        return "".join(repr(float(v)) + " " for v in p)

    txt = (
        f"# reference={transform.reference} field={transform.field}\n"
        "# x' = px[0] + px[1] * x + px[2] * y + px[3] * x*y + px[4] * x*x + px[5] * y*y\n"
        "# y' = py[0] + py[1] * x + py[2] * y + py[3] * x*y + py[4] * x*x + py[5] * y*y\n"
        "# Field to Reference\n"
        f"F2R Px=  {fmt(transform.fr_x)}\n"
        f"F2R Py=  {fmt(transform.fr_y)}\n"
        "# Reference to Field\n"
        f"R2F Px=  {fmt(transform.rf_x)}\n"
        f"R2F Py=  {fmt(transform.rf_y)}\n"
    )
    path = Path(path)
    path.write_text(txt)
    return path


def load_xytr(path) -> XYTransform:
    """Load transformation coefficients from an ``.xytr`` file."""
    ref = field = None
    coeffs = {}
    with open(path) as f:
        for line in f:
            if "reference=" in line:
                ref = line.split("reference=")[1].split()[0]
            if "field=" in line:
                field = line.split("field=")[1].split()[0]
            for key in ("F2R Px=", "F2R Py=", "R2F Px=", "R2F Py="):
                if line.startswith(key):
                    coeffs[key] = np.array(line.split("=")[1].split(), dtype=float)
    return XYTransform(
        fr_x=coeffs["F2R Px="],
        fr_y=coeffs["F2R Py="],
        rf_x=coeffs["R2F Px="],
        rf_y=coeffs["R2F Py="],
        reference=ref,
        field=field,
    )


def default_xytr_path(field_path) -> Path:
    """``dir/field.ap`` -> ``dir/field.xytr``."""
    return Path(field_path).with_suffix(".xytr")


# ------------------------------------------------------------- legacy API (v0.0)


def loadap(file):
    """Legacy: ``dane, bledy = loadap('x.ap')``."""
    return read_daophot_ap(file)


def loadout(file):
    """Legacy: ``dane = loadout('x.out')``."""
    return read_columns(file, skip=DAOPHOT_HEADER_LINES)


def load_file(min, file):  # noqa: A002 - legacy signature
    """Legacy: ``dane = load_file(skip_lines, 'x.txt')``."""
    return read_columns(file, skip=min)


def loadxytr(fname):
    """Legacy: returns ``(ref_file, field_file, p_fr_x, p_fr_y, p_rf_x, p_rf_y)``."""
    t = load_xytr(fname)
    return t.reference, t.field, t.fr_x, t.fr_y, t.rf_x, t.rf_y


def saveP2file(file_name_1, file_name_2, p_fr_x, p_fr_y, p_rf_x, p_rf_y):  # noqa: N802
    """Legacy: writes ``<field>.xytr`` next to the field file."""
    t = XYTransform(p_fr_x, p_fr_y, p_rf_x, p_rf_y, reference=os.fspath(file_name_1), field=os.fspath(file_name_2))
    return save_xytr(default_xytr_path(file_name_2), t)


# --------------------------------------------------------------------------- FITS


def read_fits_header(path) -> dict:
    """Primary FITS header as a dict (standard library only; values converted to int/float/bool/str).

    Handy for ``DATE-OBS`` (epoch), ``RA``/``DEC`` (pointing) and ``HIERARCH ESO INS PIXSCALE``.
    """
    hdr = {}
    with open(path, "rb") as f:
        while True:
            block = f.read(2880)
            if not block:
                break
            for i in range(0, len(block), 80):
                card = block[i : i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    return hdr
                if key == "HIERARCH" and "=" in card:
                    key, _, rest = card[9:].partition("=")
                    key = key.strip()
                elif card[8:10] == "= ":
                    rest = card[10:]
                else:
                    continue
                hdr[key] = _fits_value(rest)
    return hdr


def _fits_value(rest: str):
    rest = rest.strip()
    if rest.startswith("'"):
        end = rest.find("'", 1)
        while end != -1 and rest[end + 1 : end + 2] == "'":  # escaped quote
            end = rest.find("'", end + 2)
        return rest[1:end].replace("''", "'").rstrip()
    value = rest.split("/")[0].strip()
    if value in ("T", "F"):
        return value == "T"
    for conv in (int, float):
        try:
            return conv(value)
        except ValueError:
            pass
    return value

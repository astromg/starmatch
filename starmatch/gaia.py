"""Download a Gaia DR3 catalogue (cone search) from VizieR TAP - standard library only."""

from __future__ import annotations

import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from .sky import GAIA_EPOCH, SkyCatalog, load_sky_catalog

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA_TABLE = '"I/355/gaiadr3"'


def gaia_query_adql(ra: float, dec: float, radius_arcmin: float, mag_limit: float | None = None) -> str:
    """ADQL query for Gaia DR3 stars within ``radius_arcmin`` of (``ra``, ``dec``) [deg]."""
    where = (
        f"1=CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_arcmin / 60:.8f}))"
    )
    if mag_limit is not None:
        where += f" AND Gmag < {mag_limit}"
    return f"SELECT Source, RA_ICRS, DE_ICRS, Gmag, pmRA, pmDE FROM {GAIA_TABLE} WHERE {where} ORDER BY Gmag"


def query_gaia(
    ra: float,
    dec: float,
    radius_arcmin: float = 5.0,
    mag_limit: float | None = None,
    url: str = VIZIER_TAP,
    timeout: float = 120,
) -> SkyCatalog:
    """Gaia DR3 cone search (VizieR). Positions at epoch 2016.0 with proper motions [mas/yr]."""
    data = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": gaia_query_adql(ra, dec, radius_arcmin, mag_limit),
        }
    ).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=timeout) as resp:
        text = resp.read().decode()
    if not text.lstrip().lower().startswith(("source", '"source')):
        raise RuntimeError(f"unexpected response from {url}: {text[:200]!r}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gaia.csv"
        p.write_text(text)
        cat = load_sky_catalog(p, epoch=GAIA_EPOCH)
    cat.path = None
    return cat

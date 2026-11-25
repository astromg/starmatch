"""StarMatch - 2nd-order coordinate transformation between two star lists.

>>> from starmatch import match
>>> r = match("reference.ap", "field.out", pixscale=0.368)
>>> r.transform.field_to_ref(x, y)
"""

from .core import MatchError, MatchResult, StarMatch, match
from .io import (
    Catalog,
    load_catalog,
    load_file,
    load_xytr,
    loadap,
    loadout,
    loadxytr,
    read_fits_header,
    save_xytr,
    saveP2file,
)
from .sky import (
    SkyCatalog,
    SkyMatchResult,
    SkyTransform,
    load_sky_catalog,
    load_sky_transform,
    match_sky,
    save_sky_catalog,
)
from .transform import XYTransform, coo_trans, fit_poly, fit_poly2

__version__ = "0.3.0"

__all__ = [
    "match_sky",
    "SkyMatchResult",
    "SkyTransform",
    "SkyCatalog",
    "load_sky_catalog",
    "save_sky_catalog",
    "load_sky_transform",
    "match",
    "MatchResult",
    "MatchError",
    "Catalog",
    "load_catalog",
    "read_fits_header",
    "XYTransform",
    "coo_trans",
    "fit_poly",
    "fit_poly2",
    "load_xytr",
    "save_xytr",
    "StarMatch",
    # legacy
    "loadap",
    "loadout",
    "load_file",
    "loadxytr",
    "saveP2file",
]

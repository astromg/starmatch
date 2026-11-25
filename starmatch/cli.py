"""Command line interface.

    starmatch sky  FIELD CATALOG --pixscale 0.288 [--epoch 2012-10-12]   # pixels -> RA/Dec
    starmatch pix  REFERENCE FIELD [--pixscale 0.368]                     # pixels -> pixels
    starmatch gaia RA DEC [--radius 5] [--mag 20] -o gaia.txt             # download Gaia DR3

``starmatch REFERENCE FIELD`` (no sub-command) = ``starmatch pix`` (old syntax).
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from .core import MatchError, match
from .io import default_xytr_path, read_fits_header

COMMANDS = ("sky", "pix", "gaia")


def _legacy_args(argv):
    """Accept old ``pixscale=0.368 starratio=0.58`` style arguments."""
    out = []
    for a in argv:
        key, sep, value = a.partition("=")
        if sep and not a.startswith("-") and key in {"pixscale", "starratio"}:
            out += [f"--{key}", value]
        else:
            out.append(a)
    return out


def _common(p):
    p.add_argument("--nb-use", type=int, default=400, help="brightest stars for the initial solution (%(default)s)")
    p.add_argument("--axis-stars", type=int, default=None, help="neighbours tried as feature axis (sky: 2, pix: 1)")
    p.add_argument("--order", type=int, choices=(1, 2), default=None, help="polynomial order (default: auto)")
    p.add_argument(
        "--method", choices=("auto", "features", "triangles"), default="auto", help="initial solution (%(default)s)"
    )
    p.add_argument("--pairs", default=None, help="also write the list of matched stars to this file")
    p.add_argument("-q", "--quiet", action="store_true", help="print only errors")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="starmatch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sky", help="astrometry: match a pixel star list to a sky catalogue (e.g. Gaia)")
    s.add_argument("field", help="field star list (daophot .ap/.out/.coo/.lst or text ID X Y MAG)")
    s.add_argument("catalog", help="sky catalogue: ra dec mag [pmra pmdec] (text/CSV, Gaia/VizieR columns)")
    s.add_argument("--pixscale", type=float, required=True, help="pixel scale [arcsec/px], to ~3%%")
    s.add_argument("--epoch", default=None, help="epoch of observation (year or ISO date) for proper motions")
    s.add_argument("--fits", default=None, help="FITS image: epoch from DATE-OBS (unless --epoch)")
    s.add_argument("--flip", choices=("auto", "no", "yes"), default="auto", help="mirrored image (%(default)s)")
    s.add_argument("--tol", type=float, default=3.0, help="initial matching tolerance [px] (%(default)s)")
    s.add_argument("-o", "--output", default=None, help="output .radec file (default: <field>.radec)")
    s.add_argument("--wcs", default=None, help="also write FITS WCS header cards to this file")
    _common(s)

    x = sub.add_parser("pix", help="match two pixel star lists")
    x.add_argument("reference", help="reference star list")
    x.add_argument("field", help="field star list")
    x.add_argument("--pixscale", type=float, default=1.0, help="field/reference pixel scale ratio (HAWKI/SOFI 0.368)")
    x.add_argument(
        "--starratio", type=float, default=None, help="field/reference ratio of star numbers (default: auto)"
    )
    x.add_argument("--flip", choices=("auto", "no", "yes"), default="no", help="mirrored field (%(default)s)")
    x.add_argument("--tol", type=float, default=3.0, help="initial matching tolerance [reference px] (%(default)s)")
    x.add_argument("-o", "--output", default=None, help="output .xytr file (default: <field>.xytr)")
    x.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)  # deprecated, ignored
    _common(x)

    g = sub.add_parser("gaia", help="download Gaia DR3 stars around a position (VizieR)")
    g.add_argument("ra", type=float, nargs="?", help="RA [deg] (or --fits)")
    g.add_argument("dec", type=float, nargs="?", help="Dec [deg] (or --fits)")
    g.add_argument("--fits", default=None, help="FITS image: position from RA/DEC header keywords")
    g.add_argument("--radius", type=float, default=5.0, help="radius [arcmin] (%(default)s)")
    g.add_argument("--mag", type=float, default=None, help="G magnitude limit")
    g.add_argument("-o", "--output", required=True, help="output catalogue file")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


_FLIP = {"auto": None, "no": False, "yes": True}


def _run_sky(args, log):
    from .sky import match_sky

    epoch = args.epoch
    if epoch is None and args.fits:
        epoch = read_fits_header(args.fits).get("DATE-OBS")
        log.info("epoch from %s: %s", args.fits, epoch)
    r = match_sky(
        args.field,
        args.catalog,
        pixscale=args.pixscale,
        epoch=epoch,
        flip=_FLIP[args.flip],
        tol=args.tol,
        nb_use=args.nb_use,
        axis_stars=args.axis_stars or 2,
        order=args.order,
        method=args.method,
    )
    t = r.transform
    log.info("tangent point: RA=%.6f Dec=%.6f", t.ra0, t.dec0)
    log.info("residuals rms: %.3f arcsec (%d stars)", r.rms, r.n)
    if args.pairs:
        log.info("pairs saved: %s", r.save_pairs(args.pairs))
    if args.wcs:
        log.info("WCS saved: %s", r.save_wcs(args.wcs))
    from pathlib import Path

    log.info("saved: %s", r.save(args.output or Path(args.field).with_suffix(".radec")))


def _run_pix(args, log):
    r = match(
        args.reference,
        args.field,
        pixscale=args.pixscale,
        flip=_FLIP[args.flip],
        starratio=args.starratio,
        nb_use=args.nb_use,
        tol=args.tol,
        axis_stars=args.axis_stars or 1,
        order=args.order,
        method=args.method,
    )
    log.info("%s", r)
    t = r.transform
    with np.printoptions(precision=6, suppress=False):
        log.info("X field to reference: %s", t.fr_x)
        log.info("Y field to reference: %s", t.fr_y)
        log.info("X reference to field: %s", t.rf_x)
        log.info("Y reference to field: %s", t.rf_y)
    if args.pairs:
        log.info("pairs saved: %s", r.save_pairs(args.pairs))
    log.info("saved: %s", r.save(args.output or default_xytr_path(args.field)))


def _run_gaia(args, log):
    from .gaia import query_gaia
    from .sky import save_sky_catalog

    ra, dec = args.ra, args.dec
    if args.fits:
        h = read_fits_header(args.fits)
        ra, dec = h["RA"], h["DEC"]
        log.info("position from %s: RA=%.5f Dec=%.5f", args.fits, ra, dec)
    if ra is None or dec is None:
        raise SystemExit("starmatch gaia: give RA DEC or --fits")
    cat = query_gaia(ra, dec, args.radius, args.mag)
    log.info("Gaia DR3: %d stars, saved: %s", len(cat.ra), save_sky_catalog(args.output, cat))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _legacy_args(argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["pix", *argv]  # old syntax: starmatch reference field
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger("starmatch").setLevel(level)
    log = logging.getLogger("starmatch.cli")
    try:
        {"sky": _run_sky, "pix": _run_pix, "gaia": _run_gaia}[args.command](args, log)
    except MatchError as e:
        log.error("starmatch failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

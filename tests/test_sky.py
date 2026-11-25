from pathlib import Path

import numpy as np
import pytest

from starmatch import MatchError
from starmatch.cli import main
from starmatch.sky import (
    SkyCatalog,
    decimal_year,
    load_sky_catalog,
    load_sky_transform,
    match_sky,
    radec_to_std,
    save_sky_catalog,
    std_to_radec,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
FS001 = DATA / "FS001_121011_1_J.ap"
FS001_GAIA = DATA / "FS001_gaia.txt"


def synthetic_sky(
    n_cat=400,
    ra0=150.0,
    dec0=-30.0,
    radius_arcsec=400,
    pixscale=0.3,
    pa=35.0,
    flip=False,
    size=1024,
    noise_px=0.05,
    colour=0.3,
    epoch=None,
    seed=0,
    dist=0.0,
):
    """Random sky catalogue and an image of part of it with a known pixel -> sky relation.

    Returns (catalog, field (x, y, mag), truth(x, y) -> (ra, dec)).
    """
    rng = np.random.default_rng(seed)
    r = radius_arcsec * np.sqrt(rng.uniform(0, 1, n_cat))
    phi = rng.uniform(0, 2 * np.pi, n_cat)
    xi, eta = r * np.cos(phi), r * np.sin(phi)
    ra, dec = std_to_radec(xi, eta, ra0, dec0)
    mag = 12 + np.log10(1 + rng.uniform(0, 1, n_cat) * (10**2.4 - 1)) / 0.3
    pmra, pmdec = rng.normal(0, 20, n_cat), rng.normal(0, 20, n_cat)
    cat = SkyCatalog(ra, dec, mag, pmra, pmdec, epoch=2016.0)
    obs = cat if epoch is None else cat.at_epoch(epoch)

    a = np.radians(pa)
    s = -1.0 if flip else 1.0
    c = size / 2

    def sky_to_pix(ra_, dec_):
        x_, e_ = radec_to_std(ra_, dec_, ra0, dec0)
        # usual orientation: +y at PA, +x at PA - 90 (east counter-clockwise from north)
        u = (-x_ * np.cos(a) + e_ * np.sin(a)) * s / pixscale
        v = (x_ * np.sin(a) + e_ * np.cos(a)) / pixscale
        u = u + dist * (u / c) ** 2
        return u + c, v + c

    def truth(x, y):
        # numerical inverse of sky_to_pix (Newton on the small distortion)
        u, v = np.asarray(x, float) - c, np.asarray(y, float) - c
        uu = u.copy()
        for _ in range(20):
            uu = u - dist * (uu / c) ** 2
        u = uu * s * pixscale
        v = v * pixscale
        x_ = -u * np.cos(a) + v * np.sin(a)
        e_ = u * np.sin(a) + v * np.cos(a)
        return std_to_radec(x_, e_, ra0, dec0)

    x, y = sky_to_pix(obs.ra, obs.dec)
    inside = (x > 1) & (x < size) & (y > 1) & (y < size)
    fm = obs.mag[inside] + 1.2 + rng.normal(0, colour, inside.sum())
    field = (
        x[inside] + rng.normal(0, noise_px, inside.sum()),
        y[inside] + rng.normal(0, noise_px, inside.sum()),
        fm,
    )
    return cat, field, truth


def _err_arcsec(r, truth, field):
    ra, dec = r.transform.pix_to_sky(field[0], field[1])
    tra, tdec = truth(field[0], field[1])
    return np.max(np.hypot((ra - tra) * np.cos(np.radians(tdec)), dec - tdec)) * 3600


@pytest.mark.parametrize("pa,flip", [(0, False), (35, False), (270, False), (123, True), (0, True)])
def test_synthetic_orientation(pa, flip):
    cat, field, truth = synthetic_sky(pa=pa, flip=flip)
    r = match_sky(field, cat, pixscale=0.3)
    assert r.transform.flipped == flip
    assert r.transform.rotation == pytest.approx(pa, abs=0.05)
    assert r.transform.pixscale == pytest.approx(0.3, rel=1e-3)
    assert _err_arcsec(r, truth, field) < 0.05


def test_known_flip_is_faster_path():
    cat, field, _ = synthetic_sky(pa=10, flip=True)
    with pytest.raises(MatchError):
        match_sky(field, cat, pixscale=0.3, flip=False)
    assert match_sky(field, cat, pixscale=0.3, flip=True).transform.flipped


def test_proper_motion_epoch():
    cat, field, truth = synthetic_sky(epoch=2005.0, seed=2)
    r_good = match_sky(field, cat, pixscale=0.3, epoch=2005.0)
    assert r_good.rms < 0.05
    r_bad = match_sky(field, cat, pixscale=0.3)  # no correction: 11 yr x 20 mas/yr scatter
    assert r_bad.rms > 3 * r_good.rms


def test_distortion_second_order():
    cat, field, truth = synthetic_sky(n_cat=1500, dist=2.0, seed=3)
    r = match_sky(field, cat, pixscale=0.3)
    assert r.order == 2
    assert _err_arcsec(r, truth, field) < 0.05


@pytest.mark.parametrize("seed", range(5))
def test_few_stars_linear(seed):
    # ~20 catalogue stars in the image: the lower limit for reliable matching
    cat, field, truth = synthetic_sky(n_cat=120, seed=200 + seed, colour=0.5)
    assert len(field[0]) < 30
    r = match_sky(field, cat, pixscale=0.3)
    assert r.order == 1 and r.n >= 10
    assert _err_arcsec(r, truth, field) < 0.1


def test_wrong_pixscale_fails():
    cat, field, _ = synthetic_sky()
    with pytest.raises(MatchError):
        match_sky(field, cat, pixscale=0.4)


def test_unrelated_catalog_fails():
    cat, _, _ = synthetic_sky(seed=1)
    _, field, _ = synthetic_sky(seed=2)
    with pytest.raises(MatchError):
        match_sky(field, cat, pixscale=0.3)


def test_wcs_header_matches_astropy():
    wcs_mod = pytest.importorskip("astropy.wcs")
    cat, field, truth = synthetic_sky(n_cat=1500, dist=2.0, pa=77, seed=5)
    r = match_sky(field, cat, pixscale=0.3)
    w = wcs_mod.WCS(r.wcs_header())
    x, y = field[0], field[1]
    ra_w, dec_w = w.all_pix2world(x, y, 1)  # FITS convention: first pixel = 1
    ra, dec = r.transform.pix_to_sky(x, y)
    assert np.max(np.hypot((ra_w - ra) * np.cos(np.radians(dec)), dec_w - dec)) * 3600 < 0.001
    # inverse (uses AP/BP)
    xw, yw = w.all_world2pix(ra, dec, 1)
    assert np.max(np.hypot(xw - x, yw - y)) < 0.01


def test_projection_roundtrip():
    ra0, dec0 = 359.9, 89.0
    ra = np.array([359.95, 0.05, 10.0])
    dec = np.array([89.1, 88.9, 88.5])
    xi, eta = radec_to_std(ra, dec, ra0, dec0)
    ra2, dec2 = std_to_radec(xi, eta, ra0, dec0)
    np.testing.assert_allclose(np.cos(np.radians(ra2 - ra)), 1, atol=1e-14)
    np.testing.assert_allclose(dec2, dec, atol=1e-10)


def test_decimal_year():
    assert decimal_year(2012.5) == 2012.5
    assert decimal_year("2012-10-12T01:46:40") == pytest.approx(2012.78, abs=0.005)
    assert decimal_year("2016-01-01") == pytest.approx(2016.0)
    assert decimal_year("2012.78") == 2012.78


def test_load_catalog_formats(tmp_path):
    cat = load_sky_catalog(FS001_GAIA)  # plain columns with '--'
    assert len(cat.ra) == 83 and np.all(np.isfinite(cat.pmra))
    p = tmp_path / "vizier.csv"
    p.write_text("RA_ICRS,DE_ICRS,Gmag,pmRA,pmDE\n10.0,-20.0,15.1,1.5,-2.0\n10.1,-20.1,16.2,,\n")
    c2 = load_sky_catalog(p)
    np.testing.assert_allclose(c2.ra, [10.0, 10.1])
    np.testing.assert_allclose(c2.pmdec, [-2.0, 0.0])
    save_sky_catalog(tmp_path / "out.txt", c2)
    c3 = load_sky_catalog(tmp_path / "out.txt")
    np.testing.assert_allclose(c3.mag, c2.mag)


# ------------------------------------------------------------------ real data


def test_fs001_gaia():
    r = match_sky(FS001, FS001_GAIA, pixscale=0.288, epoch="2012-10-12T01:46:40")
    assert r.n >= 15
    assert r.rms < 0.1  # arcsec
    assert r.transform.pixscale == pytest.approx(0.288, rel=0.01)
    # same stars as the 2024 prototype (15 pairs, 0.05" rms): all of them found
    assert r.cat_id is None and len(set(r.field_idx)) == r.n


def test_save_load_and_cli(tmp_path):
    r = match_sky(FS001, FS001_GAIA, pixscale=0.288, epoch=2012.78)
    t2 = load_sky_transform(r.save(tmp_path / "f.radec"))
    np.testing.assert_allclose(t2.pix_to_sky(500.0, 500.0), r.transform.pix_to_sky(500.0, 500.0))
    rc = main(
        [
            "sky",
            str(FS001),
            str(FS001_GAIA),
            "--pixscale",
            "0.288",
            "--epoch",
            "2012-10-12",
            "-o",
            str(tmp_path / "cli.radec"),
            "--wcs",
            str(tmp_path / "cli.wcs"),
            "--pairs",
            str(tmp_path / "cli.pairs"),
            "-q",
        ]
    )
    assert rc == 0
    assert "CTYPE1  = 'RA---TAN'" in (tmp_path / "cli.wcs").read_text()
    assert len((tmp_path / "cli.pairs").read_text().splitlines()) > 15


def test_read_fits_header(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    from starmatch import read_fits_header

    h = fits.Header()
    h["RA"] = (8.477, "pointing")
    h["DEC"] = -12.13281
    h["DATE-OBS"] = "2012-10-12T01:46:39.9985"
    h["HIERARCH ESO INS PIXSCALE"] = 0.288
    h["OBJECT"] = "it''s"
    fits.PrimaryHDU(header=h).writeto(tmp_path / "x.fits")
    r = read_fits_header(tmp_path / "x.fits")
    assert r["RA"] == 8.477 and r["DEC"] == -12.13281
    assert r["DATE-OBS"] == "2012-10-12T01:46:39.9985"
    assert r["ESO INS PIXSCALE"] == 0.288
    assert r["NAXIS"] == 0 and r["SIMPLE"] is True


def test_query_gaia_parses_vizier_csv(monkeypatch):
    import io as _io
    import urllib.request

    from starmatch import gaia

    csv = "Source,RA_ICRS,DE_ICRS,Gmag,pmRA,pmDE\n123,8.47,-12.13,14.68,157.5,-191.5\n124,8.48,-12.12,20.3,,\n"
    seen = {}

    def fake_urlopen(req, timeout):
        seen["data"] = req.data.decode()
        return _io.BytesIO(csv.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cat = gaia.query_gaia(8.477, -12.13281, radius_arcmin=5, mag_limit=20)
    assert "Gmag+%3C+20" in seen["data"] and "I%2F355%2Fgaiadr3" in seen["data"]
    assert list(cat.id) == ["123", "124"] and cat.epoch == 2016.0
    np.testing.assert_allclose(cat.pmra, [157.5, 0.0])


# ------------------------------------------------------------------ triangles (few stars)


@pytest.mark.parametrize("seed", range(6))
def test_very_few_stars_use_triangles(seed):
    # ~12 catalogue stars in the image: local features usually fail, triangles take over
    cat, field, truth = synthetic_sky(n_cat=60, seed=300 + seed, colour=0.5)
    assert len(field[0]) <= 16
    r = match_sky(field, cat, pixscale=0.3)
    assert _err_arcsec(r, truth, field) < 0.3
    assert r.pixel_match.method in ("features", "triangles")


def test_triangles_only_same_as_features_on_fs001():
    a = match_sky(FS001, FS001_GAIA, pixscale=0.288, epoch=2012.78, method="features")
    b = match_sky(FS001, FS001_GAIA, pixscale=0.288, epoch=2012.78, method="triangles")
    assert b.pixel_match.method == "triangles"
    assert sorted(zip(a.field_idx, a.cat_idx)) == sorted(zip(b.field_idx, b.cat_idx))


@pytest.mark.parametrize("n_cat", [30, 60, 120])
def test_unrelated_few_stars_fail(n_cat):
    for s in range(10):
        cat, _, _ = synthetic_sky(n_cat=n_cat, seed=s)
        _, field, _ = synthetic_sky(n_cat=n_cat, seed=3000 + s, pa=s * 7)
        if len(field[0]) < 3:
            continue
        with pytest.raises(MatchError):
            match_sky(field, cat, pixscale=0.3)

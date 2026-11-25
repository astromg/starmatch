from pathlib import Path

import numpy as np
import pytest

import starmatch
from starmatch import (
    Catalog,
    MatchError,
    StarMatch,
    XYTransform,
    coo_trans,
    load_catalog,
    load_xytr,
    match,
    save_xytr,
)
from starmatch.cli import main
from starmatch.core import feature_scores, local_features

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
SOFI_J1 = DATA / "CEP02_081212_1_J.ap"
SOFI_J2 = DATA / "CEP02_081215_1_J.ap"
HAWKI = DATA / "CEP02_151230_K_CHIP3.out"
SMC_K1 = DATA / "SMC24_131127_1_K.out"
SMC_K2 = DATA / "SMC24_181228_1_K.out"


def synthetic_pair(n=1500, size=1000.0, angle_deg=12.0, scale=1.0, shift=(35.0, -20.0), seed=0, noise=0.0):
    """Random reference field and the same stars rotated/scaled/shifted (field), with known truth."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)
    m = 12 + np.log10(1 + rng.uniform(0, 1, n) * (10**2.7 - 1)) / 0.3  # power-law luminosity function
    a = np.deg2rad(angle_deg)

    def truth(xr, yr):  # reference -> field
        return (
            (np.cos(a) * xr - np.sin(a) * yr) / scale + shift[0],
            (np.sin(a) * xr + np.cos(a) * yr) / scale + shift[1],
        )

    fx, fy = truth(x, y)
    fx = fx + rng.normal(0, noise, n)
    fy = fy + rng.normal(0, noise, n)
    return (x, y, m), (fx, fy, m + 0.3), truth


@pytest.mark.parametrize("angle,scale", [(0.0, 1.0), (12.0, 1.0), (-35.0, 1.0), (180.0, 1.0), (90.0, 0.5), (30, 2.7)])
def test_synthetic_transform_recovered(angle, scale):
    ref, field, truth = synthetic_pair(angle_deg=angle, scale=scale)
    r = match(ref, field, pixscale=scale)
    assert r.n > 0.95 * len(ref[0])
    np.testing.assert_array_equal(r.ref_idx, r.field_idx)  # synthetic: same star order
    tx, ty = truth(ref[0], ref[1])
    x, y = r.transform.ref_to_field(ref[0], ref[1])
    np.testing.assert_allclose(x, tx, atol=1e-6)
    np.testing.assert_allclose(y, ty, atol=1e-6)
    x, y = r.transform.field_to_ref(field[0], field[1])
    np.testing.assert_allclose(x, ref[0], atol=1e-6)
    np.testing.assert_allclose(y, ref[1], atol=1e-6)
    assert r.transform.scale == pytest.approx(scale, rel=1e-6)
    assert np.cos(np.radians(r.transform.rotation + angle)) == pytest.approx(1.0)


def test_noisy_positions():
    ref, field, truth = synthetic_pair(noise=0.2, seed=3)
    r = match(ref, field)
    tx, ty = truth(ref[0], ref[1])
    x, y = r.transform.ref_to_field(ref[0], ref[1])
    assert np.max(np.hypot(x - tx, y - ty)) < 0.1
    assert r.rms == pytest.approx(0.2 * np.sqrt(2), rel=0.2)  # 2D rms, noise only in the field


def test_few_stars():
    ref, field, _ = synthetic_pair(n=100, seed=1)
    r = match(ref, field)
    assert r.n >= 95


def test_deterministic():
    ref, field, _ = synthetic_pair(n=600, noise=0.1)
    a, b = match(ref, field), match(ref, field)
    np.testing.assert_array_equal(a.transform.rf_x, b.transform.rf_x)
    np.testing.assert_array_equal(a.ref_idx, b.ref_idx)


@pytest.mark.parametrize("n", [300, 3000, 10000])
def test_unrelated_fields_fail(n):
    ref, _, _ = synthetic_pair(n=n, seed=1)
    _, other, _ = synthetic_pair(n=n, seed=2, angle_deg=40)
    with pytest.raises(MatchError):
        match(ref, other)


def test_wrong_pixscale_fails():
    with pytest.raises(MatchError):
        match(SOFI_J2, HAWKI, pixscale=1.0)


def test_input_forms():
    ref, field, _ = synthetic_pair(n=500)
    expected = match(ref, field).transform.rf_x
    as_array = np.column_stack(ref)  # (N, 3)
    as_dict = {"X": field[0], "Y": field[1], "MAG": field[2]}
    for a, b in [(as_array, field), (ref, as_dict), (Catalog(*ref), field)]:
        np.testing.assert_allclose(match(a, b).transform.rf_x, expected)


def _check_starlist(g1, g2, err):
    """Original (loop) implementation of the feature comparison."""
    x1, y1 = g1
    x2, y2 = g2
    n_max = min(len(x1), len(x2))
    if n_max == 0:
        return 0.0
    score = 0
    for i in range(n_max):
        if abs(x1[i] - x2[i]) < err and abs(y1[i] - y2[i]) < err:
            score += n_max - i
        elif i < n_max - 1 and abs(x1[i] - x2[i + 1]) < err and abs(y1[i] - y2[i + 1]) < err:
            score += n_max - i
        elif i > 1 and abs(x1[i] - x2[i - 1]) < err and abs(y1[i] - y2[i - 1]) < err:
            score += n_max - i
    return score / (0.5 * n_max * (n_max + 1))


def test_vectorized_features_equal_original_algorithm():
    x, y, m = load_catalog(SMC_K1)
    o = np.argsort(m)[:300]
    x, y = x[o], y[o]
    f = local_features(x[:60], y[:60], x, y, 70.0)
    s = feature_scores(f, f, 0.05)
    lists = [(f[0][i, : f[2][i]], f[1][i, : f[2][i]]) for i in range(60)]
    s_ref = np.array([[_check_starlist(a, b, 0.05) for b in lists] for a in lists])
    np.testing.assert_allclose(s, s_ref)


# ---------------------------------------------------------------- real data


@pytest.mark.parametrize(
    "ref,field,pixscale,min_pairs,max_rms",
    [(SOFI_J1, SOFI_J2, 1.0, 600, 0.5), (SOFI_J2, HAWKI, 0.368, 300, 0.4), (SMC_K1, SMC_K2, 1.0, 1000, 0.4)],
)
def test_real_data(ref, field, pixscale, min_pairs, max_rms):
    r = match(ref, field, pixscale=pixscale)
    assert r.n > min_pairs
    assert r.rms < max_rms
    assert r.transform.scale == pytest.approx(pixscale, rel=0.01)
    assert r.ref_id is not None and len(r.ref_id) == r.n


def test_starmatch_class_compat():
    sm = StarMatch(pixscale=0.368, fieldStarsRatio=0.58, nb_use=400, seed=1, loud=True)
    sm.set_reference(*load_catalog(SOFI_J2))
    sm.set_field(*load_catalog(HAWKI))
    sm.go()
    assert len(sm.p_fr_x) == 6 and len(sm.p_rf_y) == 6
    assert len(sm.trianglesMatch_ref_x) == sm.result.n
    assert "matched pairs" in sm.mssg
    with pytest.warns(DeprecationWarning):
        assert sm.trainglesMatch_ref_x is sm.trianglesMatch_ref_x
    with pytest.raises(AttributeError):
        sm.no_such_attribute  # noqa: B018


def test_unknown_parameter():
    with pytest.raises(TypeError):
        StarMatch(pixelscale=1)


# ------------------------------------------------------------------------- io


def test_coo_trans_scalar_and_array():
    px = [1, 2, 0, 0, 0, 0]
    py = [0, 0, 1, 0, 0, 0]
    assert coo_trans(1, 5, px, py) == (3.0, 5.0)
    x, y = coo_trans([1, 2], [5, 6], px, py)
    np.testing.assert_array_equal(x, [3, 5])
    np.testing.assert_array_equal(y, [5, 6])


@pytest.mark.parametrize("path,n", [(SOFI_J1, 1361), (SMC_K1, 2265)])
def test_load_catalog(path, n):
    cat = load_catalog(path)
    x, y, m = cat
    assert len(x) == len(y) == len(m) == len(cat.id)
    assert abs(len(x) - n) <= 2
    assert x.dtype == float and cat.id.dtype.kind == "i"


def test_load_plain_text(tmp_path):
    p = tmp_path / "stars.txt"
    p.write_text("# id x y mag\nA 10.0 20.0 12.5\nB 30.0 40.0 13.5\n")
    cat = load_catalog(p)
    np.testing.assert_array_equal(cat.x, [10, 30])
    np.testing.assert_array_equal(cat.mag, [12.5, 13.5])
    assert list(cat.id) == ["A", "B"]


def test_xytr_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    t = XYTransform(*rng.normal(size=(4, 6)), reference="ref.ap", field="field.ap")
    path = save_xytr(tmp_path / "f.xytr", t)
    t2 = load_xytr(path)
    for a in ("fr_x", "fr_y", "rf_x", "rf_y"):
        np.testing.assert_array_equal(getattr(t, a), getattr(t2, a))
    assert (t2.reference, t2.field) == ("ref.ap", "field.ap")
    assert starmatch.loadxytr(path)[0] == "ref.ap"


def test_cli(tmp_path):
    out = tmp_path / "out.xytr"
    pairs = tmp_path / "out.pairs"
    rc = main([str(SOFI_J2), str(HAWKI), "pixscale=0.368", "-o", str(out), "--pairs", str(pairs), "-q"])
    assert rc == 0
    t = load_xytr(out)
    assert t.field.endswith("CEP02_151230_K_CHIP3.out")
    assert t.scale == pytest.approx(0.37, abs=0.01)
    lines = pairs.read_text().splitlines()
    assert lines[0].startswith("#") and len(lines) > 300


def test_cli_failure(tmp_path):
    assert main([str(SOFI_J2), str(HAWKI), "-o", str(tmp_path / "x.xytr"), "-q"]) == 1


@pytest.mark.parametrize("seed", range(10))
def test_nearest_neighbour_equals_brute_force(seed):
    from starmatch.core import nearest_neighbour

    rng = np.random.default_rng(seed)
    n, m, r = rng.integers(1, 2000), rng.integers(1, 1000), rng.uniform(0.3, 5)
    px, py = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    qx, qy = rng.uniform(-8, 108, m), rng.uniform(-8, 108, m)
    i, _ = nearest_neighbour(qx, qy, px, py, r)
    d = np.hypot(qx[:, None] - px[None], qy[:, None] - py[None])
    expected = np.where(d.min(1) <= r, d.argmin(1), -1)
    np.testing.assert_array_equal(i, expected)


def test_triangles_invariants():
    from starmatch.core import triangles

    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 100, 8), rng.uniform(0, 100, 8)
    v1, l1, r1, s1 = triangles(x, y)
    a = np.radians(33)
    x2, y2 = 2 * (np.cos(a) * x - np.sin(a) * y) + 5, 2 * (np.sin(a) * x + np.cos(a) * y) - 7
    v2, l2, r2, s2 = triangles(x2, y2)
    np.testing.assert_array_equal(v1, v2)  # same vertex order
    np.testing.assert_allclose(l2, 2 * l1)
    np.testing.assert_allclose(r2, r1)
    np.testing.assert_array_equal(s2, s1)
    _, _, _, s3 = triangles(x, -y)  # mirror image: orientation flips
    np.testing.assert_array_equal(s3, -s1)


@pytest.mark.parametrize("ref,field,pixscale", [(SOFI_J2, HAWKI, 0.368), (SMC_K1, SMC_K2, 1.0)])
def test_triangles_only_same_as_features(ref, field, pixscale):
    a = match(ref, field, pixscale=pixscale, method="features")
    b = match(ref, field, pixscale=pixscale, method="triangles")
    assert b.method == "triangles"
    np.testing.assert_array_equal(np.sort(a.ref_idx), np.sort(b.ref_idx))
    np.testing.assert_allclose(a.transform.rf_x, b.transform.rf_x)


def test_bad_method():
    with pytest.raises(ValueError):
        match(SOFI_J1, SOFI_J2, method="magic")

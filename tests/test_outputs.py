import csv
import io

import numpy as np
import pytest

from transport.export import csv_bytes, electron_row, photon_row
from transport.model import Slab, electron, photon, sample_photons
from transport.plots import (SAMPLE_LIMIT, attenuation, depth_histogram, electron_paths,
                             energy_profile, photon_paths)
from transport.reference import MATERIALS


@pytest.mark.parametrize("mat,t,n", [("water",8,10000),("air",0,1),("pb",100,400)])
def test_drawn_tracks_histogram_and_hvl_match_full_simulation(mat,t,n):
    slab = Slab(MATERIALS[mat],t)
    beam = photon(slab,500)
    sample = sample_photons(slab,beam,n,42)
    fig = photon_paths(slab,beam,sample)
    k = min(n,SAMPLE_LIMIT)
    end = min(t,8*beam.hvl) if t else 1.
    expected = np.minimum(sample.distances[:k],min(end,t))
    np.testing.assert_allclose(fig.data[0].x[1::3],expected)
    hit = sample.interacted[:k] & (sample.distances[:k]<=end)
    passed = ~sample.interacted[:k] & (t<=end)
    np.testing.assert_allclose(fig.data[1].x, expected[hit])
    np.testing.assert_allclose(fig.data[2].x, expected[passed])
    assert len(fig.data[0].x) == 3*k
    if t:
        frames = fig.layout.updatemenus[0].buttons[0].args[0]
        assert all(v == 0 for v in frames[0]["data"][0]["x"][1::3])
        for i in (0,1,2):
            assert list(fig.data[i].x) == list(frames[-1]["data"][i]["x"])
    hist = depth_histogram(slab,beam,sample)
    if not sample.interacted.any():
        assert not hist.data
    else:
        assert sum(hist.data[0].y) == n - sample.transmitted
    curve = attenuation(slab,beam,sample)
    assert tuple(curve.data[1].x) == (beam.hvl,) and tuple(curve.data[1].y) == (50,)


@pytest.mark.parametrize("thickness",[0.,.2601,.4367,8.])
def test_electron_path_and_curve_end_at_slab_or_range(thickness):
    slab = Slab(MATERIALS["water"],thickness)
    b = electron(slab,1000)
    fig = electron_paths(slab,b)
    path, end = fig.data
    assert end.x[0] == pytest.approx(min(thickness,b.range_cm))
    assert path.customdata[0] == 1000 and path.customdata[-1] == pytest.approx(b.end_kev)
    profile = energy_profile(slab,b)
    assert profile.data[0].x[-1] == end.x[0]
    assert profile.data[0].y[-1] == pytest.approx(b.end_kev)
    if thickness:
        frames = fig.layout.updatemenus[0].buttons[0].args[0]
        assert frames[0]["data"][1]["x"] == [0]
        assert frames[-1]["data"][1]["x"] == [end.x[0]]
        np.testing.assert_allclose(frames[-1]["data"][0]["customdata"],path.customdata)


def test_csv_round_trip_for_two_different_quantities():
    slab = Slab(MATERIALS["oil"],.12,1.25)
    gamma = photon(slab,662)
    draw = sample_photons(slab,gamma,20_000,818)
    beta = electron(slab,850)
    payload = csv_bytes([photon_row(slab,gamma,draw),electron_row(slab,beta)])
    assert payload.startswith(b"\xef\xbb\xbf")
    g,b = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    for r in (g,b):
        assert float(r["density_g_cm3"]) == slab.density
        assert float(r["thickness_cm"]) == .12
        assert r["model_version"] == "0.1.0" and r["reference"].startswith("https://physics.nist.gov/")
        assert r["reference_retrieved"] and "トリオレイン" in r["material_definition"]
    assert int(g["incident_photons"]) == draw.count
    assert int(g["rng_PCG64_seed"]) == draw.seed
    assert float(g["transmission_mc_fraction"]) == draw.fraction
    assert float(g["transmission_theory_fraction"]) == gamma.transmission
    assert float(b["csda_range_g_cm2"]) == beta.range_mass
    assert float(b["final_energy_keV"]) == beta.end_kev
    assert not b["transmission_mc_fraction"] and not g["csda_range_g_cm2"]

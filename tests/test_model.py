import csv
from math import exp, log, sqrt

import numpy as np
import pytest

from transport.model import Slab, electron, photon, residual_kev, sample_photons
from transport.reference import DATA, MATERIALS, electron_table, mass_attenuation

WATER = MATERIALS["water"]
SLAB = Slab(WATER, 5)


@pytest.mark.parametrize("key,collision,radiative,total,rng", [
    ("air",1.661,.01271,1.674,.4912), ("oil",1.881,.01024,1.892,.4270),
    ("water",1.849,.01280,1.862,.4367), ("bone",1.659,.01824,1.677,.4857),
    ("al",1.465,.02119,1.486,.5546), ("fe",1.308,.04160,1.350,.6159),
    ("pb",.9939,.1290,1.123,.7843)])
def test_estar_original_one_mev_rows(key, collision, radiative, total, rng):
    result = electron(Slab(MATERIALS[key], 0), 1000)
    assert (result.collision, result.radiative, result.total, result.range_mass) == (
        collision, radiative, total, rng)
    assert result.range_cm * MATERIALS[key].density == pytest.approx(rng)


@pytest.mark.parametrize("key", MATERIALS)
def test_reference_integrity_and_energy_dependence(key):
    table = electron_table(key)
    assert np.all(np.diff(table.energy) > 0) and np.all(np.diff(table.range_mass) > 0)
    assert np.allclose(table.collision + table.radiative, table.total, rtol=0.001)
    assert sum(w for _, w in MATERIALS[key].composition) == pytest.approx(1, abs=2e-6)
    assert mass_attenuation(key, 30) > mass_attenuation(key, 2000)
    for e, r in zip(table.energy, table.range_mass):
        if e >= .1:
            res = electron(Slab(MATERIALS[key], 1), 1000*e)
            assert res.range_mass == pytest.approx(r)


def test_xcom_separate_queries_at_49_off_grid_points():
    with (DATA/"xcom_check.csv").open() as f:
        points = list(csv.DictReader(f))
    assert len(points) == 49
    for row in points:
        got = mass_attenuation(row["material"], 1000*float(row["energy_MeV"]))
        assert got == pytest.approx(float(row["mass_mu_cm2_g"]), rel=.004)


def test_lead_k_edge_no_smoothing():
    assert mass_attenuation("pb", 88.00) == pytest.approx(1.910, abs=.001)
    assert mass_attenuation("pb", 88.0045) == pytest.approx(7.684)
    assert mass_attenuation("pb", 88.01) > 7.68
    assert mass_attenuation("water", 500) == .09687


@pytest.mark.parametrize("key", MATERIALS)
def test_one_hvl_reproducibility_and_binomial_agreement(key):
    unit = Slab(MATERIALS[key], 1)
    slab = Slab(unit.material, photon(unit, 500).hvl)
    beam = photon(slab, 500)
    assert beam.transmission == pytest.approx(.5)
    draw = sample_photons(slab, beam, 30_000, 77)
    assert abs(draw.fraction - .5) < 5 * draw.standard_error
    repeat = sample_photons(slab, beam, 200, 77)
    np.testing.assert_array_equal(draw.distances[:200], repeat.distances)
    np.testing.assert_array_equal(draw.interacted, draw.distances <= slab.thickness)
    assert draw.transmitted + draw.interacted.sum() == draw.count


def test_uncertainty_decreases_across_repeated_experiments():
    beam = photon(SLAB, 500)
    def ensemble(n):
        return np.array([sample_photons(SLAB, beam, n, i).fraction for i in range(128)])
    small, large = ensemble(100), ensemble(10_000)
    assert .07 < large.std(ddof=1)/small.std(ddof=1) < .14
    expected = sqrt(beam.transmission*(1-beam.transmission)/10000)
    assert abs(large.mean() - beam.transmission) < 4*expected/sqrt(len(large))
    assert sample_photons(SLAB,beam,10_000,10).standard_error == pytest.approx(expected)


@pytest.mark.parametrize("key", MATERIALS)
def test_straight_path_energy_budget_and_density_scaling(key):
    a = Slab(MATERIALS[key], .12)
    b = Slab(a.material, .06, 2.)
    beam_a, beam_b = photon(a, 632), photon(b, 632)
    assert beam_b.mass_mu == beam_a.mass_mu
    assert beam_b.transmission == pytest.approx(beam_a.transmission)
    assert beam_b.hvl == pytest.approx(beam_a.hvl/2)
    e, f = electron(a, 837.5), electron(b, 837.5)
    assert e.end_kev == pytest.approx(f.end_kev)
    assert f.range_cm == pytest.approx(e.range_cm/2)
    assert f.range_mass == e.range_mass
    xs = np.linspace(0, e.range_cm*1.5, 1201)
    profile = residual_kev(a, e.energy_kev, e.range_mass, xs)
    assert profile[0] == e.energy_kev and np.all(np.diff(profile) <= 0)
    assert np.all(profile[xs >= e.range_cm] == 0)
    assert e.loss_kev + e.end_kev == pytest.approx(e.energy_kev)


def test_inverse_range_at_measured_energy_and_terminal_interval():
    # Official water rows: R(1 MeV)=0.4367, R(0.5 MeV)=0.1766, R(0.01 MeV)=0.0002515.
    assert electron(Slab(WATER,.4367-.1766),1000).end_kev == pytest.approx(500)
    assert electron(Slab(WATER,.4367-.0002515/2),1000).end_kev == pytest.approx(5)
    at_stop = electron(Slab(WATER,.4367), 1000)
    assert at_stop.stopped and at_stop.end_kev == 0


def test_zero_thickness_and_change_of_seed():
    z = Slab(WATER, 0)
    beam = photon(z, 500)
    assert beam.transmission == 1 and beam.hvl == pytest.approx(log(2)/.09687)
    s = sample_photons(z, beam, 202, 0)
    assert s.transmitted == 202 and not any(s.interacted) and s.standard_error == 0
    assert not np.array_equal(s.distances,sample_photons(z,beam,202,1).distances)
    assert electron(z, 120).end_kev == 120 and not electron(z,120).stopped


def test_slab_extremes():
    thin = photon(Slab(MATERIALS["air"], .0001, .25), 2000)
    thick_slab = Slab(MATERIALS["pb"], 100000, 4)
    thick = photon(thick_slab, 30)
    assert 0 < thin.transmission < 1 and thick.transmission == 0
    assert sample_photons(thick_slab, thick, 1, 2**32-1).transmitted == 0
    assert electron(thick_slab, 2000).stopped


@pytest.mark.parametrize("func,args", [
    (Slab,(WATER,-.01)), (Slab,(WATER,float("nan"))), (Slab,(WATER,100001)),
    (Slab,(WATER,1,0)), (Slab,(WATER,1,float("inf"))),
    (photon,(SLAB,29.999)), (photon,(SLAB,2000.001)),
    (electron,(SLAB,99)), (electron,(SLAB,2001)),
    (sample_photons,(SLAB,photon(SLAB,100),0,1)),
    (sample_photons,(SLAB,photon(SLAB,100),200001,1)),
    (sample_photons,(SLAB,photon(SLAB,100),2,-1)),
    (sample_photons,(SLAB,photon(SLAB,100),True,1)),
    (mass_attenuation,("unknown",100))])
def test_invalid_parameters(func,args):
    with pytest.raises(ValueError):
        func(*args)

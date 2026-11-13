import csv
import io

import numpy as np
import pytest

from transport.beta_spectrum import Emissions, NUCLIDES, sample_emissions, spectrum
from transport.beta_transport import (INITIAL_LOW, NO_PATH, PATH_END, RESIDUAL_LOW, STOPPED,
                                      beta_table, current_beta_run, energy_after_path, transport_emissions)
from transport.beta_outputs import particle_csv, summary_rows
from transport.beta_plots import loss_figure, path_figure, range_figure, spectrum_figure
from transport.model import Slab
from transport.reference import MATERIALS


def raw_cdf(s, e):
    """Independent quadrature to x: full trapezoids plus the final trapezoid."""
    widths = np.diff(s.energy)
    cumulative = np.r_[0., np.cumsum(.5*(s.density[1:]+s.density[:-1])*widths)]
    i = np.clip(np.searchsorted(s.energy, e, side="right")-1, 0, len(widths)-1)
    return cumulative[i]+(e-s.energy[i])*(s.density[i]+s.pdf(e))/2


@pytest.mark.parametrize("name,points,end,area,ref", [
    ("Sr-90", 102, .546, 1.00120164, .195729),
    ("Y-90", 120, 2.2801, 1.0003691375, .9329049774)])
def test_evaluated_spectra_and_beta_branch_means(name, points, end, area, ref):
    s = spectrum(name)
    assert (len(s.energy), s.energy[-1]) == (points, end)
    assert np.trapezoid(s.density, s.energy) == pytest.approx(1., abs=1e-12)
    assert s.raw_area == pytest.approx(area, abs=1e-10)
    assert s.reference_mean == pytest.approx(ref, abs=1e-9)
    assert s.mean == pytest.approx(ref, rel=.003)
    # Independently integrate E*p(E) by Simpson (exact for a quadratic).
    x, y = s.energy, s.density
    moment = sum(np.diff(x)/6*(x[:-1]*y[:-1]+(x[:-1]+x[1:])*(y[:-1]+y[1:])+x[1:]*y[1:]))
    assert s.mean == pytest.approx(moment, abs=1e-13)
    # Exercise flat sections, rising/falling PDF, the vanishing endpoint.
    u = np.r_[0., 1e-12, .99999999999, 1., s.cdf, (s.cdf[1:]+s.cdf[:-1])/2]
    np.testing.assert_allclose(raw_cdf(s, s.quantile(u)), u, rtol=0., atol=5e-15)
    assert tuple(s.quantile([0., 1.])) == (0., end)
    assert s.pdf(end+.01) == 0.


def test_equilibrium_draws_match_both_continuous_cdfs_and_mixture():
    src = sample_emissions(100_000, 205)
    n_y = np.count_nonzero(src.isotope)
    assert abs(n_y-src.count/2) < 5*np.sqrt(src.count*.25)
    for j, name in enumerate(NUCLIDES):
        s = spectrum(name)
        values = np.sort(src.energy[src.isotope == j])
        assert 0 < values[0] < .01 and values[-1] < s.energy[-1]
        u = raw_cdf(s, values)
        ranks = np.arange(len(u))/len(u)
        # Two-sided Kolmogorov statistic; conservative fixed-seed CDF check.
        assert max(np.max(u-ranks), np.max(ranks+1/len(u)-u)) < .012
        assert abs(values.mean()-s.mean) < 5*values.std()/np.sqrt(len(u))
    fig = spectrum_figure(src)
    x,y = map(np.asarray, (fig.data[2].x, fig.data[2].y))
    np.testing.assert_allclose(y, .5*spectrum("Sr-90").pdf(x)+.5*spectrum("Y-90").pdf(x))
    assert np.trapezoid(y, x) == pytest.approx(1., abs=1e-12)
    assert np.sum(np.asarray(fig.data[3].width)*np.asarray(fig.data[3].y)) == pytest.approx(1.)


# NIST ESTAR original 2.500 MeV rows, in column order collision/radiative/total/R.
@pytest.mark.parametrize("material,values", [
    ("air",(1.712,.03437,1.747,1.374)), ("oil",(1.860,.02803,1.888,1.228)),
    ("water",(1.834,.03468,1.868,1.247)), ("bone",(1.656,.0484,1.704,1.380)),
    ("al",(1.493,.05605,1.549,1.550)), ("fe",(1.333,.1063,1.440,1.699)),
    ("pb",(1.044,.2866,1.331,2.016))])
def test_estar_knots_and_interpolation_at_y90_endpoint(material, values):
    t = beta_table(material)
    assert (t.energy[0], t.energy[-1], len(t.energy)) == (.01, 2.5, 38)
    slab = Slab(MATERIALS[material], 0)
    src = Emissions(42,np.ones(2,dtype=int),np.array([2.5, 2.2801]))
    run = transport_emissions(slab, src)
    assert (run.collision[0], run.radiative[0], run.total[0], run.mass_range[0]) == values
    for data, ntab, endpoint in zip((run.collision,run.radiative,run.total,run.mass_range),
                                    (t.collision,t.radiative,t.total,t.range_mass), values):
        f = np.log(2.2801/2.0)/np.log(2.5/2.0)
        assert data[1] == pytest.approx(ntab[-2]**(1-f)*endpoint**f, rel=1e-13)
    np.testing.assert_allclose(run.mass_range, run.path_cm*slab.density)


def test_water_range_inverse_and_energy_budget_with_known_estar_points():
    slab = Slab(MATERIALS["water"], .2601)
    src = Emissions(1,np.array([1]),np.array([1.]))
    r = transport_emissions(slab,src)
    assert r.mass_range[0] == r.path_cm[0] == .4367
    assert (r.collision[0],r.radiative[0],r.total[0]) == (1.849,.0128,1.862)
    assert r.status[0] == PATH_END
    assert r.end_energy[0] == pytest.approx(.5, abs=1e-12)  # water R(.5) = .1766
    assert r.loss[0] == pytest.approx(.5, abs=1e-12)
    half = transport_emissions(Slab(slab.material,.2601/2,2.),src)
    assert half.path_cm[0] == r.path_cm[0]/2
    np.testing.assert_array_equal(half.end_energy,r.end_energy)
    t = beta_table("water")
    s = np.array([0., .1, .2601, .4367-.0001, .4367, 1000.])
    res, status = energy_after_path(t,1.,1.,.4367,s)
    np.testing.assert_array_equal(status, [NO_PATH,PATH_END,PATH_END,RESIDUAL_LOW,STOPPED,STOPPED])
    assert np.isnan(res[3]) and np.all(np.diff(res[np.isfinite(res)]) <= 0)
    np.testing.assert_allclose(res[[0,2,4,5]],[1.,.5,0.,0.],atol=1e-12)


def test_low_energy_rows_retained_unknown_distinct_from_zero_and_missing_isotope():
    src = Emissions(9,np.zeros(3,dtype=np.uint8),np.array([.009, .010, .50]))
    water = MATERIALS["water"]
    run = transport_emissions(Slab(water,.0001),src)
    np.testing.assert_array_equal(run.status,[INITIAL_LOW,RESIDUAL_LOW,PATH_END])
    assert np.isnan(run.mass_range[0]) and np.isnan(run.end_energy[:2]).all()
    assert np.isnan(run.loss[:2]).all() and run.mass_range[1] == .0002515
    sr, y = summary_rows(run)
    assert (sr["n_emitted"],sr["n_range_known"],sr["n_exact_loss"]) == (3,2,1)
    assert sr["n_residual_below_table"] == 1 and y["n_emitted"] == 0
    assert y["median_path_cm"] is None and y["sample_mean_initial_MeV"] is None
    zero = transport_emissions(Slab(water,0.),src)
    np.testing.assert_array_equal(zero.end_energy,src.energy)
    np.testing.assert_array_equal(zero.loss,[0.,0.,0.])
    assert np.isnan(zero.path_cm[0])
    cutoff = transport_emissions(Slab(water,.0002515),src)
    assert cutoff.status[1] == STOPPED and cutoff.end_energy[1] == 0.
    data = list(csv.DictReader(io.StringIO(particle_csv(run,slice(None)).decode("utf-8-sig"))))
    assert data[0]["csda_path_cm"] == "" and data[1]["path_end_energy_MeV"] == ""
    assert data[0]["initial_energy_MeV"] == "0.009" and data[1]["transport_status"] == "residual_below_table"


def test_seed_page_and_material_reuse_and_restore():
    state, slab = {}, Slab(MATERIALS["oil"],.21)
    r = current_beta_run(state,slab,2_000,42)
    assert current_beta_run(state,slab,2_000,42) is r
    assert not r.emissions.energy.flags.writeable
    fig = path_figure(r,slice(100,120))
    for j in range(2):
        trace = fig.data[2*j+1]
        idx = np.asarray(trace.y,dtype=int)-1
        assert (idx>=100).all() and (idx<120).all()
        assert (r.emissions.isotope[idx]==j).all()
        np.testing.assert_array_equal(trace.customdata,r.emissions.energy[idx])
        np.testing.assert_array_equal(trace.x,np.minimum(slab.thickness,r.path_cm[idx]))
    assert sum(sum(layer.y) for layer in range_figure(r).data) == np.isfinite(r.path_cm).sum()
    assert len(loss_figure(r).data) >= 2
    other = current_beta_run(state,Slab(MATERIALS["pb"],.21),2_000,42)
    assert other.emissions is r.emissions and not np.allclose(r.path_cm,other.path_cm,equal_nan=True)
    changed = current_beta_run(state,slab,2_000,43)
    assert not np.array_equal(changed.emissions.energy,r.emissions.energy)
    prefix = current_beta_run(state,slab,2_100,42)
    np.testing.assert_array_equal(prefix.emissions.energy[:2_000],r.emissions.energy)
    same = current_beta_run(state,slab,2_000,42)
    assert particle_csv(same,slice(1,101)) == particle_csv(r,slice(1,101))


@pytest.mark.parametrize("count,seed", [(0,42),(100_001,42),(True,42),(10,-1),(10,2**32)])
def test_invalid_sample_inputs(count,seed):
    with pytest.raises(ValueError):
        sample_emissions(count,seed)

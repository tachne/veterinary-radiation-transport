"""Read the rendered marks and the public CSV, including ambiguous edge cases."""
import csv
import io
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from transport.beta_outputs import particle_csv, summary_rows
from transport.beta_plots import loss_figure, path_figure
from transport.beta_spectrum import Emissions, sample_emissions
from transport.beta_transport import transport_emissions
from transport.beta_view import EQUILIBRIUM_MODE
from transport.model import Slab
from transport.reference import MATERIALS


def water(energies, thickness):
    src = Emissions(42, np.array([int(e > .546) for e in energies]), np.array(energies))
    return transport_emissions(Slab(MATERIALS["water"], thickness), src)


def marks(fig):
    measured, unknown = {}, {}
    for trace in fig.data:
        if trace.mode != "markers":
            continue
        if trace.xaxis == "x2":
            assert trace.marker.symbol == "x" and trace.marker.color == "#646B72"
            unknown.update(zip(trace.y, trace.text))
        else:
            for pid, x, symbol, color in zip(trace.y, trace.x, trace.marker.symbol, trace.marker.color):
                assert pid not in measured
                measured[int(pid)] = (symbol, color, x)
    return measured, unknown


def decoded(run, selection):
    return list(csv.DictReader(io.StringIO(particle_csv(run, selection).decode("utf-8-sig"))))


def test_range_missing_is_not_a_distance_and_low_residual_keeps_its_exit():
    # NIST water: R(1 MeV) = 0.4367 cm. At 0.4366 cm the remaining range
    # is positive but below R(10 keV); it has an exit and no numeric residual.
    run = water([.009, .5, 1., 1.5], .4366)
    use = slice(0, 4)
    fig = path_figure(run, use)
    measured, unknown = marks(fig)
    assert measured == {2: ("circle", "#2463A3", .1766), 3: ("arrow-right", "#187044", .4366),
                        4: ("arrow-right", "#187044", .4366)}
    assert set(unknown) == {1, 3}
    assert "飛程算出不可" in unknown[1] and "残存エネルギー算出不可" in unknown[3]
    assert fig.layout.xaxis.domain[1] < fig.layout.xaxis2.domain[0] and not fig.layout.xaxis2.visible
    lines = {pid for trace in fig.data if trace.mode == "lines" for pid in trace.y if pid is not None}
    assert lines == set(measured)
    assert list(fig.layout.yaxis.tickvals) == [1, 2, 3, 4]
    low, stop, bounded, regular = decoded(run, use)
    assert low["csda_path_cm"] == low["path_end_energy_MeV"] == ""
    assert float(stop["path_end_energy_MeV"]) == 0 and stop["transport_status"] == "csda_path_exhausted"
    assert bounded["transport_status"] == "residual_below_table" and bounded["path_end_energy_MeV"] == ""
    assert float(bounded["csda_path_cm"]) > measured[3][-1] and float(regular["path_end_energy_MeV"]) > .01


@pytest.mark.parametrize("thickness,expected", [(.0002515, "circle"), (0., "arrow-right")])
def test_exact_stop_at_exit_and_zero_thickness_are_distinct(thickness, expected):
    run = water([.009, .010], thickness)
    fig = path_figure(run, slice(0, 2))
    measured, unknown = marks(fig)
    assert set(measured) == {2} and set(unknown) == {1}
    assert measured[2][::2] == (expected, thickness)
    low, edge = decoded(run, slice(0, 2))
    assert low["csda_path_cm"] == ""
    if thickness == 0:
        assert float(low["path_end_energy_MeV"]) == .009 and float(low["electron_energy_loss_MeV"]) == 0
        assert float(edge["path_end_energy_MeV"]) == .010 and edge["transport_status"] == "zero_thickness"
    else:
        assert low["path_end_energy_MeV"] == "" and float(edge["path_end_energy_MeV"]) == 0


def test_missing_only_last_page_has_no_path_or_fake_origin():
    run = water([.2, .005], 1.)
    fig = path_figure(run, slice(1, 2))
    measured, unknown = marks(fig)
    assert not measured and set(unknown) == {2}
    assert all(not len(tr.x) for tr in fig.data if tr.mode == "lines")
    assert tuple(fig.layout.yaxis.tickvals) == (2,)


def test_loss_curve_zero_uses_same_stop_mark_without_extrapolation():
    run = water([.5, 1.], 1.)
    fig = loss_figure(run)
    zeros = [tr for tr in fig.data if tr.mode == "markers"]
    assert {tr.x[0] for tr in zeros} == {.1766, .4367}
    assert all(tuple(tr.y) == (0.,) and (tr.marker.symbol, tr.marker.color) == ("circle", "#2463A3")
               for tr in zeros)
    assert all(min(tr.y) >= .01 for tr in fig.data if tr.mode == "lines")


def test_pages_match_csv_and_keep_v020_reference_statistics():
    run = transport_emissions(Slab(MATERIALS["water"], .2), sample_emissions(10_000, 42))
    before = summary_rows(run)
    for selection in (slice(19, 39), slice(2329, 2379), slice(9900, 10000)):
        measured, unknown = marks(path_figure(run, selection))
        rows = decoded(run, selection)
        assert set(measured) | set(unknown) == {int(r["particle_id"]) for r in rows}
        for r in rows:
            pid = int(r["particle_id"])
            assert (pid in unknown) == (r["path_end_energy_MeV"] == "")
            if r["csda_path_cm"] == "":
                assert pid not in measured
            else:
                assert measured[pid][2] == min(float(r["csda_path_cm"]), .2)
                assert measured[pid][0] == ("circle" if r["transport_status"] == "csda_path_exhausted"
                                            else "arrow-right")
    assert summary_rows(run) == before
    sr, y = before
    assert [(r["n_emitted"], r["n_range_known"], r["n_initial_below_table"], r["n_exact_loss"])
            for r in before] == [(4953, 4804, 149, 4804), (5047, 5028, 19, 5026)]
    assert [r["sample_mean_initial_MeV"] for r in (sr, y)] == pytest.approx([.19817987, .92578306])
    assert [r["median_path_cm"] for r in (sr, y)] == pytest.approx([.043262396, .382380459])


def test_screen_particle_table_and_unavailable_fractions_agree_with_csv():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    app.number_input(key="thickness").set_value(.2)
    app.radio(key="beta_source").set_value(EQUILIBRIUM_MODE).run()
    app.number_input(key="eq_first").set_value(2335).run()
    assert not app.exception
    summary = next(t.value for t in app.table if "残存E算出不可" in t.value.columns)
    assert list(summary["飛程算出不可（初期10 keV未満）"]) == ["149個（3.01%）", "19個（0.38%）"]
    assert list(summary["残存E算出不可"]) == ["149個（3.01%）", "21個（0.42%）"]
    table = next(t.value for t in app.dataframe if "計算状態" in t.value.columns)
    row = table.iloc[0]
    run = app.session_state["_beta_equilibrium_run"]
    exported = decoded(run, slice(2334, 2335))[0]
    assert int(exported["particle_id"]) == row["粒子番号"] == 2335
    assert row["経路長 (cm)"] == float(exported["csda_path_cm"]) > .2
    assert row["計算状態"] == "残存0超〜10 keV未満"
    assert np.isnan(row["残存 E (MeV)"]) and exported["path_end_energy_MeV"] == ""

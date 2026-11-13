import csv
import io
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.runtime import get_instance
from streamlit.testing.v1 import AppTest

from transport import tracks
from transport.beta_view import EQUILIBRIUM_MODE

APP = Path(__file__).resolve().parents[1] / "app.py"


def equilibrium_app():
    at = AppTest.from_file(str(APP),default_timeout=30).run()
    at.radio(key="beta_source").set_value(EQUILIBRIUM_MODE).run()
    assert not at.exception
    return at


def test_beta_paging_and_reirradiation_do_not_change_gamma_or_another_session(monkeypatch):
    monkeypatch.setattr(tracks.secrets,"randbelow",lambda limit: 42)
    a, b = equilibrium_app(), equilibrium_app()
    gamma = a.session_state["_gamma_irradiation"]
    original = a.session_state["_beta_equilibrium_run"]
    for size in (50,100,20):
        a.selectbox(key="eq_size").set_value(size).run()
        assert a.session_state["_beta_equilibrium_run"] is original
        assert any(f"粒子番号 1〜{size}" in m.value for m in a.markdown)
    a.button(key="eq_next").click().run()
    assert a.number_input(key="eq_first").value == 21
    a.button(key="eq_prev").click().run()
    assert a.number_input(key="eq_first").value == 1
    a.number_input(key="eq_first").set_value(9999).run()
    assert a.button(key="eq_next").disabled
    a.button(key="eq_reroll").click().run()
    after = a.session_state["_beta_equilibrium_run"]
    assert after.emissions.seed == a.number_input(key="eq_seed").value == 43
    assert not np.array_equal(after.emissions.energy,original.emissions.energy)
    a.number_input(key="eq_seed").set_value(42).run()
    restored = a.session_state["_beta_equilibrium_run"]
    np.testing.assert_array_equal(restored.emissions.energy,original.emissions.energy)
    assert a.session_state["_gamma_irradiation"] is gamma
    assert a.number_input(key="seed").value == b.number_input(key="eq_seed").value == 42
    assert b.number_input(key="eq_first").value == 1 and b.selectbox(key="eq_size").value == 20
    a.selectbox(key="material").set_value("oil")
    a.number_input(key="thickness").set_value(.2).run()
    other = a.session_state["_beta_equilibrium_run"]
    assert other.emissions is restored.emissions and other.slab.material.key == "oil"
    assert b.session_state["_beta_equilibrium_run"].slab.material.key == "water"
    a.number_input(key="eq_count").set_value(1).run()
    assert not a.exception and a.number_input(key="eq_first").value == 1
    assert a.button(key="eq_prev").disabled and a.button(key="eq_next").disabled


def test_beta_actual_downloads_match_visible_sample_and_selected_range(monkeypatch):
    downloads = {}
    original = st.download_button
    def capture(*args,**kwargs):
        if kwargs.get("key","").startswith("eq_") or kwargs.get("file_name") == "comparison.csv":
            payload = kwargs["data"] if "data" in kwargs else args[1]
            downloads[args[0]] = (kwargs["file_name"], payload, get_instance().media_file_mgr)
        return original(*args,**kwargs)
    monkeypatch.setattr(st,"download_button",capture)
    at = equilibrium_app()
    at.number_input(key="thickness").set_value(.2)
    at.number_input(key="eq_count").set_value(211)
    at.selectbox(key="eq_size").set_value(100).run()
    at.button(key="eq_next").click().run()
    assert not at.exception
    def decode(data):
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    page, complete = [], []
    for label,expected,output in (("表示中のβ粒子CSV","beta_s42_101-200.csv",page),
                                  ("全β粒子CSV","beta_s42_all_211.csv",complete)):
        name, _, manager = downloads[label]
        button = next(b for b in at.get("download_button") if b.label == label)
        url = manager.execute_deferred(button.proto.deferred_file_id)
        file = manager._storage.get_file(url.rsplit("/",1)[-1])
        assert name == file.filename == expected and file.mimetype == "text/csv"
        output.extend(decode(file.content))
    assert len(complete) == 211 and page == complete[100:200]
    run = at.session_state["_beta_equilibrium_run"]
    np.testing.assert_array_equal([float(r["initial_energy_MeV"]) for r in page],
                                  run.emissions.energy[100:200])
    for r in page:
        assert r["seed_PCG64"] == "42" and r["density_g_cm3"] == "1.0" and r["thickness_cm"] == "0.2"
        assert r["model_version"] and "0.5" in r["equilibrium"] and r["nuclide"] in r["spectrum_source"]
        assert r["estar_source"].startswith("https://physics.nist.gov/")
    summary = decode(downloads["核種別の集計CSV"][1])
    gamma, mono = decode(downloads["両モードの条件・結果をCSV保存"][1])
    assert gamma["radiation"] == "gamma" and gamma["app_version"] == "0.1.1"
    assert mono["radiation"] == "beta_monoenergetic_electron" and float(mono["energy_keV"]) == 1000.
    for r in summary:
        group = [p for p in complete if p["nuclide"] == r["nuclide"]]
        losses = [float(p["electron_energy_loss_MeV"]) for p in group if p["electron_energy_loss_MeV"]]
        assert int(r["n_emitted"]) == len(group)
        assert int(r["n_exact_loss"]) == len(losses)
        np.testing.assert_allclose(float(r["mean_loss_MeV_known_only"]),np.mean(losses))

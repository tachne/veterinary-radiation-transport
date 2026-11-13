from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app():
    at = AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    assert not at.exception
    return at


def read_metrics(at):
    return {m.label:m.value for m in at.metric}


def test_default_and_independent_student_sessions():
    first, second = app(), app()
    assert [t.label for t in first.tabs] == ["γ線の相互作用","β線の飛程","γ線・β線比較","物質特性"]
    assert len(first.get("download_button")) == 4
    first.selectbox(key="material").set_value("pb")
    first.number_input(key="thickness").set_value(0.)
    first.number_input(key="gamma_kev").set_value(88.01)
    first.number_input(key="count").set_value(1234)
    first.number_input(key="seed").set_value(10)
    first.toggle(key="virtual").set_value(True).run()
    first.slider(key="density_scale").set_value(2.).run()
    assert not first.exception
    a = read_metrics(first)
    assert a["一次線透過率 · 標本"] == a["一次線透過率 · 理論"] == "100.00%"
    assert a["CSDA飛程 · 質量厚さ"] == "0.7843 g/cm²"
    assert float(a["密度換算した経路長"].split()[0]) == pytest.approx(.7843/(11.34*2),rel=.001)
    assert a["試料内の経路終点エネルギー"] == "1000 keV"
    second.run()
    b = read_metrics(second)
    assert b["一次線透過率 · 標本"] == "46.42%"
    assert b["一次線透過率 · 理論"] == "46.07%"
    assert second.selectbox(key="material").value == "water"
    assert second.number_input(key="thickness").value == 8
    assert second.number_input(key="seed").value == 42
    assert not second.toggle(key="virtual").value


def test_common_thickness_energy_and_virtual_density_switch():
    at = app()
    at.number_input(key="thickness").set_value(.2601).run()
    assert read_metrics(at)["試料内の経路終点エネルギー"] == "500 keV"
    assert read_metrics(at)["一次光子が相互作用せず透過する確率"] == "97.51%"
    at.toggle(key="virtual").set_value(True).run()
    at.slider(key="density_scale").set_value(2.).run()
    assert read_metrics(at)["試料内の経路終点エネルギー"] == "0 keV"
    at.toggle(key="virtual").set_value(False).run()
    assert not at.slider
    assert read_metrics(at)["試料内の経路終点エネルギー"] == "500 keV"
    assert not at.exception

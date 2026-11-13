import csv
import io
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from transport import tracks
from transport.export import tracks_csv
from transport.model import Slab, photon
from transport.plots import photon_paths
from transport.reference import MATERIALS
from transport.tracks import current_irradiation, different_seed, track_slice

APP = Path(__file__).resolve().parents[1] / "app.py"
OIL = Slab(MATERIALS["oil"], 10.9)


@pytest.fixture
def run42():
    return current_irradiation({}, OIL, 510, 11_000, 42)


def read_csv(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


@pytest.mark.parametrize("first,size",[(1,20),(21,20),(101,100),(201,100),
                                       (5449,50),(9013,13),(10980,100),(11000,20)])
def test_global_ids_geometry_and_animation(run42,first,size):
    r = run42
    ids = np.arange(first,min(first+size, r.sample.count+1))
    ix = ids-1
    f = photon_paths(r.slab,r.beam,r.sample,first=first,size=size)
    lines, dots, arrows = f.data
    np.testing.assert_array_equal(lines.y[::3],ids)
    np.testing.assert_allclose(lines.x[1::3],np.minimum(r.sample.distances[ix],10.9))
    np.testing.assert_array_equal(dots.y,ids[r.sample.interacted[ix]])
    np.testing.assert_array_equal(arrows.y,ids[~r.sample.interacted[ix]])
    assert sum(np.asarray(lines.x[1::3]) > r.beam.hvl) == r.hvl_passed[ix].sum()
    frames=f.layout.updatemenus[0].buttons[0].args[0]
    for i,trace in enumerate(f.data):
        assert frames[-1]["data"][i]["x"] == list(trace.x)
    for i,trace in [(1,dots),(2,arrows)]:
        assert frames[-1]["data"][i]["y"] == list(trace.y)
        assert not any(frames[0]["data"][i]["marker"]["opacity"])
        assert all(frames[-1]["data"][i]["marker"]["opacity"])
        for frame in frames:
            front=max(v for v in frame["data"][0]["x"] if v is not None)
            assert list(np.asarray(trace.x) <= front) == frame["data"][i]["marker"]["opacity"]


def test_teacher_reference_and_changed_window(run42):
    assert run42.hvl_count == 5497 and run42.sample.transmitted == 4193
    for a,b,hvl,exit_count in [(0,20,8,5),(100,200,51,40)]:
        assert run42.hvl_passed[a:b].sum() == hvl
        assert (~run42.sample.interacted[a:b]).sum() == exit_count
    ids=np.flatnonzero(run42.hvl_passed[:20] & run42.sample.interacted[:20])+1
    np.testing.assert_array_equal(ids,[2,10,15])


@pytest.mark.parametrize("where",["zero","below","exact","above"])
def test_hvl_boundary_and_csv(where):
    hvl=photon(OIL,510).hvl
    t={"zero":0,"below":np.nextafter(hvl,0.),"exact":hvl,"above":10.9}[where]
    r=current_irradiation({},Slab(OIL.material,t),510,500,42)
    rows=read_csv(tracks_csv(r,slice(None)))
    if t < hvl:
        assert r.hvl_count is None and r.hvl_passed is None
        assert all(row["passed_hvl_in_sample"]=="" for row in rows)
    else:
        assert r.hvl_count==sum(int(row["passed_hvl_in_sample"]) for row in rows)
    if t==hvl:
        assert r.hvl_count == r.sample.transmitted
        assert all(row["passed_hvl_in_sample"]==row["transmitted_exit"] for row in rows)
    for row in rows:
        if int(row["transmitted_exit"]):
            assert row["first_interaction_depth_cm"]==""
            assert float(row["track_end_cm"])==t
        else:
            assert float(row["first_interaction_depth_cm"])==float(row["track_end_cm"])


def test_csv_selected_matches_full_field_by_field(run42):
    all_rows=read_csv(tracks_csv(run42,slice(0,11000)))
    subset=read_csv(tracks_csv(run42,track_slice(11000,101,100)))
    assert len(all_rows)==11000 and subset==all_rows[100:200]
    for i,row in enumerate(all_rows):
        assert int(row["photon_id"]) == i+1
        assert row["material"]=="油" and int(row["rng_PCG64_seed"])==42
        assert float(row["energy_keV"])==510 and float(row["thickness_cm"])==10.9
        d=run42.sample.distances[i]
        assert float(row["sampled_free_path_cm"])==d
        assert int(row["transmitted_exit"])==int(d>10.9)
        assert int(row["passed_hvl_in_sample"])==int(d>run42.beam.hvl)


@pytest.mark.parametrize("first,size",[(0,20),(11001,50),(1,0),(1,101)])
def test_invalid_view_range(first,size):
    with pytest.raises(ValueError):
        track_slice(11000,first,size)


def test_only_physical_inputs_request_a_new_sample(monkeypatch):
    state={}
    calls=[]
    original=tracks.sample_photons
    def count_draw(slab,beam,n,seed):
        calls.append((slab,beam.energy_kev,n,seed))
        return original(slab,beam,n,seed)
    monkeypatch.setattr(tracks,"sample_photons",count_draw)
    args=(OIL,510,11000,42)
    r=current_irradiation(state,*args)
    for first,width in [(1,20),(395,50),(10921,100)]:
        result=current_irradiation(state,*args)
        assert result is r
        photon_paths(r.slab,r.beam,r.sample,first=first,size=width)
        tracks_csv(r,track_slice(11000,first,width))
    assert len(calls)==1
    for changed in [(Slab(OIL.material,3),510,11000,42),
                    (Slab(OIL.material,3,2),510,11000,42),
                    (Slab(MATERIALS["al"],3,2),510,11000,42),
                    (Slab(MATERIALS["al"],3,2),511,11000,42),
                    (Slab(MATERIALS["al"],3,2),511,150,42),
                    (Slab(MATERIALS["al"],3,2),511,150,43)]:
        current_irradiation(state,*changed)
    assert len(calls)==7
    restored=current_irradiation(state,*args)
    np.testing.assert_array_equal(restored.sample.distances,r.sample.distances)


@pytest.mark.parametrize("previous,choice,expected",[
    (0,0,1),(2**32-1,2**32-2,2**32-2),(42,42,43),(42,41,41)])
def test_new_seed_excludes_current_even_at_boundaries(monkeypatch,previous,choice,expected):
    monkeypatch.setattr(tracks.secrets,"randbelow",lambda limit: choice)
    assert different_seed(previous)==expected


def app():
    at=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not at.exception
    return at


def test_navigation_reexposure_and_two_sessions(monkeypatch):
    monkeypatch.setattr(tracks.secrets,"randbelow",lambda limit: 42)
    a,b=app(),app()
    a.selectbox(key="material").set_value("oil")
    a.number_input(key="thickness").set_value(10.9)
    a.number_input(key="gamma_kev").set_value(510.)
    a.number_input(key="count").set_value(11000).run()
    before=a.session_state["_gamma_irradiation"]
    a.selectbox(key="track_preset").set_value(100).run()
    a.button(key="track_next").click().run()
    assert a.number_input(key="track_first").value==101
    assert a.session_state["_gamma_irradiation"] is before
    a.number_input(key="track_first").set_value(10973).run()
    assert a.button(key="track_next").disabled
    a.button(key="track_prev").click().run()
    assert a.number_input(key="track_first").value==10873
    a.button(key="reirradiate").click().run()
    after=a.session_state["_gamma_irradiation"]
    assert after.sample.seed == a.number_input(key="seed").value == 43
    assert after.beam==before.beam and after.slab==before.slab
    assert not np.array_equal(after.sample.distances,before.sample.distances)
    assert a.number_input(key="track_first").value==10873
    a.number_input(key="seed").set_value(42).run()
    np.testing.assert_array_equal(a.session_state["_gamma_irradiation"].sample.distances,before.sample.distances)
    assert b.number_input(key="track_first").value==1 and b.number_input(key="seed").value==42
    assert b.selectbox(key="material").value=="water" and b.selectbox(key="track_preset").value==20
    a.number_input(key="count").set_value(1).run()
    assert not a.exception and a.number_input(key="track_first").value==1
    assert a.button(key="track_prev").disabled and a.button(key="track_next").disabled


def test_events_exactly_at_hvl_or_exit(monkeypatch):
    from transport.model import Sample
    beam = photon(OIL,510)
    h,t = beam.hvl,OIL.thickness
    d = np.array([h, np.nextafter(h,np.inf), t, np.nextafter(t,np.inf)])
    controlled = Sample(42,d,d<=t,1,.25,0.)
    monkeypatch.setattr(tracks,"sample_photons",lambda *args:controlled)
    run = current_irradiation({},OIL,510,4,42)
    np.testing.assert_array_equal(run.hvl_passed,[False,True,True,True])
    rows=read_csv(tracks_csv(run,slice(None)))
    assert [r["transmitted_exit"] for r in rows]==["0","0","0","1"]
    assert [r["passed_hvl_in_sample"] for r in rows]==["0","1","1","1"]


def test_download_buttons_bind_the_displayed_snapshot(monkeypatch):
    import streamlit as st
    from streamlit.runtime import get_instance
    original=st.download_button
    exports={}
    managers={}
    def capture(*args,**kwargs):
        exports[kwargs["key"]]=(kwargs["file_name"],kwargs["data"])
        managers[kwargs["key"]]=get_instance().media_file_mgr
        return original(*args,**kwargs)
    monkeypatch.setattr(st,"download_button",capture)
    t=app()
    t.selectbox(key="material").set_value("oil")
    t.number_input(key="gamma_kev").set_value(510.)
    t.number_input(key="thickness").set_value(10.9)
    t.number_input(key="count").set_value(11000)
    t.session_state["track_exports"]=True
    t.selectbox(key="track_preset").set_value(100).run()
    t.button(key="track_next").click().run()
    assert not t.exception
    name,selected=exports["tracks_csv_view"]
    all_name,all_photons=exports["tracks_csv_all"]
    assert name=="photon_tracks_s42_101-200.csv" and "all_11000" in all_name
    for key, label, expected_file, payload in (
        ("tracks_csv_view","表示中の 100 個をCSV保存",name,selected),
        ("tracks_csv_all","全 11,000 個をCSV保存",all_name,all_photons)
    ):
        button=next(b for b in t.get("download_button") if b.label==label)
        manager=managers[key]
        url=manager.execute_deferred(button.proto.deferred_file_id)
        media=manager._storage.get_file(url.rsplit("/",1)[-1])
        assert (media.filename,media.mimetype,media.content)==(expected_file,"text/csv",payload())
    g=read_csv(exports["download_gamma_results.csv"][1])[0]
    assert (g["passed_hvl_count"],g["transmitted_primaries"])==("5497","4193")
    t.number_input(key="seed").set_value(69).run()
    t.number_input(key="track_first").set_value(10797).run()
    new_name,new_page=exports["tracks_csv_view"]
    assert new_name=="photon_tracks_s69_10797-10896.csv"
    assert all(r["rng_PCG64_seed"]=="69" for r in read_csv(new_page()))
    # A download already requested in the previous state retains its sample.
    page=read_csv(selected())
    assert page==read_csv(all_photons())[100:200]
    assert sum(int(r["passed_hvl_in_sample"]) for r in page)==51
    assert sum(int(r["transmitted_exit"]) for r in page)==40
    t.number_input(key="thickness").set_value(.1).run()
    g=read_csv(exports["download_gamma_results.csv"][1])[0]
    assert (g["hvl_in_sample"],g["passed_hvl_count"],g["hvl_survival_fraction"])==("0","","")


def test_maximum_count_last_photon_and_arbitrary_width():
    slab=Slab(MATERIALS["water"],8)
    run=current_irradiation({},slab,500,200_000,42)
    selected=track_slice(run.sample.count,199999,100)
    assert selected==slice(199998,200000)
    rows=read_csv(tracks_csv(run,selected))
    assert [int(r["photon_id"]) for r in rows]==[199999,200000]
    t=app()
    t.selectbox(key="track_preset").set_value("任意").run()
    t.number_input(key="track_custom").set_value(7)
    t.number_input(key="track_first").set_value(31).run()
    assert any("31〜37" in md.value for md in t.markdown)
    t.button(key="track_next").click().run()
    assert any("38〜44" in md.value for md in t.markdown)

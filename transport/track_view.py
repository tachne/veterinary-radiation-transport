from functools import partial

import numpy as np
import streamlit as st

from .export import tracks_csv
from .plots import number, percent, photon_paths
from .tracks import MAX_TRACKS, different_seed, track_slice


def reirradiate():
    st.session_state.seed = different_seed(st.session_state.seed)


def _go_to(first):
    st.session_state.track_first = first


def _rate_text(count, total):
    return "試料外" if count is None else f"{count:,} 個（{percent(count / total)}）"


@st.fragment
def photon_browser(run):
    """Only the selected rows and their plot rerun when navigating this fragment."""
    draw, beam, slab = run.sample, run.beam, run.slab
    total = draw.count
    with st.container(border=True):
        st.subheader("一次光子の飛跡")
        st.caption(f"{slab.material.name}　·　{number(beam.energy_kev)} keV　·　"
                   f"試料出口 {number(slab.thickness)} cm　·　シード {draw.seed}")
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            preset = st.selectbox("表示本数", (20, 50, 100, "任意"), key="track_preset", width=105)
            size = preset if preset != "任意" else st.number_input(
                "任意の本数", 1, MAX_TRACKS, 20, 1, key="track_custom", width=135)
            if st.session_state.get("track_first", 1) > total:
                st.session_state.track_first = total
            first = st.number_input("開始光子番号", 1, total, step=1, key="track_first", width=160)
            selected = track_slice(total, first, size)
            st.button("前ページ", key="track_prev", on_click=_go_to,
                      args=(max(1, first - size),), disabled=first == 1)
            st.button("次ページ", key="track_next", on_click=_go_to,
                      args=(selected.stop + 1,), disabled=selected.stop == total)
        count = selected.stop - selected.start
        st.markdown(f"**光子番号 {first:,}〜{selected.stop:,}**　｜　表示 {count:,} 個 / 全 {total:,} 個")
        local_hvl = (None if run.hvl_passed is None
                     else int(np.count_nonzero(run.hvl_passed[selected])))
        local_exit = count - int(np.count_nonzero(draw.interacted[selected]))
        hvl_label = f"HVL通過 · {number(beam.hvl)} cm"
        exit_label = f"出口透過 · {number(slab.thickness)} cm"
        st.table([
            {"対象":"全計算光子", "光子数":f"{total:,} 個",
             hvl_label:_rate_text(run.hvl_count, total), exit_label:_rate_text(draw.transmitted, total)},
            {"対象":"現在表示中", "光子数":f"{count:,} 個",
             hvl_label:_rate_text(local_hvl, count), exit_label:_rate_text(local_exit, count)},
        ])
        if run.hvl_passed is None:
            st.caption(f"HVL（{number(beam.hvl)} cm）は試料外です。出口までの透過を集計しています。")
        else:
            st.caption("HVL線を越えた後、試料内で初回相互作用した光子もHVL通過数に含みます。")
        fig = photon_paths(slab, beam, draw, first=first, size=size)
        with st.container(height=520 if count > 20 else "content", border=False):
            st.plotly_chart(fig, width="stretch", theme=None,
                            key=f"gamma_tracks_{first}_{selected.stop}_{draw.seed}",
                            config={"displaylogo":False,"responsive":True,"scrollZoom":False})
        st.markdown("**● 初回相互作用**（橙）　 **▸ 試料を透過した一次光子**（青緑）")
        st.caption("「飛跡を再生」で表示範囲の同じ標本を再生します。縦軸は全標本での光子番号です。"
                   "一次光子の追跡は初回相互作用または試料出口で終了します。")
        export = st.expander("光子ごとのCSV出力", key="track_exports", on_change="rerun")
        if export.open:
            with export:
                st.caption("試料内で初回相互作用した深さを記録します。出口透過時の相互作用深度と、"
                           "HVLが試料外の場合の通過判定は空欄です。")
                with st.container(horizontal=True):
                    st.download_button(
                        f"表示中の {count:,} 個をCSV保存",
                        data=partial(tracks_csv, run, selected), mime="text/csv",
                        file_name=f"photon_tracks_s{draw.seed}_{first}-{selected.stop}.csv",
                        key="tracks_csv_view", on_click="ignore")
                    st.download_button(
                        f"全 {total:,} 個をCSV保存",
                        data=partial(tracks_csv, run, slice(0, total)), mime="text/csv",
                        file_name=f"photon_tracks_s{draw.seed}_all_{total}.csv",
                        key="tracks_csv_all", on_click="ignore")

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

from .beta_outputs import particle_csv, summary_rows
from .beta_plots import loss_figure, path_figure, range_figure, spectrum_figure
from .beta_spectrum import MAX_COUNT, MAX_SEED, NUCLIDES
from .beta_transport import LABELS, current_beta_run
from .export import csv_bytes
from .tracks import different_seed, track_slice

EQUILIBRIUM_MODE = "Sr-90 / Y-90 放射平衡"


def _chart(fig, key):
    st.plotly_chart(fig, key="eq_"+key, width="stretch", theme=None,
                    config={"displaylogo": False, "scrollZoom": False, "responsive": True})


def _format(value, digits=5):
    return "—" if value is None else f"{value:.{digits}g}"


def _count_and_percent(count, total):
    return f"{count:,}個（{count/total:.2%}）" if total else "0個（対象なし）"


def _set_first(first):
    st.session_state.eq_first = first


def _reroll():
    st.session_state.eq_seed = different_seed(st.session_state.eq_seed)


@st.fragment
def particle_browser(run):
    src = run.emissions
    with st.container(horizontal=True, vertical_alignment="bottom"):
        size = st.selectbox("表示粒子数", (20, 50, 100), key="eq_size", width=125)
        if st.session_state.get("eq_first", 1) > src.count:
            st.session_state.eq_first = src.count
        first = st.number_input("開始粒子番号", 1, src.count, step=1, key="eq_first", width=155)
        selected = track_slice(src.count, first, size)
        st.button("前の粒子", key="eq_prev", disabled=first == 1,
                  on_click=_set_first, args=(max(1, first-size),))
        st.button("次の粒子", key="eq_next", disabled=selected.stop == src.count,
                  on_click=_set_first, args=(selected.stop+1,))
    st.markdown(f"**粒子番号 {first:,}〜{selected.stop:,}**　／　全 {src.count:,} 個　·　シード {src.seed}")
    st.markdown(":blue[● CSDA経路を消尽]　:green[➜ 試料出口で打ち切り]　:gray[× 算出不可]")
    st.caption("横軸はCSDA経路長を直線的に配置した教育用模式表示です。Sr-90は青緑の実線、Y-90は橙の破線です。"
               "実際の投影飛程や停止深度は表しません。緑の矢印は図の終点で、実際の電子透過を保証しません。")
    with st.container(height=460 if selected.stop-selected.start > 20 else "content", border=False):
        _chart(path_figure(run, selected), "paths")
    st.caption("右欄の×は距離を持ちません。初期10 keV未満で飛程を求められない粒子は×のみ、"
               "出口の位置が分かり残存エネルギーだけが表外の粒子は矢印と×を表示します。")
    data = pd.DataFrame({
        "粒子番号": np.arange(first, selected.stop+1),
        "核種": np.asarray(NUCLIDES)[src.isotope[selected]],
        "初期 E (MeV)": src.energy[selected],
        "CSDA (g/cm²)": run.mass_range[selected], "経路長 (cm)": run.path_cm[selected],
        "残存 E (MeV)": run.end_energy[selected], "損失 (MeV)": run.loss[selected],
        "計算状態": np.asarray(LABELS)[run.status[selected]],
    })
    st.dataframe(data, hide_index=True, width="stretch", column_config={
        col: st.column_config.NumberColumn(format="%.6g") for col in data.columns[2:-1]})
    st.caption("残存値が表の下限を下回り、CSDA経路がまだ尽きていない場合は「残存0超〜10 keV未満」です。"
               "空欄は算出対象外、0は計算で確定した値です。")
    with st.container(horizontal=True):
        for label, part, suffix in (
                ("表示中のβ粒子CSV", selected, f"{first}-{selected.stop}"),
                ("全β粒子CSV", slice(0, src.count), f"all_{src.count}")):
            st.download_button(label, data=partial(particle_csv, run, part), mime="text/csv",
                               file_name=f"beta_s{src.seed}_{suffix}.csv", on_click="ignore",
                               key="eq_dl_"+label)


def render_equilibrium(slab):
    st.subheader("Sr-90 / Y-90 の連続β線")
    st.markdown("Sr-90（28.79年）→ Y-90（64.1時間）→ 安定な Zr-90。"
                "両者の放射能を等しいと近似し、各粒子の由来を確率 1/2 ずつで抽選します。"
                "粒子数の比は標本ごとに変わります。")
    st.caption("永続平衡に近い試料を想定します。任意の時刻に成立する比ではありません。"
               "左欄の試料・厚さ・密度を使用します。「単色電子」のエネルギー入力はこの線源には適用しません。")
    # Keep beta and gamma seeds independent. When switching source modes,
    # recover the last beta sample defaults without writing to a gamma widget.
    previous = st.session_state.get("_beta_equilibrium_run")
    st.session_state.setdefault("eq_seed", 42 if previous is None else previous.emissions.seed)
    st.session_state.setdefault("eq_count", 10_000 if previous is None else previous.emissions.count)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        n = st.number_input("β粒子数", 1, MAX_COUNT, step=1000, key="eq_count", width=165)
        seed = st.number_input("β乱数シード", 0, MAX_SEED, step=1, key="eq_seed", width=165)
        st.button("新しい乱数でβ線を再照射", key="eq_reroll", on_click=_reroll)
    run = current_beta_run(st.session_state, slab, n, seed)
    rows = summary_rows(run)
    st.markdown("#### A　βエネルギースペクトル")
    _chart(spectrum_figure(run.emissions), "spectrum")
    st.caption("実線は各核種を面積1に規格化した評価スペクトル。破線は各核種に1/2の重みを付けた混合です。"
               "凡例「抽選した全粒子」をクリックすると、標本のヒストグラムを重ねられます。")
    st.table([{"核種": r["nuclide"], "生成数": f'{r["n_emitted"]:,}',
               "平均：スペクトル (MeV)": _format(r["spectrum_mean_MeV"]),
               "平均：今回の標本 (MeV)": _format(r["sample_mean_initial_MeV"]),
               "最大：表の上端 (MeV)": _format(r["endpoint_MeV"])} for r in rows])
    st.markdown("#### B　粒子別の経路とエネルギー")
    st.markdown(f"**{slab.material.name}**　·　厚さ {slab.thickness:g} cm　·　"
                f"密度 {slab.density:g} g/cm³　·　質量厚さ {slab.thickness*slab.density:.5g} g/cm²")
    particle_browser(run)
    st.markdown("#### C　同じ試料での核種比較")
    left, right = st.columns(2)
    with left:
        st.markdown("**CSDA経路長の分布**")
        _chart(range_figure(run), "ranges")
        st.caption("実線源のエネルギー分布が生む経路長の違いです。点線は試料厚。"
                   "同じ初期エネルギーでの飛程ストラグリングは含みません。")
    with right:
        st.markdown("**経路に沿ったエネルギー損失**")
        _chart(loss_figure(run), "loss")
        st.caption("各核種で、飛程を算出できた粒子の初期エネルギー中央値に最も近い1個を表示。"
                   "曲線は10 keV以上の範囲、青い丸は試料内でCSDA経路を使い切った点です。")
    st.table([{"核種": r["nuclide"], "飛程算出数": r["n_range_known"],
               "飛程算出不可（初期10 keV未満）": _count_and_percent(r["n_initial_below_table"], r["n_emitted"]),
               "残存E算出不可": _count_and_percent(r["n_emitted"]-r["n_exact_loss"], r["n_emitted"]),
               "経路長 ≤ 厚さ": r["n_path_le_thickness"], "経路長 > 厚さ": r["n_path_gt_thickness"],
               "経路長中央値 (cm)": _format(r["median_path_cm"]),
               "平均損失 (MeV)": _format(r["mean_loss_MeV_known_only"]),
               "損失算出数": r["n_exact_loss"]} for r in rows])
    st.caption("算出不可の割合は各核種の全生成数を分母とします。飛程・損失の代表値はそれぞれ算出できた粒子のみの値です。"
               "CSDA経路長は実際の投影飛程ではなく、"
               "経路長と厚さの比較は遮蔽材の実測透過率を示しません。")
    st.download_button("核種別の集計CSV", csv_bytes(rows), file_name=f"beta_summary_s{seed}.csv",
                       mime="text/csv", on_click="ignore", key="eq_summary")
    with st.expander("参照データ・計算法・扱わない現象"):
        st.markdown("連続スペクトル：[MIRDspecs Sr-90](https://mirdsoft.org/products/MIRDspecs/MIRDspecs_HTMLs/Sr-90.htm) ／ "
                    "[Y-90](https://mirdsoft.org/products/MIRDspecs/MIRDspecs_HTMLs/Y-90.htm)"
                    "（[ICRP Publication 107](https://www.icrp.org/publication.asp?id=ICRP%20Publication%20107)）。"
                    "密度を区分線形補間し、その積分の逆関数から抽選します。")
        st.markdown("[NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/method.html) の"
                    "10 keV〜2.5 MeVの表から両対数補間し、残存質量飛程 R(E₀) − ρs を逆引きします。"
                    "密度を変えた仮想試料では質量阻止能・質量飛程を固定します。")
        st.write("電子の損失には衝突と放射の両方を含みます。局所吸収線量、多重散乱、後方散乱、"
                 "制動放射光子の輸送は計算しません。特に鉛などでは電子の停止だけで遮蔽性能を判断できません。")

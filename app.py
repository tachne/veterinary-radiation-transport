# Finish pandas initialization before Plotly inspects its types in concurrent sessions.
import pandas as pd
import streamlit as st

from transport import APP_VERSION, MODEL_VERSION
from transport.beta_spectrum import LAB_VERSION
from transport.beta_view import EQUILIBRIUM_MODE, render_equilibrium
from transport.export import csv_bytes, electron_row, photon_row
from transport.model import Slab, electron, photon
from transport.plots import attenuation, depth_histogram, electron_paths, energy_profile, number, percent
from transport.track_view import photon_browser, reirradiate
from transport.tracks import current_irradiation
from transport.reference import CUSTOM_ESTAR_URL, ESTAR_URL, MATERIALS, RETRIEVED, XCOM_URL


st.set_page_config(page_title=f"放射線輸送ラボ v{LAB_VERSION} | Veterinary Radiation Transport Lab",
                   layout="wide", initial_sidebar_state="expanded")


def download(label, filename, rows):
    st.download_button(label, data=csv_bytes(rows), file_name=filename, mime="text/csv",
                       on_click="ignore", key="download_" + filename)


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, theme=None,
                    config={"displaylogo":False,"responsive":True,"scrollZoom":False})


def show_metrics(items):
    for col, (name, value) in zip(st.columns(len(items)), items):
        col.metric(name, value)


with st.sidebar:
    st.header("試料の設定")
    key = st.selectbox("遮蔽材", tuple(MATERIALS), index=2,
                       format_func=lambda k: MATERIALS[k].name, key="material")
    material = MATERIALS[key]
    st.caption(material.description)
    thickness = st.number_input("試料の厚さ (cm)", 0.0, 100_000.0, 8.0, .1, format="%.4f",
                                key="thickness")
    virtual = st.toggle("密度だけを変える", key="virtual")
    scale = st.slider("代表密度に対する倍率", .25, 4.0, 1.0, .05,
                      key="density_scale") if virtual else 1.0
    slab = Slab(material, thickness, scale)
    st.markdown(f"使用密度 **{number(slab.density)} g/cm³**")
    if virtual:
        st.caption("仮想試料。組成と質量係数を固定し、密度だけを換算に使います。")
    st.divider()
    st.subheader("入射エネルギー")
    g_energy = st.number_input("γ線 · 光子 (keV)", 30.0, 2000.0, 500.0, 10.0,
                               format="%.2f", key="gamma_kev")
    e_energy = st.number_input("β線 · 単色電子 (keV)", 100.0, 2000.0, 1000.0, 10.0,
                               format="%.2f", key="beta_kev")
    st.caption("試料は両モードで共通です。係数と組成は「物質特性」で確認できます。")

st.title("放射線輸送ラボ")
st.caption(f"Veterinary Radiation Transport Lab　·　v{LAB_VERSION}　｜　均一な試料中の一次元輸送")
tab_g, tab_e, tab_c, tab_m = st.tabs(
    ["γ線の相互作用", "β線の飛程", "γ線・β線比較", "物質特性"])
beta = electron(slab, e_energy)

with tab_g:
    with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
        n = st.number_input("入射光子数", 1, 200_000, 10_000, 1000, key="count",
                            width=170)
        seed = st.number_input("乱数シード", 0, 2**32-1, 42, 1, key="seed", width=170)
        st.button("新しい乱数で再照射", key="reirradiate", on_click=reirradiate,
                  help="物質・厚さ・エネルギー・光子数を保ち、新しいシードで全光子を計算します。")
    run = current_irradiation(st.session_state, slab, g_energy, n, seed)
    beam, draw = run.beam, run.sample
    expected_percent = percent(beam.transmission) if beam.transmission else "< 10⁻³⁰⁰ %"
    gamma_csv = {
        **photon_row(slab, beam, draw), "app_version":APP_VERSION,
        "hvl_in_sample":int(run.hvl_passed is not None), "passed_hvl_count":run.hvl_count,
        "hvl_survival_fraction":None if run.hvl_count is None else run.hvl_count / n,
    }
    st.caption(f"一次線透過率の判定位置は試料出口（{number(thickness)} cm）です。")
    show_metrics([("一次線透過率 · 標本",percent(draw.fraction)),
                  ("一次線透過率 · 理論",expected_percent),
                  ("半価層 HVL",f"{number(beam.hvl)} cm")])
    photon_browser(run)
    st.write(f"一次線から除去 **{n-draw.transmitted:,} 個**　／　透過 **{draw.transmitted:,} 個**"
             f"　 ·　透過率の理論標準誤差 **{number(100*draw.standard_error)} パーセントポイント**")
    left, right = st.columns(2)
    with left:
        st.subheader("一次線の減弱")
        chart(attenuation(slab, beam, draw), "attenuation")
        st.caption("曲線は理論値、◇は試料出口の標本値。HVLは一次線の期待強度が入射時の50%になる厚さです。")
    with right:
        st.subheader("初回相互作用の深さ")
        chart(depth_histogram(slab, beam, draw), "depth_histogram")
        st.caption("棒は全入射光子のうち試料内で相互作用した個数、橙線は各区間の理論期待数です。")
    with st.expander("減弱係数・計算式"):
        st.latex(r"\mu=\rho(\mu/\rho),\quad T=e^{-\mu t},\quad"
                 r"\mathrm{HVL}=\ln(2)/\mu,\quad x=-\ln(U)/\mu")
        st.write(f"μ/ρ = {beam.mass_mu:.6g} cm²/g（干渉性散乱を含む全減弱）　"
                 f"μ = {beam.mu:.6g} cm⁻¹。厚さ t は cm、U は (0, 1) の一様乱数です。")
        st.caption("狭いビームの一次光子が対象です。透過割合の標準誤差は √[T(1−T)/N] です。")
    download("γ線の条件・結果をCSV保存","gamma_results.csv",[gamma_csv])

with tab_e:
    # The existing comparison tab always uses the monoenergetic electron row.
    beta_csv = electron_row(slab,beta)
    source = st.radio("β線源", ("単色電子", EQUILIBRIUM_MODE), horizontal=True, key="beta_source")
    if source == EQUILIBRIUM_MODE:
        render_equilibrium(slab)
    else:
        st.caption(f"単色電子　·　初期 {number(e_energy)} keV　｜　{material.description}")
        show_metrics([("CSDA飛程 · 質量厚さ",f"{number(beta.range_mass)} g/cm²"),
                      ("密度換算した経路長",f"{number(beta.range_cm)} cm"),
                      ("試料内の経路終点エネルギー",f"{number(beta.end_kev)} keV")])
        with st.container(border=True):
            st.subheader("電子の代表経路")
            st.caption("均一な物質を直進すると仮定した1本の経路。色はその位置での残存エネルギーを表します。"
                       "再生時間は表示用です。")
            chart(electron_paths(slab,beta), "beta_track")
            st.markdown("**試料内でCSDA終点に到達**" if beta.stopped else "**残存エネルギーをもって試料出口に到達**")
            st.caption(f"試料厚 {number(thickness)} cm ／ 代表経路長 {number(beta.range_cm)} cm"
                       f" = {number(thickness/beta.range_cm)}。CSDA飛程は平均的な減速に対応する経路長です。"
                       "多重散乱のある実際の電子では、進行方向への到達深さと経路長が異なります。")
        left, right = st.columns(2)
        with left:
            st.subheader("経路に沿ったエネルギー損失")
            chart(energy_profile(slab,beta), "energy_profile")
            st.caption(f"試料内のエネルギー損失 {number(beta.loss_kev)} keV。"
                       "残存質量飛程からESTAR表を逆引きし、最後の10 keV未満のみ原点まで線形近似します。")
        with right:
            st.subheader("入射時の質量阻止能")
            st.table([{"成分":name, "MeV cm²/g":f"{v:.5g}"}
                      for name,v in [("衝突",beta.collision),("放射",beta.radiative),("全阻止能",beta.total)]])
            share = beta.radiative / (beta.radiative+beta.collision)
            st.caption(f"放射成分は入射時の成分和の {percent(share)}。高原子番号物質では制動放射の寄与も考慮します。"
                       "放射損失は阻止能に含み、制動放射光子の輸送は扱いません。")
        with st.expander("電子モデルの適用範囲"):
            st.write("放射性核種のβ線には連続エネルギースペクトルがあります。ここでは初期エネルギーを"
                     "1つに固定し、直線CSDA近似で連続減速を計算します。電子透過率、飛程の分布、"
                     "線量分布、遮蔽の安全性は評価対象外です。")
        download("β線の条件・結果をCSV保存","electron_results.csv",[beta_csv])

with tab_c:
    st.subheader("同じ試料での比較")
    st.write(f"**{material.description}**　·　密度 {number(slab.density)} g/cm³"
             f"　·　厚さ {number(thickness)} cm　·　質量厚さ {number(thickness*slab.density)} g/cm²")
    left,right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**γ線　{number(g_energy)} keV**")
        st.metric("一次光子が相互作用せず透過する確率",expected_percent)
        st.write(f"半価層 {number(beam.hvl)} cm")
        st.caption(f"標本での透過は {draw.transmitted:,} / {draw.count:,} 個です。")
    with right, st.container(border=True):
        st.markdown(f"**単色電子　{number(e_energy)} keV**")
        st.metric("CSDAの代表経路長",f"{number(beta.range_cm)} cm")
        st.write(f"質量飛程 {number(beta.range_mass)} g/cm²")
        st.caption(f"同じ厚さでの直線経路終点のエネルギーは {number(beta.end_kev)} keV です。")
    st.caption("γ線の値は確率、電子の値は経路長です。電子の実際の透過割合を求めるには散乱を含む別の輸送計算が必要です。")
    download("両モードの条件・結果をCSV保存","comparison.csv",[gamma_csv,beta_csv])

with tab_m:
    st.subheader("7物質の参照特性")
    st.caption(f"γ線 {number(g_energy)} keV ／ 電子 {number(e_energy)} keV。表中の線減弱係数と経路長は"
               "各物質の代表密度で換算しています。")
    props = []
    compositions = []
    for mat in MATERIALS.values():
        ref_slab = Slab(mat,thickness)
        g = photon(ref_slab,g_energy)
        e = electron(ref_slab,e_energy)
        props.append({"物質":mat.name,"ρ (g/cm³)":mat.density,"μ/ρ (cm²/g)":g.mass_mu,
                      "μ (cm⁻¹)":g.mu,"衝突 (MeV cm²/g)":e.collision,
                      "放射 (MeV cm²/g)":e.radiative,"全阻止能 (MeV cm²/g)":e.total,
                      "CSDA (g/cm²)":e.range_mass, "経路長 (cm)":e.range_cm})
        compositions.append({"物質":mat.name,"代表組成":mat.description,
                             "元素質量分率":"　".join(f"{s} {v:g}" for s,v in mat.composition),
                             "組成表":mat.composition_url})
    st.dataframe(pd.DataFrame(props), hide_index=True, width="stretch", column_config={
        k:st.column_config.NumberColumn(format="%.6g") for k in props[0] if k != "物質"})
    st.markdown("**元素組成**")
    st.dataframe(pd.DataFrame(compositions),hide_index=True,width="stretch",column_config={
        "組成表":st.column_config.LinkColumn(display_text="出典")})
    if virtual:
        st.caption(f"操作中の仮想試料は {material.name} × {scale:g} 倍、密度 {number(slab.density)} g/cm³です。")
    st.caption("代表密度は実習指定値。油はトリオレイン、皮質骨はICRP組成を採用しました。"
               "密度だけを変える操作では基準の質量係数・質量飛程を使います。")
    with st.expander("参照表と補間"):
        st.write(f"取得日 {RETRIEVED}。γ線はXCOMの全質量減弱係数、電子はESTARの衝突・放射・全阻止能と"
                 "CSDA飛程を使用します。参照点の間は両対数線形補間です。鉛のK吸収端"
                 "（88.0045 keV）は両側を個別に補間し、端の値には直上側を採用します。")
        st.markdown(f"[NIST XCOM]({XCOM_URL})　·　[NIST ESTAR]({ESTAR_URL})　·　"
                    f"[ESTAR 任意組成の計算]({CUSTOM_ESTAR_URL})")
    download("物質特性をCSV保存","material_properties.csv",[
        {"model_version":MODEL_VERSION,"gamma_keV":g_energy,"electron_keV":e_energy,**p,**c,
         "参照表取得日":RETRIEVED, "XCOM":XCOM_URL,
         "ESTAR":CUSTOM_ESTAR_URL if m.key=="oil" else ESTAR_URL}
        for p,c,m in zip(props,compositions,MATERIALS.values())])

"""The horizontal coordinate is path length, not measured penetration depth."""
import numpy as np
import plotly.graph_objects as go

from .beta_spectrum import NUCLIDES, spectrum
from .beta_transport import LABELS, RESIDUAL_LOW, STOPPED, beta_table, energy_after_path
from .plots import AMBER, INK, TEAL, base, segments

COLORS = (TEAL, AMBER)
STOP_COLOR = "#2463A3"
EXIT_COLOR = "#187044"
UNKNOWN_COLOR = "#646B72"


def legend(fig):
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-.25, x=0),
                      margin_b=55)
    return fig


def spectrum_figure(src):
    fig = legend(base(350, "初期エネルギー (MeV)", "確率密度 (1/MeV)"))
    grid = np.unique(np.concatenate([spectrum(n).energy for n in NUCLIDES]))
    mixture = np.zeros_like(grid)
    for name, color in zip(NUCLIDES, COLORS):
        s = spectrum(name)
        fig.add_scatter(x=s.energy, y=s.density, mode="lines", name=name,
                        line=dict(color=color, width=2.4))
        mixture += .5*s.pdf(grid)
    fig.add_scatter(x=grid, y=mixture, mode="lines", name="平衡混合（1：1）",
                    line=dict(color=INK, width=2.6, dash="dash"))
    counts, edges = np.histogram(src.energy, bins=np.linspace(0, grid[-1], 49))
    fig.add_bar(x=(edges[1:]+edges[:-1])/2, y=counts/(src.count*np.diff(edges)),
                width=np.diff(edges), opacity=.28, marker_color=INK, visible="legendonly",
                name="抽選した全粒子")
    fig.update_yaxes(rangemode="tozero")
    return fig


def _mark_thickness(fig, thickness, xmax):
    if thickness <= xmax:
        fig.add_vline(x=thickness, line=dict(color=INK, width=1.5, dash="dot"))
    else:
        fig.add_annotation(x=1., xref="x domain", y=1.02, yref="paper", xanchor="right",
                           text=f"試料厚 {thickness:g} cm は表示範囲外 →", showarrow=False)


def path_figure(run, selected):
    src = run.emissions
    ids = np.arange(selected.start+1, selected.stop+1)
    status = run.status[selected]
    paths = run.path_cm[selected]
    known = np.isfinite(paths)
    ends = np.minimum(paths, run.slab.thickness)
    isotope = src.isotope[selected]
    fig = base(max(270, 125+17*len(ids)), "CSDA経路長 (cm)<br>直線配置した教育用模式表示", "粒子番号")
    for j, (name, color, dash) in enumerate(zip(NUCLIDES, COLORS, ("solid", "dash"))):
        mask = known & (isotope == j)
        xs, ys = segments(ends[mask], ids[mask])
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2, dash=dash),
                        name=name, hoverinfo="skip", showlegend=False)
        stopped = status[mask] == STOPPED
        labels = np.asarray(LABELS)[status[mask]]
        descriptions = ["CSDA経路を消尽・残存0 MeV" if is_stop else f"出口で打ち切り・{label}"
                        for is_stop, label in zip(stopped, labels)]
        fig.add_scatter(x=ends[mask], y=ids[mask], mode="markers", name=name, text=descriptions,
                        marker=dict(symbol=np.where(stopped, "circle", "arrow-right").tolist(),
                                    color=np.where(stopped, STOP_COLOR, EXIT_COLOR).tolist(), size=12),
                        customdata=src.energy[selected][mask],
                        hovertemplate=name+" #%{y}<br>%{text}<br>初期 %{customdata:.5g} MeV"
                                           "<br>表示経路 %{x:.5g} cm<extra></extra>")

    # This separate strip has no distance scale. A missing range must never
    # appear at a physical x, including x=0. A known exit can have unknown E.
    outside = ~known | (status == RESIDUAL_LOW)
    outside_notes = []
    for j, has_range, label in zip(isotope[outside], known[outside], np.asarray(LABELS)[status[outside]]):
        reason = "残存エネルギー算出不可" if has_range else "CSDA飛程算出不可（初期10 keV未満）"
        outside_notes.append(f"{NUCLIDES[j]} · {reason}<br>{label}")
    fig.add_scatter(x=np.full(outside.sum(), .5), xaxis="x2", y=ids[outside], mode="markers",
                    name="算出不可", marker=dict(color=UNKNOWN_COLOR, symbol="x", size=10),
                    text=outside_notes, customdata=src.energy[selected][outside],
                    hovertemplate="粒子 #%{y}<br>%{text}<br>初期 %{customdata:.5g} MeV<extra></extra>")
    fig.update_layout(xaxis2=dict(domain=[.86, 1.], range=[0., 1.], anchor="y",
                                  visible=False, fixedrange=True))
    fig.add_shape(type="rect", xref="x2 domain", x0=0, x1=1, yref="paper", y0=0, y1=1,
                  fillcolor="#F0F2F4", line_width=0, layer="below")
    fig.add_annotation(x=.5, xref="x2 domain", y=1.02, yref="paper", yanchor="bottom",
                       text="算出不可", font=dict(color=UNKNOWN_COLOR, size=12), showarrow=False)
    xmax = float(ends[known].max(initial=0.))
    xmax = xmax*1.08 if xmax > 0 else (min(1., run.slab.thickness) or .01)
    _mark_thickness(fig, run.slab.thickness, xmax)
    if run.slab.thickness <= xmax:
        fig.add_annotation(x=run.slab.thickness, xref="x", y=1.02, yref="paper", yanchor="bottom",
                           xanchor="right" if run.slab.thickness else "left", showarrow=False,
                           text=f"試料出口 {run.slab.thickness:g} cm", font=dict(color=INK, size=12))
    fig.update_layout(xaxis=dict(domain=[0., .80], range=[-.05*xmax, xmax]))
    fig.update_yaxes(range=[ids[-1]+.7, ids[0]-.7], tickmode="array", tickvals=ids,
                     tickfont=dict(size=11))
    return fig


def range_figure(run):
    fig = legend(base(345, "CSDA経路長 R/ρ (cm)", "粒子数"))
    paths = run.path_cm[np.isfinite(run.path_cm)]
    if not len(paths):
        fig.add_annotation(x=.5, y=.5, xref="paper", yref="paper", text="飛程を算出できる粒子がありません",
                           showarrow=False)
        return fig
    edges = np.linspace(0., paths.max()*1.01, 33)
    for j, (name, color) in enumerate(zip(NUCLIDES, COLORS)):
        vals = run.path_cm[(run.emissions.isotope == j) & np.isfinite(run.path_cm)]
        counts, _ = np.histogram(vals, bins=edges)
        fig.add_bar(x=(edges[:-1]+edges[1:])/2, y=counts, width=np.diff(edges),
                    name=name, marker_color=color, opacity=.64)
    fig.update_layout(barmode="overlay")
    _mark_thickness(fig, run.slab.thickness, edges[-1])
    return fig


def loss_figure(run):
    fig = legend(base(345, "電子がたどった経路 (cm)", "残存エネルギー (MeV)"))
    src, slab = run.emissions, run.slab
    table = beta_table(slab.material.key)
    for j, (name, color) in enumerate(zip(NUCLIDES, COLORS)):
        indices = np.flatnonzero((src.isotope == j) & np.isfinite(run.path_cm))
        if not len(indices):
            continue
        es = src.energy[indices]
        i = indices[np.argmin(abs(es-np.median(es)))]
        e, r = src.energy[i], run.mass_range[i]
        low_at = (r - table.range_mass[0])/slab.density
        upper = min(slab.thickness, low_at)
        x = np.linspace(0., upper, 80) if upper > 0 else np.array([0.])
        values, _ = energy_after_path(table, slab.density, e, r, x)
        if upper == low_at:
            values[-1] = table.energy[0]  # known reference point, including roundoff
        fig.add_scatter(x=x, y=values, mode="lines" if upper > 0 else "markers",
                        line=dict(color=color, width=2.7), marker_color=color, name=f"{name} #{i+1}")
        if slab.thickness >= run.path_cm[i]:
            fig.add_scatter(x=[run.path_cm[i]], y=[0.], mode="markers",
                            marker=dict(color=STOP_COLOR, symbol="circle", size=9),
                            showlegend=False, name=f"{name} CSDA終点")
    if not fig.data:
        fig.add_annotation(x=.5, y=.5, xref="paper", yref="paper", text="参照範囲内の粒子なし",
                           showarrow=False)
    fig.update_yaxes(rangemode="tozero")
    return fig

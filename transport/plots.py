"""Plot in slab coordinates. Vertical photon lanes only identify display samples."""
import numpy as np
import plotly.graph_objects as go

from .model import residual_kev
from .tracks import track_slice

TEAL = "#006B67"
TEAL_LIGHT = "#509B9C"
AMBER = "#AE4C1A"
INK = "#1D3745"
GRID = "#DFE8E8"
SLAB_FILL = "#EDF5F3"
SAMPLE_LIMIT = 20
ENERGY_SCALE = [(0, "#314660"), (.35, "#35798B"), (.65, "#36A69C"), (1, "#DE9F41")]


def base(height, x_title, y_title=None):
    fig = go.Figure()
    fig.update_layout(height=height, template="plotly_white", showlegend=False,
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(color=INK, size=13), hoverlabel=dict(font_size=13),
                      margin=dict(l=24, r=22, t=35, b=12))
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def number(x):
    return f"{x:.4g}"


def percent(p):
    if .99995 <= p < 1:
        return "> 99.99%"
    return f"{100*p:.3g}%" if 0 < p < .0001 else f"{100*p:.2f}%"


def boundary(fig, position, label, color, y=1.04, anchor="center"):
    fig.add_vline(x=position, line=dict(color=color, width=1.3, dash="dot"))
    fig.add_annotation(x=position, y=y, yref="paper", text=label, showarrow=False,
                       xanchor=anchor, yanchor="bottom", font=dict(color=color, size=11))


def replay(fig, frames, button_y=None):
    # Inline Plotly frames work without a server callback or an animation clock in Python.
    options = dict(mode="immediate", frame=dict(duration=70, redraw=False),
                   transition=dict(duration=0))
    fig.update_layout(margin_t=66, updatemenus=[dict(
        type="buttons", direction="left", showactive=False, x=0, xanchor="left",
        y=button_y if button_y is not None else (1.40 if fig.layout.height <= 210 else 1.14),
        yanchor="bottom",
        pad=dict(t=0,b=0), font=dict(size=12),
        bgcolor="#F3F8F7", bordercolor="#BFD2D0",
        buttons=[dict(label="▶　飛跡を再生", method="animate", args=[frames,options])])])


def segments(ends, lanes):
    xs, ys = [], []
    for end, lane in zip(ends, lanes):
        xs.extend([0, float(end), None])
        ys.extend([int(lane), int(lane), None])
    return xs, ys


def photon_paths(slab, beam, draw, *, first=1, size=SAMPLE_LIMIT):
    selected = track_slice(draw.count, first, size)
    y = np.arange(first, selected.stop + 1)
    height = max(320, 146 + 16 * len(y))
    fig = base(height, "試料の深さ x (cm)")
    top_label = 1 + 6 / (height - 146)
    t = slab.thickness
    # Follow the current slab. For extreme optical depths zoom to the entrance.
    span = min(t, 8*beam.hvl) if t else 1.0
    fig.add_vrect(x0=0, x1=min(t,span), fillcolor=SLAB_FILL, line_width=0, layer="below")
    distances = draw.distances[selected]
    ends = np.minimum(distances, span if t else 0.)
    xs, ys = segments(ends, y)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TEAL_LIGHT, width=1.8),
                             hoverinfo="skip", name="一次光子"))
    hit = draw.interacted[selected] & (distances <= span)
    passed = ~draw.interacted[selected] & (t <= span)
    for mask, style, name in [(hit, dict(symbol="circle", color=AMBER, size=8), "初回相互作用"),
                              (passed, dict(symbol="triangle-right", color=TEAL, size=9), "一次光子の透過")]:
        fig.add_trace(go.Scatter(x=ends[mask], y=y[mask], mode="markers", marker=style,
                                 name=name, hovertemplate=name+"<br>光子 %{y} · %{x:.5g} cm<extra></extra>"))
    boundary(fig, 0, "入射面" if t else "入射面＝出口", INK, y=top_label, anchor="left")
    if 0 < t <= span:
        label = "出口＝HVL" if t == beam.hvl else "出口"
        boundary(fig, t, f"{label} {number(t)} cm", INK, y=top_label, anchor="right")
    if beam.hvl < t:
        boundary(fig, beam.hvl, "HVL", AMBER, y=1, anchor="left")
    if t > span:
        fig.add_annotation(x=1, xref="paper", y=top_label, yref="paper", xanchor="right",
                           text=f"試料は右へ続く · 出口 {number(t)} cm", showarrow=False)
    fig.update_xaxes(range=[-.035*span,1.04*span])
    fig.update_yaxes(title_text="光子番号", range=[selected.stop + 1, first - 1],
                     tickmode="array", tickvals=y, ticktext=[str(i) for i in y],
                     tickfont=dict(size=11), showgrid=False)
    if t:
        frames = []
        for front in np.linspace(0, span, 22):
            path_x, _ = segments(np.minimum(ends, front), y)
            traces = [dict(x=path_x)]
            for mask in (hit, passed):
                # Empty frame arrays can leave old Plotly markers visible during replay.
                arrived = (ends[mask] <= front).astype(int).tolist()
                traces.append(dict(x=ends[mask].tolist(), y=y[mask].tolist(),
                                   marker=dict(opacity=arrived)))
            frames.append(dict(traces=[0,1,2], data=traces))
        replay(fig, frames, button_y=1 + 32 / (height - 146))
    return fig


def attenuation(slab, beam, draw):
    fig = base(280, "均一物質の厚さ (cm)", "一次線透過率 (%)")
    end = min(max(slab.thickness, 1.4*beam.hvl), 8*beam.hvl)
    x = np.linspace(0, end, 180)
    fig.add_trace(go.Scatter(x=x, y=100*np.exp(-beam.mu*x), mode="lines",
                             line=dict(color=TEAL,width=3), name="理論値"))
    boundary(fig, beam.hvl, f"HVL {number(beam.hvl)} cm", AMBER)
    fig.add_trace(go.Scatter(x=[beam.hvl], y=[50], mode="markers",
                             marker=dict(color=AMBER,size=9),name="半価層 · 50%"))
    if slab.thickness <= end:
        fig.add_trace(go.Scatter(x=[slab.thickness], y=[100*draw.fraction], mode="markers",
                                 marker=dict(symbol="diamond-open",color=INK,size=12,line_width=2),
                                 name="今回の試料の標本値"))
    else:
        fig.add_annotation(x=1,xref="paper",y=.75,yref="paper",xanchor="right",
                           text=f"試料出口 {number(slab.thickness)} cm は図外", showarrow=False)
    fig.update_xaxes(range=[-.025*end,1.035*end])
    fig.update_yaxes(range=[-3,104])
    return fig


def depth_histogram(slab, beam, draw):
    fig = base(280, "初回相互作用深度 (cm)", "光子数 / 区間")
    hits = draw.distances[draw.interacted]
    if not hits.size:
        fig.add_annotation(x=.5,y=.5,xref="paper",yref="paper",showarrow=False,
                           text="試料内の初回相互作用は 0 件です")
        fig.update_xaxes(range=[0,slab.thickness or 1])
        fig.update_yaxes(range=[0,1],tickvals=[0,1])
        return fig
    # All observed events are included, even deep in a very thick slab.
    end = min(slab.thickness, max(4*beam.hvl, float(np.max(hits))))
    counts, edges = np.histogram(hits,bins=np.linspace(0,end,25))
    centers = (edges[1:]+edges[:-1])/2
    fig.add_trace(go.Bar(x=centers,y=counts,width=.93*np.diff(edges),marker_color=TEAL_LIGHT,
                         name="標本", hovertemplate="%{y} 個<extra>標本</extra>"))
    expected = draw.count * np.exp(-beam.mu*edges[:-1]) * -np.expm1(-beam.mu*np.diff(edges))
    fig.add_trace(go.Scatter(x=centers,y=expected,mode="lines",line=dict(color=AMBER,width=2),
                             name="理論期待数", hovertemplate="%{y:.3g} 個<extra>理論期待数</extra>"))
    fig.update_yaxes(rangemode="tozero")
    return fig


def electron_paths(slab, beam):
    fig = base(210, "直線CSDA近似での経路長 s (cm)")
    t, r = slab.thickness, beam.range_cm
    # Preserve the stop point as a visible feature; label an off-screen slab exit.
    span = min(t,1.25*r) if t else max(r,1e-6)
    fig.add_vrect(x0=0, x1=min(t,span), fillcolor=SLAB_FILL, line_width=0,layer="below")
    xs = np.linspace(0, min(t,r), 130)
    energies = residual_kev(slab,beam.energy_kev,beam.range_mass,xs)
    fig.add_trace(go.Scatter(x=xs,y=np.zeros_like(xs),mode="lines+markers",
                             line=dict(color=TEAL,width=2),
                             marker=dict(color=energies, colorscale=ENERGY_SCALE, cmin=0,
                                         cmax=beam.energy_kev,size=7),
                             customdata=energies,name="代表経路",
                             hovertemplate="%{x:.5g} cm · %{customdata:.5g} keV<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[-1]],y=[0],mode="markers",
                             marker=dict(symbol="square" if beam.stopped else "triangle-right",
                                         size=12,color=INK),name="経路終点",
                             hovertemplate="%{x:.5g} cm<extra>経路終点</extra>"))
    boundary(fig, 0, "入射面" if t else "入射面＝出口", INK,anchor="left")
    if t and t <= span:
        boundary(fig,t,"試料出口",INK,anchor="right")
    if t > span:
        fig.add_annotation(x=1,xref="paper",y=1.04,yref="paper",xanchor="right",yanchor="bottom",
                           text=f"試料は右へ続く · 出口 {number(t)} cm",showarrow=False)
    label = "CSDA 終点" if beam.stopped else "出口での残存エネルギー"
    fig.add_annotation(x=xs[-1],y=-.35,text=f"{label} · {number(beam.end_kev)} keV",
                       xanchor="left" if t == 0 else "right", showarrow=False)
    fig.update_xaxes(range=[-.04*span,1.06*span])
    fig.update_yaxes(range=[-.7,.7],visible=False)
    if t:
        frames = []
        for stop in np.linspace(1,len(xs),22,dtype=int):
            frames.append(dict(traces=[0,1],data=[
                dict(x=xs[:stop].tolist(), y=[0]*stop, customdata=energies[:stop].tolist(),
                     marker=dict(color=energies[:stop].tolist())),
                dict(x=[float(xs[stop-1])], y=[0])]))
        replay(fig, frames)
    return fig


def energy_profile(slab, beam):
    fig = base(282, "試料内の経路長 s (cm)", "残存エネルギー (keV)")
    end = min(slab.thickness, beam.range_cm)
    x = np.linspace(0,end,220) if end else np.array([0.])
    y = residual_kev(slab,beam.energy_kev,beam.range_mass,x)
    fig.add_trace(go.Scatter(x=x,y=y,mode="lines" if end else "markers",
                             line=dict(width=3,color=TEAL), fill="tozeroy",
                             fillcolor="rgba(0,107,103,.09)",name="残存エネルギー"))
    fig.add_trace(go.Scatter(x=[end],y=[beam.end_kev],mode="markers",
                             marker=dict(color=AMBER,size=9),name="経路終点"))
    fig.update_xaxes(range=[0,(end or beam.range_cm)*1.035])
    fig.update_yaxes(range=[0,beam.energy_kev*1.05])
    return fig

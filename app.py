"""SpareTransformer dashboard (all data is synthetic).

Tabs: Command center (live failure dispatch on a city map), Results, Spare allocation,
Risk map, What-if lab, Method & limits. Reads saved outputs in results/ and data/.
"""
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "results" / "strategy_summary.csv"
PAIRED_CSV = ROOT / "results" / "paired_differences.csv"
FLEET_CSV = ROOT / "data" / "transformers.csv"

STRATEGIES = ["Equal split", "Proportional count", "Risk-based"]
COLORS = {"Equal split": "#94a3b8", "Proportional count": "#3b82f6", "Risk-based": "#10b981"}

try:
    from src.config import (CENTRAL_STORE_DELAY_HOURS, HANDLING_TIME_MAX_HOURS,
                            HANDLING_TIME_MIN_HOURS, MAX_BORROW_KM, ROAD_DISTANCE_FACTOR,
                            RURAL_DEADLINE_HOURS, RURAL_SPEED_KMH, URBAN_DEADLINE_HOURS,
                            URBAN_SPEED_KMH)
except Exception:  # fall back to the documented defaults
    CENTRAL_STORE_DELAY_HOURS, HANDLING_TIME_MIN_HOURS, HANDLING_TIME_MAX_HOURS = 12, 3, 6
    MAX_BORROW_KM, ROAD_DISTANCE_FACTOR = 80, 1.3
    RURAL_SPEED_KMH, URBAN_SPEED_KMH, RURAL_DEADLINE_HOURS, URBAN_DEADLINE_HOURS = 30, 40, 24, 8

st.set_page_config(page_title="SpareTransformer", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1300px;}
    .hero {background: linear-gradient(120deg,#0b1220 0%,#1e3a8a 55%,#0f766e 100%);
           padding: 26px 34px; border-radius: 18px; color: white; margin-bottom: 14px;}
    .hero h1 {margin:0; font-size:2.3rem; color:white;}
    .hero p {margin:6px 0 0 0; font-size:1.05rem; color:#cbd5e1;}
    .pill {display:inline-block; background:#fef3c7; color:#92400e; padding:5px 12px;
           border-radius:999px; font-size:.82rem; margin:12px 8px 0 0; font-weight:600;}
    .pill2 {background:#d1fae5; color:#065f46;}
    div[data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0;
           padding:14px 16px; border-radius:14px;}
    .callout {background:#ecfdf5; border-left:5px solid #10b981; padding:12px 16px;
           border-radius:8px; color:#064e3b; margin:8px 0;}
    .warn {background:#fffbeb; border-left:5px solid #f59e0b; padding:12px 16px;
           border-radius:8px; color:#78350f; margin:8px 0;}
    .legend span {margin-right:18px; font-size:.9rem;}
    .dot {display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_CSV)


@st.cache_data
def load_paired():
    df = pd.read_csv(PAIRED_CSV)
    if "comparison" in df:
        df["comparison"] = (df["comparison"].astype(str)
                            .str.replace("\u00e2\u02c6\u2019", "-", regex=False)
                            .str.replace("\u2212", "-", regex=False))
    return df


@st.cache_data
def load_fleet():
    return pd.read_csv(FLEET_CSV)


@st.cache_data
def depot_table():
    from src.data import DEPOTS
    rows = []
    if isinstance(DEPOTS, dict):
        for name, (lat, lon) in DEPOTS.items():
            rows.append({"name": name, "lat": lat, "lon": lon})
    else:
        for d in DEPOTS:
            rows.append({"name": d["name"], "lat": d["lat"], "lon": d["lon"]})
    return pd.DataFrame(rows)


@st.cache_data
def load_allocations():
    from src.allocate import get_all_allocations, load_data
    return pd.DataFrame(get_all_allocations(load_data())).reindex(columns=STRATEGIES)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


@st.cache_data
def road_distances(lat, lon, dlat, dlon):
    return haversine(np.asarray(lat)[:, None], np.asarray(lon)[:, None],
                     np.asarray(dlat)[None, :], np.asarray(dlon)[None, :]) * ROAD_DISTANCE_FACTOR


def dispatch_event(fleet, depots, stock0, scenario, seed):
    """One random failure event, served using the same rules as the simulation."""
    rng = np.random.default_rng(seed)
    target = 70.0 if scenario == "normal" else 140.0
    p = fleet["true_failure_prob"].to_numpy(dtype=float)
    p = np.clip(p * target / p.sum(), 0, 1)
    failed = np.flatnonzero(rng.random(len(fleet)) < p)
    rng.shuffle(failed)
    handling = rng.uniform(HANDLING_TIME_MIN_HOURS, HANDLING_TIME_MAX_HOURS, size=len(fleet))
    dist = road_distances(fleet["lat"].to_numpy(), fleet["lon"].to_numpy(),
                          depots["lat"].to_numpy(), depots["lon"].to_numpy())
    stock = np.array(stock0, dtype=int)
    rows = []
    for i in failed:
        rural = fleet["area_type"].iloc[i] == "rural"
        speed = RURAL_SPEED_KMH if rural else URBAN_SPEED_KMH
        deadline = RURAL_DEADLINE_HOURS if rural else URBAN_DEADLINE_HOURS
        served, km = -1, np.nan
        for d in np.argsort(dist[i]):
            if dist[i, d] > MAX_BORROW_KM:
                break
            if stock[d] > 0:
                stock[d] -= 1
                served, km = int(d), float(dist[i, d])
                break
        hours = km / speed + handling[i] if served >= 0 else float(CENTRAL_STORE_DELAY_HOURS)
        rows.append({"idx": int(i), "depot": served, "km": km, "hours": hours,
                     "on_time": hours <= deadline})
    return pd.DataFrame(rows), stock


def style(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


# ------------------------------------------------------------- header
st.markdown(
    """
    <div class="hero">
      <h1>⚡ SpareTransformer</h1>
      <p>Where should a power utility keep its limited spare distribution transformers?
         Risk-based allocation versus two baselines, tested by Monte Carlo simulation
         across eight depots in Rajasthan.</p>
      <span class="pill">Synthetic data</span><span class="pill">Assumed depot locations</span>
      <span class="pill pill2">500 simulated runs per strategy</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    summary = load_summary()
except Exception as exc:
    st.error(f"Could not read results/strategy_summary.csv: {exc}")
    st.stop()

with st.sidebar:
    st.header("Scenario")
    scenario = st.radio("Demand scenario", ["normal", "peak"],
                        format_func=lambda s: "Normal (about 70 failures)" if s == "normal"
                        else "Peak (about 140 failures)")
    st.caption("100 spares across 8 depots in the JVVNL region. Deadlines: 8 h urban, 24 h rural.")
    st.divider()
    st.caption("All transformer data and failure labels are synthetic. Depots are assumed at "
               "district HQ towns. Results are not real DISCOM performance.")

sel = summary[summary["scenario"] == scenario].set_index("strategy").reindex(STRATEGIES)
rb, eq = sel.loc["Risk-based"], sel.loc["Equal split"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Restored on time (risk-based)", f"{rb['restored_mean_pct']:.2f}%",
          f"{rb['restored_mean_pct'] - eq['restored_mean_pct']:+.2f} pp vs equal split")
k2.metric("Stock-outs per run (risk-based)", f"{rb['mean_stock_outs']:.1f}",
          f"{(rb['mean_stock_outs'] / eq['mean_stock_outs'] - 1) * 100:+.0f}% vs equal split"
          if eq["mean_stock_outs"] else None, delta_color="inverse")
if "p95_hours" in sel:
    k3.metric("Worst-case (p95) time", f"{rb['p95_hours']:.1f} h",
              f"{rb['p95_hours'] - eq['p95_hours']:+.1f} h vs equal split", delta_color="inverse")
    k4.metric("Average restoration time", f"{rb['mean_hours']:.2f} h",
              f"{rb['mean_hours'] - eq['mean_hours']:+.2f} h vs equal split", delta_color="inverse")

tabs = st.tabs(["Command center", "Results", "Spare allocation", "Risk map", "What-if lab",
                "Method & limits"])
tab_cc, tab_res, tab_alloc, tab_map, tab_lab, tab_method = tabs

VIEW = pdk.ViewState(latitude=26.15, longitude=76.45, zoom=6.2, pitch=30)


def make_deck(layers):
    return pdk.Deck(layers=layers, initial_view_state=VIEW, map_provider="carto",
                    map_style="road", tooltip={"html": "{label}"})


# ----------------------------------------------------------- command center
with tab_cc:
    st.subheader("Live failure dispatch")
    st.caption("Trigger a random failure event and watch spares being sent from depots. "
               "Switch the strategy to replay the same failures with a different spare split.")
    try:
        fleet, depots, alloc = load_fleet(), depot_table(), load_allocations()
        if "event" not in st.session_state:
            st.session_state.event = 1
        c1, c2, c3 = st.columns([2, 1.3, 1.3])
        strategy = c1.radio("Allocation strategy", STRATEGIES, index=2, horizontal=True)
        if c2.button("New failure event", type="primary"):
            st.session_state.event += 1
        show_heat = c3.checkbox("Show risk heatmap", value=True)

        stock0 = alloc[strategy].reindex(depots["name"]).fillna(0).astype(int).to_numpy()
        ev, left = dispatch_event(fleet, depots, stock0, scenario, st.session_state.event)

        risk_col = "predicted_risk" if "predicted_risk" in fleet else "true_failure_prob"
        layers = []
        if show_heat:
            heat = pd.DataFrame({"lat": fleet["lat"], "lon": fleet["lon"], "w": fleet[risk_col]})
            layers.append(pdk.Layer("HeatmapLayer", data=heat, get_position="[lon, lat]",
                                    get_weight="w", radius_pixels=45, opacity=0.55,
                                    color_range=[[255, 255, 178], [254, 204, 92], [253, 141, 60],
                                                 [240, 59, 32], [189, 0, 38]]))
        served = ev[ev["depot"] >= 0].copy()
        out = ev[ev["depot"] < 0].copy()
        if len(served):
            arcs = pd.DataFrame({
                "slon": fleet["lon"].to_numpy()[served["idx"]],
                "slat": fleet["lat"].to_numpy()[served["idx"]],
                "dlon": depots["lon"].to_numpy()[served["depot"]],
                "dlat": depots["lat"].to_numpy()[served["depot"]],
                "label": ["Served: " + str(round(k)) + " km road" for k in served["km"]]})
            layers.append(pdk.Layer("ArcLayer", data=arcs, get_source_position="[dlon, dlat]",
                                    get_target_position="[slon, slat]", get_width=2.5,
                                    get_source_color=[16, 185, 129, 220],
                                    get_target_color=[239, 68, 68, 220], pickable=True))
        fl = fleet.iloc[ev["idx"]]
        ok = pd.DataFrame({"lon": fl["lon"].to_numpy(), "lat": fl["lat"].to_numpy(),
                           "label": np.where(ev["depot"] >= 0, "Failed transformer (served)",
                                             "STOCK-OUT: no spare within range"),
                           "r": np.where(ev["depot"] >= 0, 0, 1)})
        layers.append(pdk.Layer("ScatterplotLayer", data=ok[ok.r == 0], get_position="[lon, lat]",
                                get_fill_color=[239, 68, 68, 230], get_radius=1500,
                                radius_min_pixels=4, pickable=True))
        layers.append(pdk.Layer("ScatterplotLayer", data=ok[ok.r == 1], get_position="[lon, lat]",
                                get_fill_color=[17, 24, 39, 240], get_line_color=[245, 158, 11, 255],
                                stroked=True, line_width_min_pixels=3, get_radius=2500,
                                radius_min_pixels=8, pickable=True))
        dep = depots.copy()
        dep["spares"] = stock0
        dep["left"] = left
        dep["label"] = [f"{n}: {s} spares, {l} left after this event"
                        for n, s, l in zip(dep["name"], dep["spares"], dep["left"])]
        dep["radius"] = 6000 + dep["spares"] * 800
        layers.append(pdk.Layer("ScatterplotLayer", data=dep, get_position="[lon, lat]",
                                get_radius="radius", radius_min_pixels=9,
                                get_fill_color=[30, 58, 138, 190], get_line_color=[255, 255, 255, 255],
                                stroked=True, line_width_min_pixels=2, pickable=True))
        layers.append(pdk.Layer("TextLayer", data=dep, get_position="[lon, lat]", get_text="name",
                                get_size=15, get_color=[15, 23, 42, 255], get_pixel_offset=[0, -26]))
        st.pydeck_chart(make_deck(layers))
        st.markdown(
            '<div class="legend"><span><span class="dot" style="background:#1e3a8a"></span>Depot (size = spares)</span>'
            '<span><span class="dot" style="background:#ef4444"></span>Failed transformer</span>'
            '<span><span class="dot" style="background:#10b981"></span>Spare dispatched (arc)</span>'
            '<span><span class="dot" style="background:#111827;border:2px solid #f59e0b"></span>Stock-out</span></div>',
            unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Failures this event", len(ev))
        m2.metric("Spares dispatched", len(served))
        m3.metric("Stock-outs", len(out))
        m4.metric("Restored on time", f"{100 * ev['on_time'].mean():.0f}%" if len(ev) else "-")
        st.caption("Event ID %d · scenario %s · strategy %s. Same event ID = same failures, so "
                   "switching strategy shows how the spare split changes the outcome. One random "
                   "event is illustrative; the Results tab averages 500 runs." %
                   (st.session_state.event, scenario, strategy))
    except Exception as exc:
        st.warning(f"Command center unavailable: {exc}")

# ---------------------------------------------------------------- results
with tab_res:
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for s in STRATEGIES:
        d = summary[summary["strategy"] == s]
        fig.add_bar(name=s, x=d["scenario"].str.title(), y=d["restored_mean_pct"],
                    marker_color=COLORS[s], text=d["restored_mean_pct"].round(1), textposition="outside")
    fig.update_layout(barmode="group", title="Restored within deadline (%)")
    fig.update_yaxes(range=[80, 101])
    c1.plotly_chart(style(fig))
    c1.caption("The y-axis starts at 80% to make differences visible.")
    fig = go.Figure()
    for s in STRATEGIES:
        d = summary[summary["strategy"] == s]
        fig.add_bar(name=s, x=d["scenario"].str.title(), y=d["mean_stock_outs"],
                    marker_color=COLORS[s], text=d["mean_stock_outs"].round(1), textposition="outside")
    fig.update_layout(barmode="group", title="Stock-outs per run (lower is better)")
    c2.plotly_chart(style(fig))
    c2.caption("Stock-out: no depot within 80 km had a spare.")
    st.markdown('<div class="callout"><b>Key insight:</b> placement helps, but stock size matters more. '
                'In the peak scenario even the best plan runs short, because 140 expected failures '
                'exceed 100 spares.</div>', unsafe_allow_html=True)
    try:
        p = load_paired()
        p = p[p["scenario"] == scenario]
        fig = go.Figure(go.Bar(
            y=p["comparison"], x=p["mean_difference_pp"], orientation="h", marker_color="#10b981",
            error_x=dict(type="data", symmetric=False,
                         array=p["ci95_high_pp"] - p["mean_difference_pp"],
                         arrayminus=p["mean_difference_pp"] - p["ci95_low_pp"])))
        fig.update_layout(title=f"Risk-based advantage in on-time restoration, {scenario} (pp, 95% CI)")
        st.plotly_chart(style(fig, 280))
        st.caption("Paired runs: all strategies face the same failures. Intervals reflect simulation "
                   "noise only, not real-world uncertainty.")
    except Exception:
        st.info("Paired-difference file not found; skipping that chart.")
    table = summary[summary["scenario"] == scenario].rename(columns={
        "strategy": "Strategy", "restored_mean_pct": "On-time %", "restored_std_pct": "Std dev (pp)",
        "mean_stock_outs": "Stock-outs", "mean_hours": "Mean time (h)", "p95_hours": "p95 time (h)"})
    st.dataframe(table.drop(columns=["scenario"]), hide_index=True)
    if scenario == "peak":
        st.caption("In the peak scenario p95 equals the assumed 12 h central-store delay for every "
                   "strategy: a property of that assumption, not a finding.")

# ------------------------------------------------------------- allocation
with tab_alloc:
    st.subheader("How the 100 spares are split across depots")
    try:
        alloc = load_allocations()
        fig = go.Figure()
        for s in STRATEGIES:
            fig.add_bar(name=s, x=alloc.index, y=alloc[s], marker_color=COLORS[s])
        fig.update_layout(barmode="group", title="Spares per depot")
        st.plotly_chart(style(fig, 420))
        shown = alloc.copy()
        shown.loc["Total"] = shown.sum()
        st.dataframe(shown.astype(int))
        st.caption("Risk-based gives more spares to depots whose transformers carry the most "
                   "predicted failure risk.")
    except Exception as exc:
        st.warning(f"Could not compute allocations: {exc}")

# ---------------------------------------------------------------- risk map
with tab_map:
    try:
        fleet, depots = load_fleet(), depot_table()
        risk_col = "predicted_risk" if "predicted_risk" in fleet else "true_failure_prob"
        n_top = st.slider("Highlight the highest-risk transformers", 50, 1000, 250, 50)
        top = fleet.sort_values(risk_col, ascending=False).head(n_top).copy()
        top["label"] = [f"{t}: risk {r:.2f}, {a} yrs, {l:.0f}% load" for t, r, a, l in
                        zip(top["transformer_id"], top[risk_col], top["age_years"], top["load_pct"])]
        heat = pd.DataFrame({"lat": fleet["lat"], "lon": fleet["lon"], "w": fleet[risk_col]})
        dep = depots.copy()
        dep["label"] = dep["name"]
        layers = [
            pdk.Layer("HeatmapLayer", data=heat, get_position="[lon, lat]", get_weight="w",
                      radius_pixels=50, opacity=0.6,
                      color_range=[[255, 255, 178], [254, 204, 92], [253, 141, 60],
                                   [240, 59, 32], [189, 0, 38]]),
            pdk.Layer("ScatterplotLayer", data=top, get_position="[lon, lat]", get_radius=1500,
                      radius_min_pixels=4, get_fill_color=[127, 29, 29, 235], pickable=True),
            pdk.Layer("ScatterplotLayer", data=dep, get_position="[lon, lat]", get_radius=9000,
                      radius_min_pixels=9, get_fill_color=[30, 58, 138, 200],
                      get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=2,
                      pickable=True),
            pdk.Layer("TextLayer", data=dep, get_position="[lon, lat]", get_text="name",
                      get_size=15, get_color=[15, 23, 42, 255], get_pixel_offset=[0, -24]),
        ]
        st.pydeck_chart(make_deck(layers))
        st.caption("Heat = predicted failure risk across the synthetic fleet. Dark red dots = "
                   "highest-risk transformers. Blue = depots.")
        st.subheader("Top 20 highest-risk transformers")
        cols = [c for c in ["transformer_id", "nearest_depot", "area_type", "age_years", "load_pct",
                            "oil_temp_c", risk_col] if c in fleet]
        top20 = fleet.sort_values(risk_col, ascending=False).head(20)[cols]
        st.dataframe(top20, hide_index=True)
        st.download_button("Download top 20 as CSV", top20.to_csv(index=False),
                           "top20_risky_transformers.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Risk map unavailable: {exc}")

# ---------------------------------------------------------------- what-if
with tab_lab:
    st.subheader("What-if lab: rerun the simulation with different assumptions")
    st.caption("Runs the real Monte Carlo code live. Results are not saved. Use fewer runs for "
               "speed. Failure counts scale with the 100-spare stock (normal = 70% of spares).")
    c1, c2, c3, c4 = st.columns(4)
    mult = c1.slider("Peak multiplier", 1.0, 3.0, 2.0, 0.25)
    borrow = c2.slider("Max borrowing distance (km)", 30, 150, 80, 10)
    delay = c3.slider("Central-store delay (h)", 6, 36, 12, 6)
    runs = c4.slider("Monte Carlo runs", 50, 300, 100, 50)

    @st.cache_data(show_spinner=False)
    def run_live(mult, borrow, delay, runs):
        import src.simulate as sim
        from src.allocate import load_data
        sim.MAX_BORROW_KM = borrow
        sim.CENTRAL_STORE_DELAY_HOURS = delay
        sim.PROJECT_ROOT = Path(tempfile.gettempdir())  # never touch committed result files
        with contextlib.redirect_stdout(io.StringIO()):
            raw, _, paired = sim.run_simulation(df=load_data(), num_runs=runs, peak_multiplier=mult)
        out = raw.groupby(["scenario", "strategy"]).agg(
            on_time_pct=("restored_within_deadline_pct", "mean"), stock_outs=("stock_outs", "mean"),
            mean_hours=("mean_restoration_hours", "mean"),
            p95_hours=("p95_restoration_hours", "mean")).round(3).reset_index()
        return out, paired

    if st.button("Run simulation", type="primary"):
        try:
            with st.spinner("Simulating..."):
                live, live_paired = run_live(mult, borrow, delay, runs)
            lv = live[live["scenario"] == scenario]
            fig = go.Figure()
            for s in STRATEGIES:
                d = lv[lv["strategy"] == s]
                fig.add_bar(name=s, x=["Stock-outs"], y=d["stock_outs"], marker_color=COLORS[s])
            fig.update_layout(barmode="group", title=f"Live result ({scenario}): stock-outs per run")
            st.plotly_chart(style(fig, 320))
            st.dataframe(lv.drop(columns=["scenario"]), hide_index=True)
            st.dataframe(live_paired[live_paired["scenario"] == scenario], hide_index=True)
        except Exception as exc:
            st.error(f"Live simulation failed: {exc}")

# ------------------------------------------------------------------ method
with tab_method:
    st.subheader("How it works")
    st.markdown(
        """
1. **Synthetic fleet:** 2,000 transformers around 8 assumed depot towns, with age, load and oil temperature.
2. **Risk model:** Logistic Regression with 5-fold cross-validation gives each transformer a failure risk (AUC 0.7089).
3. **Allocation:** 100 spares split three ways: equal, proportional to transformer count, or proportional to total predicted risk.
4. **Simulation:** 500 random failure scenarios per strategy. Failures are drawn from the hidden *true* risk, while the plan uses *predicted* risk. All strategies face the same failures in each run.
5. **Restoration:** nearest depot within 80 km with stock; otherwise a stock-out with a 12 h central-store delay. On time means within 8 h (urban) or 24 h (rural).
        """)
    st.markdown('<div class="warn"><b>Limits:</b> synthetic data; depot locations and regional fleet '
                'differences are assumptions; the 12 h central-store delay and 80 km borrowing limit '
                'are assumptions; no restocking during a run; road distance is straight-line x 1.3. '
                'Results show the method under stated assumptions, not real-world impact.</div>',
                unsafe_allow_html=True)
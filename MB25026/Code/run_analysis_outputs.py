from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dataset"
OUT_DIR = ROOT / "Analysis_Outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA_FILES = {
    "era": "Energy Requirment and availbilty.csv",
    "renewable": "Daily Renewable Generation.csv",
    "capacity": "Installed Capacity Statewise.csv",
    "coal": "Daily Coal Stocks.csv",
    "generation": "Daily power generation.csv",
    "outage": "Daily Power Outage.csv",
}


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def read_csv(key, parse_dates=None, usecols=None):
    path = DATA_DIR / DATA_FILES[key]
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    df = clean_columns(df)
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def font(size=18, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE = font(28, True)
F_SUB = font(16)
F_AXIS = font(14)
F_SMALL = font(12)


def canvas(title, subtitle=None, w=1200, h=720):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, h], fill="#ffffff")
    d.text((55, 28), title, fill="#18202b", font=F_TITLE)
    if subtitle:
        d.text((55, 66), subtitle, fill="#4b5563", font=F_SUB)
    return img, d


def nice_ticks(vmin, vmax, n=5):
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    raw = (vmax - vmin) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(abs(raw)))
    step = round(raw / mag) * mag
    start = math.floor(vmin / step) * step
    return [start + i * step for i in range(n + 1)]


def save_bar(labels, values, title, ylabel, filename, color="#2563eb"):
    values = pd.Series(values, dtype=float).fillna(0).to_numpy()
    labels = [str(x) for x in labels]
    img, d = canvas(title, ylabel)
    left, top, right, bottom = 115, 115, 1130, 610
    d.line([left, bottom, right, bottom], fill="#111827", width=2)
    d.line([left, top, left, bottom], fill="#111827", width=2)
    ymax = max(values.max() * 1.12, 1)
    for tick in nice_ticks(0, ymax, 5):
        y = bottom - (tick / ymax) * (bottom - top)
        d.line([left, y, right, y], fill="#e5e7eb", width=1)
        d.text((20, y - 8), f"{tick:,.0f}", fill="#374151", font=F_SMALL)
    gap = 10
    bw = max(8, (right - left - gap * (len(values) + 1)) / max(len(values), 1))
    for i, (lab, val) in enumerate(zip(labels, values)):
        x0 = left + gap + i * (bw + gap)
        x1 = x0 + bw
        y0 = bottom - (val / ymax) * (bottom - top)
        d.rectangle([x0, y0, x1, bottom], fill=color)
        d.text((x0, y0 - 20), f"{val:,.1f}", fill="#111827", font=F_SMALL)
        wrapped = "\n".join(textwrap.wrap(lab.replace("_", " ").title(), width=11)[:2])
        d.multiline_text((x0 - 6, bottom + 10), wrapped, fill="#374151", font=F_SMALL, align="center")
    img.save(OUT_DIR / filename)


def save_grouped_line(x, series_map, title, ylabel, filename):
    img, d = canvas(title, ylabel)
    left, top, right, bottom = 95, 120, 1130, 610
    d.line([left, bottom, right, bottom], fill="#111827", width=2)
    d.line([left, top, left, bottom], fill="#111827", width=2)
    vals = np.concatenate([pd.Series(v, dtype=float).dropna().to_numpy() for v in series_map.values()])
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    ymin = min(0, ymin)
    ymax = ymax * 1.10 if ymax > 0 else 1
    for tick in nice_ticks(ymin, ymax, 5):
        y = bottom - ((tick - ymin) / (ymax - ymin)) * (bottom - top)
        d.line([left, y, right, y], fill="#e5e7eb", width=1)
        d.text((18, y - 8), f"{tick:,.0f}", fill="#374151", font=F_SMALL)
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    n = len(x)
    for idx, (name, yvals) in enumerate(series_map.items()):
        yvals = pd.Series(yvals, dtype=float).to_numpy()
        pts = []
        for i, val in enumerate(yvals):
            if np.isfinite(val):
                px = left + (i / max(n - 1, 1)) * (right - left)
                py = bottom - ((val - ymin) / (ymax - ymin)) * (bottom - top)
                pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=colors[idx % len(colors)], width=3)
        for px, py in pts[:: max(1, len(pts) // 18)]:
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=colors[idx % len(colors)])
        lx = left + 10 + idx * 210
        d.rectangle([lx, 86, lx + 18, 100], fill=colors[idx % len(colors)])
        d.text((lx + 26, 82), name, fill="#111827", font=F_AXIS)
    if len(x) > 1:
        for i in np.linspace(0, len(x) - 1, 6, dtype=int):
            px = left + (i / max(n - 1, 1)) * (right - left)
            lab = str(x.iloc[i].date() if hasattr(x.iloc[i], "date") else x.iloc[i])
            d.text((px - 35, bottom + 12), lab[:10], fill="#374151", font=F_SMALL)
    img.save(OUT_DIR / filename)


def save_heatmap(df, title, filename):
    img, d = canvas(title, "NPV values in Rs crore; green is economically stronger", w=1220, h=760)
    left, top = 150, 130
    cell_w, cell_h = 120, 58
    values = df.to_numpy(dtype=float)
    vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1)
    for j, col in enumerate(df.columns):
        d.text((left + j * cell_w + 35, top - 30), str(col), fill="#111827", font=F_AXIS)
    for i, idx in enumerate(df.index):
        d.text((35, top + i * cell_h + 18), str(idx), fill="#111827", font=F_AXIS)
        for j, col in enumerate(df.columns):
            val = float(df.iloc[i, j])
            if val >= 0:
                intensity = int(235 - min(170, 170 * val / vmax))
                fill = f"#{intensity:02x}f0{intensity:02x}"
            else:
                intensity = int(235 - min(170, 170 * abs(val) / vmax))
                fill = f"#f0{intensity:02x}{intensity:02x}"
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            d.rectangle([x0, y0, x0 + cell_w - 3, y0 + cell_h - 3], fill=fill, outline="#ffffff")
            d.text((x0 + 24, y0 + 18), f"{val:,.0f}", fill="#111827", font=F_SMALL)
    d.text((left + 330, 690), "Capex: Rs crore per MWh", fill="#374151", font=F_AXIS)
    d.text((20, 370), "Discount rate", fill="#374151", font=F_AXIS)
    img.save(OUT_DIR / filename)


def main():
    print("Loading datasets...")
    df_era = read_csv("era", parse_dates=["month"])
    df_ren = read_csv("renewable", parse_dates=["date"])
    df_cap = read_csv("capacity", parse_dates=["date"])
    df_coal = read_csv(
        "coal",
        parse_dates=["date"],
        usecols=[
            "date", "state_name", "power_station_name", "capacity", "daily_requirement",
            "daily_receipt", "daily_consumption", "normative_stock_days",
            "total_stock", "stock_days", "plf_prcnt", "is_critical",
        ],
    )
    df_out = read_csv(
        "outage",
        parse_dates=["date"],
        usecols=["date", "region", "state_name", "sector", "station_type", "outage_type", "monitored_capacity", "cap_under_outage"],
    )

    df_era = numeric(df_era, ["energy_requirement", "energy_availability"])
    df_ren = numeric(df_ren, ["wind_energy", "solar_energy", "other_renewable_energy", "total_renewable_energy"])
    df_cap = numeric(df_cap, ["coal_cap", "gas_cap", "diesel_cap", "lignite_cap", "nuclear_cap", "hydro_cap", "res_cap"])
    df_coal = numeric(df_coal, ["capacity", "daily_requirement", "daily_receipt", "daily_consumption", "normative_stock_days", "total_stock", "stock_days", "plf_prcnt"])
    df_out = numeric(df_out, ["monitored_capacity", "cap_under_outage"])

    print("Running analysis...")
    summary = pd.DataFrame({
        "dataset": ["Energy requirement", "Renewable generation", "Installed capacity", "Coal stocks", "Power outage"],
        "rows": [len(df_era), len(df_ren), len(df_cap), len(df_coal), len(df_out)],
        "start": [df_era["month"].min(), df_ren["date"].min(), df_cap["date"].min(), df_coal["date"].min(), df_out["date"].min()],
        "end": [df_era["month"].max(), df_ren["date"].max(), df_cap["date"].max(), df_coal["date"].max(), df_out["date"].max()],
    })

    ren_all = df_ren[df_ren["state_name"].str.lower().eq("all india")].copy()
    ren_monthly = (
        ren_all.assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)[["solar_energy", "wind_energy", "other_renewable_energy", "total_renewable_energy"]]
        .sum()
    )

    latest_date = df_cap["date"].max()
    cap_latest = df_cap[df_cap["date"].eq(latest_date)].copy()
    cap_cols = ["coal_cap", "gas_cap", "diesel_cap", "lignite_cap", "nuclear_cap", "hydro_cap", "res_cap"]
    capacity_mix_mw = cap_latest[cap_cols].sum()
    capacity_mix_gw = capacity_mix_mw / 1000

    state_capacity = (
        cap_latest.groupby("state_name", as_index=False)[cap_cols]
        .sum()
        .assign(total_cap=lambda d: d[cap_cols].sum(axis=1))
    )
    state_capacity["renewable_share"] = state_capacity["res_cap"] / state_capacity["total_cap"].replace(0, np.nan)
    top_states = state_capacity.sort_values("res_cap", ascending=False).head(10).copy()

    coal_valid = df_coal.dropna(subset=["stock_days"]).copy()
    coal_valid["critical_flag"] = coal_valid["is_critical"].astype(str).str.lower().isin(["yes", "true", "1", "critical"])
    coal_monthly = (
        coal_valid.assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(avg_stock_days=("stock_days", "mean"), critical_units=("critical_flag", "sum"), stations=("power_station_name", "nunique"))
    )

    era = df_era.copy()
    era["deficit"] = era["energy_requirement"] - era["energy_availability"]
    era_monthly = era.groupby("month", as_index=False)[["energy_requirement", "energy_availability", "deficit"]].sum()
    era_monthly["deficit_pct"] = era_monthly["deficit"] / era_monthly["energy_requirement"].replace(0, np.nan) * 100

    peak_hist = pd.DataFrame({
        "year": list(range(2010, 2026)),
        "peak_demand_mw": [119166, 130006, 135453, 135918, 141160, 148166, 153366, 160752, 170164, 183804, 190198, 203014, 215888, 243271, 250000, 270000],
    })
    x = peak_hist["year"].to_numpy()
    y = peak_hist["peak_demand_mw"].to_numpy()
    coef = np.polyfit(x - x.min(), y, deg=2)
    future_years = np.arange(2026, 2031)
    forecast = pd.DataFrame({"year": future_years, "forecast_peak_demand_mw": np.polyval(coef, future_years - x.min()).round(0).astype(int)})

    def eoq(row, S=0.08, H=0.002):
        D = max(float(row["daily_requirement"]), 0.001)
        q = np.sqrt((2 * D * S) / H)
        return pd.Series({"eoq_thousand_tonne": q, "cycle_days": q / D})

    coal_station = (
        coal_valid.groupby("power_station_name", as_index=False)
        .agg(daily_requirement=("daily_requirement", "mean"), avg_stock_days=("stock_days", "mean"), capacity_mw=("capacity", "mean"))
        .dropna(subset=["daily_requirement"])
    )
    coal_eoq = pd.concat([coal_station, coal_station.apply(eoq, axis=1)], axis=1)
    coal_eoq["reorder_needed"] = coal_eoq["avg_stock_days"] < 14
    coal_eoq_top = coal_eoq.sort_values(["reorder_needed", "avg_stock_days"], ascending=[False, True]).head(15)

    sources = ["Solar", "Wind", "Hydro", "Coal", "Gas"]
    capacity_gw = {
        "Solar": max(capacity_mix_mw.get("res_cap", 0) * 0.55 / 1000, 1),
        "Wind": max(capacity_mix_mw.get("res_cap", 0) * 0.35 / 1000, 1),
        "Hydro": max(capacity_mix_mw.get("hydro_cap", 0) / 1000, 1),
        "Coal": max((capacity_mix_mw.get("coal_cap", 0) + capacity_mix_mw.get("lignite_cap", 0)) / 1000, 1),
        "Gas": max(capacity_mix_mw.get("gas_cap", 0) / 1000, 1),
    }
    cost = {"Solar": 2.2, "Wind": 2.8, "Hydro": 3.5, "Coal": 4.6, "Gas": 7.0}
    emission = {"Solar": 0.0, "Wind": 0.0, "Hydro": 0.02, "Coal": 0.95, "Gas": 0.45}
    demand_gw_2030 = float(forecast.loc[forecast["year"].eq(2030), "forecast_peak_demand_mw"].iloc[0]) / 1000
    emission_cap = demand_gw_2030 * 1000 * 0.55
    res = linprog(
        np.array([cost[s] for s in sources]),
        A_ub=[np.array([emission[s] * 1000 for s in sources])],
        b_ub=[emission_cap],
        A_eq=[np.ones(len(sources))],
        b_eq=[demand_gw_2030],
        bounds=[(0, capacity_gw[s]) for s in sources],
        method="highs",
    )
    dispatch = pd.DataFrame({"source": sources, "generation_gw": res.x if res.success else np.zeros(len(sources))})
    dispatch["share_pct"] = dispatch["generation_gw"] / dispatch["generation_gw"].sum() * 100
    dispatch["cost_rs_kwh"] = dispatch["source"].map(cost)
    dispatch["emission_tco2_mwh"] = dispatch["source"].map(emission)

    readiness = state_capacity[["state_name", "res_cap", "hydro_cap", "total_cap", "renewable_share"]].copy()
    readiness["res_gw"] = readiness["res_cap"] / 1000
    readiness["hydro_gw"] = readiness["hydro_cap"] / 1000
    readiness["readiness_score"] = (
        readiness["renewable_share"].fillna(0).rank(pct=True) * 0.45
        + readiness["res_gw"].rank(pct=True) * 0.40
        + readiness["hydro_gw"].rank(pct=True) * 0.15
    ) * 100
    readiness_top = readiness.sort_values("readiness_score", ascending=False).head(10)

    bess_gwh = 300
    annual_benefit_cr = 42000
    life_years = 15
    rates = np.arange(0.06, 0.141, 0.01)
    costs = np.arange(2.5, 5.6, 0.5)

    def bess_npv(rate, capex_cr_mwh):
        capex = bess_gwh * 1000 * capex_cr_mwh
        pv = sum(annual_benefit_cr / ((1 + rate) ** t) for t in range(1, life_years + 1))
        return pv - capex

    bess_sensitivity = pd.DataFrame(
        [[bess_npv(r, c) for c in costs] for r in rates],
        index=[f"{r:.0%}" for r in rates],
        columns=[f"{c:.1f}" for c in costs],
    )

    carbon_tax_rs_kwh = 51 * 83 * 0.95 / 1000
    roadmap = pd.DataFrame({
        "source": ["Solar", "Wind", "Hydro/Flexible", "BESS"],
        "current_gw_est": [capacity_gw["Solar"], capacity_gw["Wind"], capacity_gw["Hydro"], 0],
        "2030_target_gw": [280, 140, 65, 75],
    })
    roadmap["new_capacity_needed_gw"] = (roadmap["2030_target_gw"] - roadmap["current_gw_est"]).clip(lower=0)
    roadmap["annual_addition_needed_gw"] = roadmap["new_capacity_needed_gw"] / 5

    print("Writing charts...")
    save_bar(capacity_mix_gw.index, capacity_mix_gw.values, "Latest Installed Capacity Mix", "GW by source", "01_capacity_mix.png", "#2563eb")
    save_grouped_line(
        ren_monthly["month"],
        {
            "Solar": ren_monthly["solar_energy"],
            "Wind": ren_monthly["wind_energy"],
            "Other RE": ren_monthly["other_renewable_energy"],
        },
        "All-India Renewable Generation Trend",
        "Monthly generation (MU)",
        "02_renewable_generation_trend.png",
    )
    save_grouped_line(
        coal_monthly["month"],
        {"Avg stock days": coal_monthly["avg_stock_days"], "Critical units": coal_monthly["critical_units"]},
        "Coal Inventory Stress",
        "Stock days and critical unit count",
        "03_coal_inventory_stress.png",
    )
    save_grouped_line(
        era_monthly["month"],
        {"Requirement": era_monthly["energy_requirement"], "Availability": era_monthly["energy_availability"]},
        "Energy Requirement vs Availability",
        "Monthly energy (MU)",
        "04_requirement_vs_availability.png",
    )
    save_grouped_line(
        pd.Series(list(peak_hist["year"]) + list(forecast["year"])),
        {
            "Historical": pd.Series(list(peak_hist["peak_demand_mw"]) + [np.nan] * len(forecast)),
            "Forecast": pd.Series([np.nan] * len(peak_hist) + list(forecast["forecast_peak_demand_mw"])),
        },
        "Peak Demand Forecast to 2030",
        "MW",
        "05_peak_demand_forecast.png",
    )
    save_bar(dispatch["source"], dispatch["generation_gw"], "Minimum-Cost Dispatch Mix for 2030 Peak", "Generation GW", "06_dispatch_mix.png", "#059669")
    save_bar(readiness_top["state_name"], readiness_top["readiness_score"], "Top State Renewable Readiness Scores", "Score out of 100", "07_state_readiness.png", "#7c3aed")
    save_heatmap(bess_sensitivity, "BESS Investment Sensitivity", "08_bess_npv_sensitivity.png")

    print("Writing result tables...")
    tables = {
        "dataset_summary.csv": summary,
        "renewable_monthly.csv": ren_monthly,
        "capacity_mix_gw.csv": capacity_mix_gw.rename("capacity_gw").reset_index().rename(columns={"index": "source"}),
        "top_renewable_states.csv": top_states,
        "coal_monthly.csv": coal_monthly,
        "energy_gap_monthly.csv": era_monthly,
        "peak_demand_forecast.csv": forecast,
        "coal_eoq_top15.csv": coal_eoq_top,
        "dispatch_mix.csv": dispatch,
        "state_readiness_top10.csv": readiness_top,
        "bess_sensitivity.csv": bess_sensitivity,
        "capacity_roadmap.csv": roadmap,
    }
    for name, df in tables.items():
        df.to_csv(OUT_DIR / name, index=True if name == "bess_sensitivity.csv" else False)

    latest_ren = ren_monthly.iloc[-1]
    latest_gap = era_monthly.iloc[-1]
    latest_coal = coal_monthly.iloc[-1]
    recs = pd.DataFrame({
        "priority": [1, 2, 3, 4, 5],
        "recommendation": [
            "Build storage and flexible capacity with renewable expansion.",
            "Use EOQ/reorder dashboards for critical coal stations during transition.",
            "Prioritise high-readiness states for renewable integration pilots.",
            "Introduce gradual carbon pricing or equivalent externality correction.",
            "Accelerate transmission corridors linked to renewable-heavy states.",
        ],
        "evidence": [
            "2030 peak demand forecast and dispatch optimisation require firm capacity.",
            "Coal stock stress shows reliability risk when stock days fall below norms.",
            "Readiness score highlights states with capacity and flexibility advantages.",
            f"Carbon externality estimate is about Rs {carbon_tax_rs_kwh:.2f}/kWh for coal.",
            "Renewable growth needs evacuation capacity to affect peak reliability.",
        ],
    })
    recs.to_csv(OUT_DIR / "final_recommendations.csv", index=False)

    summary_md = f"""# Analysis Results: Can Renewable Energy Solve India's Peak Demand Problem?

## Key Results

- Datasets processed: {len(summary)} official power-sector datasets from `{DATA_DIR}`.
- Latest installed capacity snapshot: {latest_date.date()}.
- Renewable capacity in latest snapshot: {capacity_mix_gw.get('res_cap', 0):,.1f} GW.
- Coal + lignite capacity in latest snapshot: {(capacity_mix_gw.get('coal_cap', 0) + capacity_mix_gw.get('lignite_cap', 0)):,.1f} GW.
- Latest all-India renewable generation month: {latest_ren['month'].date()}, total renewable generation {latest_ren['total_renewable_energy']:,.0f} MU.
- Latest energy deficit month: {latest_gap['month'].date()}, deficit {latest_gap['deficit']:,.0f} MU ({latest_gap['deficit_pct']:.2f}%).
- Latest coal-stock month: {latest_coal['month'].date()}, average coal stock {latest_coal['avg_stock_days']:.1f} days.
- Forecast 2030 peak demand: {forecast.loc[forecast['year'].eq(2030), 'forecast_peak_demand_mw'].iloc[0]:,} MW.
- Dispatch model total generation for 2030 peak: {dispatch['generation_gw'].sum():,.1f} GW.
- Implied Pigouvian carbon tax estimate for coal: Rs {carbon_tax_rs_kwh:.2f}/kWh.

## Main Conclusion

Renewable energy can solve a large part of India's energy requirement problem, but peak demand reliability needs a portfolio: renewables plus BESS/storage, flexible generation, transmission expansion, and disciplined coal inventory management during the transition.

## Output Charts

1. `01_capacity_mix.png`
2. `02_renewable_generation_trend.png`
3. `03_coal_inventory_stress.png`
4. `04_requirement_vs_availability.png`
5. `05_peak_demand_forecast.png`
6. `06_dispatch_mix.png`
7. `07_state_readiness.png`
8. `08_bess_npv_sensitivity.png`
"""
    (OUT_DIR / "results_summary.md").write_text(summary_md, encoding="utf-8")

    print("Done:", OUT_DIR)
    print(summary_md)


if __name__ == "__main__":
    main()

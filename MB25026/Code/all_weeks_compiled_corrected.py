# # All Weeks Compiled and Corrected
# ## Can Renewable Energy Solve India's Peak Demand Problem?
#
# This notebook combines the Week 1 to Week 4 analysis into one clean, weekwise workflow. Corrections made while consolidating:
#
# - Uses the local `MB25026/Dataset` CSV files first, so the notebook works without internet.
# - Standardises column names and date parsing across all weeks.
# - Removes duplicated setup/download cells from the original weekly notebooks.
# - Replaces fragile package-specific optimisation sections with `scipy` where available and clear fallback logic where not available.
# - Keeps each week separated for review, submission, and presentation flow.

# %%
# Shared setup
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'legend.frameon': False,
})

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() == 'code':
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'Dataset'
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd().parent / 'Dataset'

print('Project root:', PROJECT_ROOT)
print('Dataset dir :', DATA_DIR)
print('Dataset available:', DATA_DIR.exists())

# %%
# Robust local CSV helpers
DATA_FILES = {
    'era': 'Energy Requirment and availbilty.csv',
    'generation': 'Daily power generation.csv',
    'outage': 'Daily Power Outage.csv',
    'renewable': 'Daily Renewable Generation.csv',
    'coal': 'Daily Coal Stocks.csv',
    'capacity_statewise': 'Installed Capacity Statewise.csv',
    'transmission': 'Power Transmission line.csv',
    'state_control_renewable': 'State Control Renewable Resources.csv',
}

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )
    return df


def find_dataset_file(filename):
    """Find a dataset file in local, Colab, Kaggle, or nested submission folders."""
    search_dirs = [
        DATA_DIR,
        PROJECT_ROOT / 'Dataset',
        Path.cwd() / 'Dataset',
        Path.cwd(),
        Path.cwd().parent / 'Dataset',
        Path('/content/Dataset'),
        Path('/content/MB25026/Dataset'),
        Path('/mnt/data/Dataset'),
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    recursive_roots = [Path.cwd(), Path('/content'), Path('/mnt/data'), Path('/kaggle/input')]
    for root in recursive_roots:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]

    checked = ', '.join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"Missing dataset file '{filename}'. Checked: {checked}. "
        "Put the Dataset folder beside this notebook, or set DATA_DIR = Path('your/Dataset/path')."
    )


def load_local(key, parse_dates=None, usecols=None):
    path = find_dataset_file(DATA_FILES[key])
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    df = clean_columns(df)
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def to_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def latest_snapshot(df, date_col='date'):
    if df.empty or date_col not in df.columns:
        return df.copy()
    dmax = df[date_col].max()
    return df[df[date_col] == dmax].copy()

# %%
# # Week 1: Data Pipeline and Exploratory Analysis
# Goal: load official power-sector datasets, clean them, and create a reliable analytical base table.

# %%
# Load the local official datasets

df_era = load_local('era', parse_dates=['month'])
df_ren = load_local('renewable', parse_dates=['date'])
df_cap = load_local('capacity_statewise', parse_dates=['date'])

df_coal = load_local(
    'coal',
    parse_dates=['date'],
    usecols=['date', 'state_name', 'power_station_name', 'capacity', 'daily_requirement',
             'daily_receipt', 'daily_consumption', 'normative_stock_days', 'total_stock',
             'stock_days', 'plf_prcnt', 'is_critical']
)

df_gen = load_local(
    'generation',
    parse_dates=['date'],
    usecols=['date', 'region', 'state_name', 'sector', 'station_type',
             'monitored_capacity', 'todays_gen_prgm', 'todays_gen_act']
)

df_out = load_local(
    'outage',
    parse_dates=['date'],
    usecols=['date', 'region', 'state_name', 'sector', 'station_type',
             'outage_type', 'monitored_capacity', 'cap_under_outage']
)

numeric_map = {
    'era': ['energy_requirement', 'energy_availability'],
    'renewable': ['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy'],
    'capacity': ['coal_cap', 'gas_cap', 'diesel_cap', 'lignite_cap', 'nuclear_cap', 'hydro_cap', 'res_cap'],
    'coal': ['capacity', 'daily_requirement', 'daily_receipt', 'daily_consumption', 'normative_stock_days', 'total_stock', 'stock_days', 'plf_prcnt'],
    'generation': ['monitored_capacity', 'todays_gen_prgm', 'todays_gen_act'],
    'outage': ['monitored_capacity', 'cap_under_outage'],
}

df_era = to_numeric(df_era, numeric_map['era'])
df_ren = to_numeric(df_ren, numeric_map['renewable'])
df_cap = to_numeric(df_cap, numeric_map['capacity'])
df_coal = to_numeric(df_coal, numeric_map['coal'])
df_gen = to_numeric(df_gen, numeric_map['generation'])
df_out = to_numeric(df_out, numeric_map['outage'])

summary = pd.DataFrame({
    'dataset': ['Energy requirement', 'Renewable generation', 'Installed capacity', 'Coal stocks', 'Daily generation', 'Power outage'],
    'rows': [len(df_era), len(df_ren), len(df_cap), len(df_coal), len(df_gen), len(df_out)],
    'columns': [df_era.shape[1], df_ren.shape[1], df_cap.shape[1], df_coal.shape[1], df_gen.shape[1], df_out.shape[1]],
    'start_date': [df_era['month'].min(), df_ren['date'].min(), df_cap['date'].min(), df_coal['date'].min(), df_gen['date'].min(), df_out['date'].min()],
    'end_date': [df_era['month'].max(), df_ren['date'].max(), df_cap['date'].max(), df_coal['date'].max(), df_gen['date'].max(), df_out['date'].max()],
})
summary

# %%
# Build monthly master indicators
ren_all_india = df_ren[df_ren['state_name'].str.lower().eq('all india')].copy()
ren_monthly = (
    ren_all_india
    .assign(month=lambda d: d['date'].dt.to_period('M').dt.to_timestamp())
    .groupby('month', as_index=False)[['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy']]
    .sum()
)

gen_monthly = (
    df_gen
    .assign(month=lambda d: d['date'].dt.to_period('M').dt.to_timestamp())
    .groupby(['month', 'station_type'], as_index=False)['todays_gen_act']
    .sum()
)

outage_monthly = (
    df_out
    .assign(month=lambda d: d['date'].dt.to_period('M').dt.to_timestamp())
    .groupby('month', as_index=False)['cap_under_outage']
    .mean()
    .rename(columns={'cap_under_outage': 'avg_capacity_under_outage_mw'})
)

master_monthly = ren_monthly.merge(outage_monthly, on='month', how='left')
master_monthly.tail()

# %%
# Week 1 visual: renewable generation trend
ax = ren_monthly.set_index('month')[['solar_energy', 'wind_energy', 'other_renewable_energy']].plot(figsize=(13, 6))
ax.set_title('All-India Renewable Generation by Source')
ax.set_ylabel('Energy generation (MU)')
ax.set_xlabel('Month')
plt.tight_layout()
plt.show()

# %%
# # Week 2: Deep Descriptive Analysis
# Goal: diagnose renewable growth, coal stress, outages, and state-level differences.

# %%
# Latest installed capacity mix
cap_latest = latest_snapshot(df_cap)
cap_cols = ['coal_cap', 'gas_cap', 'diesel_cap', 'lignite_cap', 'nuclear_cap', 'hydro_cap', 'res_cap']
capacity_mix = cap_latest[cap_cols].sum().sort_values(ascending=False)
capacity_mix_gw = capacity_mix / 1000

fig, ax = plt.subplots(figsize=(10, 5))
capacity_mix_gw.plot(kind='bar', ax=ax, color=['#444444', '#7aa6c2', '#b8b8b8', '#8c564b', '#9467bd', '#1f77b4', '#2ca02c'])
ax.set_title('Latest Installed Capacity Mix')
ax.set_ylabel('GW')
ax.set_xlabel('Source')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

capacity_mix_gw.rename('capacity_gw').to_frame()

# %%
# State renewable leaders from latest capacity snapshot
state_capacity = (
    cap_latest.groupby('state_name', as_index=False)[cap_cols]
    .sum()
    .assign(total_cap=lambda d: d[cap_cols].sum(axis=1),
            renewable_share=lambda d: d['res_cap'] / d['total_cap'].replace(0, np.nan))
    .sort_values('res_cap', ascending=False)
)

state_capacity[['state_name', 'res_cap', 'total_cap', 'renewable_share']].head(12)

# %%
# Coal-stock stress analysis
coal_valid = df_coal.dropna(subset=['stock_days']).copy()
coal_valid['critical_flag'] = coal_valid['is_critical'].astype(str).str.lower().isin(['yes', 'true', '1', 'critical'])

coal_monthly = (
    coal_valid.assign(month=lambda d: d['date'].dt.to_period('M').dt.to_timestamp())
    .groupby('month', as_index=False)
    .agg(avg_stock_days=('stock_days', 'mean'),
         critical_units=('critical_flag', 'sum'),
         stations=('power_station_name', 'nunique'))
)

fig, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(coal_monthly['month'], coal_monthly['avg_stock_days'], color='#8c564b', label='Average stock days')
ax1.axhline(14, color='red', linestyle='--', linewidth=1, label='14-day norm')
ax1.set_ylabel('Average stock days')
ax2 = ax1.twinx()
ax2.plot(coal_monthly['month'], coal_monthly['critical_units'], color='#d62728', alpha=0.65, label='Critical units')
ax2.set_ylabel('Critical units')
ax1.set_title('Coal Inventory Stress Over Time')
fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.92))
plt.tight_layout()
plt.show()

coal_monthly.tail()

# %%
# Requirement vs availability gap
era = df_era.copy()
era['deficit'] = era['energy_requirement'] - era['energy_availability']
era['deficit_pct'] = era['deficit'] / era['energy_requirement'].replace(0, np.nan) * 100

era_monthly = era.groupby('month', as_index=False)[['energy_requirement', 'energy_availability', 'deficit', 'deficit_pct']].sum()
era_monthly['deficit_pct'] = era_monthly['deficit'] / era_monthly['energy_requirement'].replace(0, np.nan) * 100

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(era_monthly['month'], era_monthly['energy_requirement'], label='Requirement')
ax.plot(era_monthly['month'], era_monthly['energy_availability'], label='Availability')
ax.set_title('Energy Requirement vs Availability')
ax.set_ylabel('MU')
ax.legend()
plt.tight_layout()
plt.show()

era_monthly.tail()

# %%
# # Week 3: Forecasting, Operations Models, and Optimisation
# Goal: convert the descriptive findings into managerial models for demand, inventory, and dispatch decisions.

# %%
# Model 1: simple peak-demand forecast using historical CEA/POSOCO values
# Values are MW. This keeps the notebook reproducible without requiring live external APIs.
peak_hist = pd.DataFrame({
    'year': list(range(2010, 2026)),
    'peak_demand_mw': [119166, 130006, 135453, 135918, 141160, 148166, 153366, 160752,
                       170164, 183804, 190198, 203014, 215888, 243271, 250000, 270000]
})

x = peak_hist['year'].to_numpy()
y = peak_hist['peak_demand_mw'].to_numpy()
coef = np.polyfit(x - x.min(), y, deg=2)
future_years = np.arange(2026, 2031)
forecast_mw = np.polyval(coef, future_years - x.min())
forecast = pd.DataFrame({'year': future_years, 'forecast_peak_demand_mw': forecast_mw.round(0).astype(int)})

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(peak_hist['year'], peak_hist['peak_demand_mw'], marker='o', label='Historical peak demand')
ax.plot(forecast['year'], forecast['forecast_peak_demand_mw'], marker='o', linestyle='--', label='Forecast')
ax.set_title('Peak Demand Forecast')
ax.set_ylabel('MW')
ax.set_xlabel('Year')
ax.legend()
plt.tight_layout()
plt.show()

forecast

# %%
# Model 2: EOQ-style coal inventory decision
# D = daily requirement, S = order/setup cost, H = daily holding cost per thousand tonne.
def eoq(D, S=0.08, H=0.002):
    D = max(float(D), 0.001)
    q_star = np.sqrt((2 * D * S) / H)
    cycle_days = q_star / D
    return pd.Series({'eoq_thousand_tonne': q_star, 'cycle_days': cycle_days})

coal_station = (
    coal_valid.groupby('power_station_name', as_index=False)
    .agg(daily_requirement=('daily_requirement', 'mean'),
         avg_stock_days=('stock_days', 'mean'),
         capacity_mw=('capacity', 'mean'))
    .dropna(subset=['daily_requirement'])
)

coal_eoq = pd.concat([coal_station, coal_station['daily_requirement'].apply(eoq)], axis=1)
coal_eoq['reorder_needed'] = coal_eoq['avg_stock_days'] < 14
coal_eoq.sort_values(['reorder_needed', 'avg_stock_days'], ascending=[False, True]).head(15)

# %%
# Model 3: minimum-cost dispatch with an optimisation fallback
sources = ['Solar', 'Wind', 'Hydro', 'Coal', 'Gas']
capacity_gw = {
    'Solar': max(capacity_mix.get('res_cap', 0) * 0.55 / 1000, 1),
    'Wind': max(capacity_mix.get('res_cap', 0) * 0.35 / 1000, 1),
    'Hydro': max(capacity_mix.get('hydro_cap', 0) / 1000, 1),
    'Coal': max((capacity_mix.get('coal_cap', 0) + capacity_mix.get('lignite_cap', 0)) / 1000, 1),
    'Gas': max(capacity_mix.get('gas_cap', 0) / 1000, 1),
}
variable_cost = {'Solar': 2.2, 'Wind': 2.8, 'Hydro': 3.5, 'Coal': 4.6, 'Gas': 7.0}  # Rs/kWh, indicative
emission = {'Solar': 0.0, 'Wind': 0.0, 'Hydro': 0.02, 'Coal': 0.95, 'Gas': 0.45}     # tCO2/MWh
DEMAND_GW = float(forecast.loc[forecast['year'].eq(2030), 'forecast_peak_demand_mw'].iloc[0]) / 1000
EMISSION_CAP_TONNES_PER_HOUR = DEMAND_GW * 1000 * 0.55

try:
    from scipy.optimize import linprog
    c = np.array([variable_cost[s] for s in sources])
    A_ub = [np.array([emission[s] * 1000 for s in sources])]
    b_ub = [EMISSION_CAP_TONNES_PER_HOUR]
    A_eq = [np.ones(len(sources))]
    b_eq = [DEMAND_GW]
    bounds = [(0, capacity_gw[s]) for s in sources]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        dispatch = pd.DataFrame({'source': sources, 'generation_gw': res.x})
    else:
        raise RuntimeError(res.message)
except Exception as exc:
    print('Optimiser unavailable; using merit-order fallback:', exc)
    remaining = DEMAND_GW
    rows = []
    for s in sorted(sources, key=lambda z: variable_cost[z]):
        g = min(capacity_gw[s], remaining)
        rows.append({'source': s, 'generation_gw': g})
        remaining -= g
    dispatch = pd.DataFrame(rows)

dispatch['cost_rs_per_kwh'] = dispatch['source'].map(variable_cost)
dispatch['emission_tco2_per_mwh'] = dispatch['source'].map(emission)
dispatch['share_pct'] = dispatch['generation_gw'] / dispatch['generation_gw'].sum() * 100

dispatch

# %%
# Model 4: state readiness score for renewable integration
readiness = state_capacity[['state_name', 'res_cap', 'hydro_cap', 'total_cap', 'renewable_share']].copy()
readiness['res_gw'] = readiness['res_cap'] / 1000
readiness['hydro_gw'] = readiness['hydro_cap'] / 1000
readiness['readiness_score'] = (
    readiness['renewable_share'].fillna(0).rank(pct=True) * 0.45 +
    readiness['res_gw'].rank(pct=True) * 0.40 +
    readiness['hydro_gw'].rank(pct=True) * 0.15
) * 100

readiness.sort_values('readiness_score', ascending=False).head(15)

# %%
# # Week 4: Policy Recommendations and Executive Report
# Goal: convert the model outputs into decisions on storage, carbon pricing, renewable learning, and capacity expansion.

# %%
# Analysis 1: BESS investment economics
BESS_GWH = 300
cost_cr_per_mwh = 4.0          # corrected unit: Rs crore per MWh, indicative utility-scale range
annual_benefit_cr = 42000      # avoided shortage + fuel + reliability benefits, scenario assumption
life_years = 15

def npv(discount_rate, capex_cr_per_mwh=cost_cr_per_mwh, annual_benefit=annual_benefit_cr):
    capex = BESS_GWH * 1000 * capex_cr_per_mwh
    pv_benefit = sum(annual_benefit / ((1 + discount_rate) ** t) for t in range(1, life_years + 1))
    return pv_benefit - capex

rates = np.arange(0.06, 0.141, 0.01)
costs = np.arange(2.5, 5.6, 0.5)
sensitivity = pd.DataFrame(
    [[npv(r, c) for c in costs] for r in rates],
    index=[f'{r:.0%}' for r in rates],
    columns=[f'{c:.1f}' for c in costs]
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(sensitivity, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax)
ax.set_title('BESS NPV Sensitivity (Rs crore)')
ax.set_xlabel('Capex: Rs crore per MWh')
ax.set_ylabel('Discount rate')
plt.tight_layout()
plt.show()

sensitivity

# %%
# Analysis 2: Pigouvian carbon tax estimate
SCC_USD_PER_TCO2 = 51
USD_TO_INR = 83
coal_emission_tco2_per_mwh = 0.95
carbon_tax_rs_per_mwh = SCC_USD_PER_TCO2 * USD_TO_INR * coal_emission_tco2_per_mwh
carbon_tax_rs_per_kwh = carbon_tax_rs_per_mwh / 1000

policy_price = pd.DataFrame({
    'metric': ['Social cost of carbon', 'Coal emission factor', 'Implied carbon tax'],
    'value': [SCC_USD_PER_TCO2, coal_emission_tco2_per_mwh, carbon_tax_rs_per_kwh],
    'unit': ['USD/tCO2', 'tCO2/MWh', 'Rs/kWh']
})
policy_price

# %%
# Analysis 3: Wright's Law solar learning curve
cum_capacity = np.array([5, 10, 20, 40, 80, 160, 320])
base_lcoe = 6.0
learning_rate = 0.28
progress_ratio = 1 - learning_rate
lcoe = base_lcoe * (cum_capacity / cum_capacity[0]) ** (np.log2(progress_ratio))
learning_curve = pd.DataFrame({'cumulative_solar_gw': cum_capacity, 'solar_lcoe_rs_kwh': lcoe})

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(learning_curve['cumulative_solar_gw'], learning_curve['solar_lcoe_rs_kwh'], marker='o')
ax.set_xscale('log', base=2)
ax.set_title("Solar Learning Curve: Wright's Law Scenario")
ax.set_xlabel('Cumulative solar capacity (GW, log scale)')
ax.set_ylabel('LCOE (Rs/kWh)')
plt.tight_layout()
plt.show()

learning_curve

# %%
# Analysis 4: capacity expansion roadmap to 2030
roadmap = pd.DataFrame({
    'source': ['Solar', 'Wind', 'Hydro/Flexible', 'BESS'],
    'current_gw_est': [capacity_gw['Solar'], capacity_gw['Wind'], capacity_gw['Hydro'], 0],
    '2030_target_gw': [280, 140, 65, 75],
})
roadmap['new_capacity_needed_gw'] = (roadmap['2030_target_gw'] - roadmap['current_gw_est']).clip(lower=0)
roadmap['annual_addition_needed_gw'] = roadmap['new_capacity_needed_gw'] / 5
roadmap

# %%
# Final executive recommendation table
recommendations = pd.DataFrame({
    'priority': [1, 2, 3, 4, 5],
    'recommendation': [
        'Build storage and flexible capacity alongside solar and wind additions',
        'Use coal inventory dashboards and EOQ/reorder triggers for critical stations',
        'Prioritise high-readiness states for renewable integration pilots',
        'Introduce gradual carbon pricing or equivalent coal externality correction',
        'Accelerate transmission corridors where renewable capacity is growing fastest',
    ],
    'managerial_logic': [
        'Renewables solve energy quantity, but storage solves peak reliability',
        'Inventory discipline reduces outage risk during peak demand periods',
        'State readiness improves speed of implementation and lowers execution risk',
        'Price signals shift dispatch and investment toward lower-emission sources',
        'Transmission removes bottlenecks between renewable supply and demand centres',
    ],
    'evidence_from_notebook': [
        'Week 1 renewable trends + Week 3 dispatch model + Week 4 BESS NPV',
        'Week 2 coal stock stress + Week 3 EOQ model',
        'Week 2 state capacity + Week 3 readiness score',
        'Week 3 dispatch emissions + Week 4 Pigouvian tax estimate',
        'Week 1/2 state and generation diagnostics',
    ]
})
recommendations

# %%
# # Conclusion
# Renewable energy can materially reduce India's energy deficit and emissions, but it does not automatically solve peak demand unless paired with storage, flexible generation, transmission expansion, and disciplined operational planning. The combined weekwise analysis supports a portfolio answer: scale renewables, add BESS/flexibility, manage coal reliability during transition, and use economic instruments to align private dispatch with public system cost.

"""
MSEDCL Tariff Analysis & ML Scenario Prediction
===============================================
Features:
1. ML Scenario Forecasting (Optimistic, Expected, Pessimistic).
2. Saves PNG images directly to the root folder for GitHub visibility.
3. Saves output CSVs to data/reports/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import warnings
import os
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PATHS (UPDATED FOR GITHUB STRUCTURE)
# ══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "data" / "csv" / "msedcl_ALL_YEARS_master.csv"

# Images go to the root folder (BASE_DIR)
IMG_DIR = BASE_DIR
# CSVs go to the reports folder
CSV_DIR = BASE_DIR / "data" / "reports"
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  STYLE CONFIG
# ══════════════════════════════════════════════════════════════════
PALETTE = {
    'bg': '#0F1117', 'panel': '#1A1D27', 'border': '#2A2D3A',
    'text': '#E8EAF0', 'subtext': '#8B90A0', 'accent': '#4FC3F7',
    'green': '#66BB6A', 'orange': '#FFA726', 'red': '#EF5350',
    'purple': '#AB47BC', 'teal': '#26C6DA', 'gold': '#FFCA28',
    'pink': '#EC407A', 'lime': '#9CCC65',
}

CATEGORY_COLORS = {
    'LT Residential (0-100u)' : PALETTE['accent'],
    'LT Commercial (0-20kW)'  : PALETTE['orange'],
    'LT Industry (0-20kW)'    : PALETTE['green'],
    'LT Agriculture (Metered)': PALETTE['lime'],
    'LT Street Light'         : PALETTE['gold'],
    'HT Industry'             : PALETTE['purple'],
    'HT Commercial'           : PALETTE['pink'],
    'HT Agriculture'          : PALETTE['teal'],
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.facecolor': PALETTE['panel'], 'figure.facecolor': PALETTE['bg'],
    'axes.edgecolor': PALETTE['border'], 'axes.labelcolor': PALETTE['text'],
    'xtick.color': PALETTE['subtext'], 'ytick.color': PALETTE['subtext'],
    'text.color': PALETTE['text'], 'grid.color': PALETTE['border'],
    'grid.linewidth': 0.5, 'axes.grid': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.framealpha': 0.15, 'legend.edgecolor': PALETTE['border'],
    'legend.facecolor': PALETTE['panel'],
})

def _style_ax(ax):
    ax.set_facecolor(PALETTE['panel'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['border'])
    ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
    ax.grid(True, color=PALETTE['border'], linewidth=0.4, alpha=0.7)

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════
def load_and_clean(path):
    print(f"\n{'═'*60}\n  Loading: {path}")
    df = pd.read_csv(path)
    df = df[df['Energy_Charge'].between(0.5, 22)].copy()
    
    # PDF parsing sanity checks
    is_agri_error = df['Category'].str.contains('Agri|Pumpset', case=False, na=False) & (df['Energy_Charge'] > 5.0)
    is_ind_error = df['Category'].str.contains('LT Industry', case=False, na=False) & (df['Energy_Charge'] > 10.0)
    df = df[~(is_agri_error | is_ind_error)]
    
    df['Year'] = df['FY'].str[:4].astype(int)
    df = df[df['Year'] >= 2017].copy()
    print(f"  After filter: {len(df)} rows")
    return df

# ══════════════════════════════════════════════════════════════════
#  STEP 2 — EXTRACT CANONICAL
# ══════════════════════════════════════════════════════════════════
CANONICAL_TARGETS = {
    'LT Residential (0-100u)' : ('LT_Residential', ['1-100', '0-100', '0–100']),
    'LT Commercial (0-20kW)'  : ('LT_Commercial',  ['0 – 20', '0-20', '0–20']),
    'LT Industry (0-20kW)'    : ('LT_Industry',    ['0-20', '0 – 20', '0–20']),
    'LT Agriculture (Metered)': ('LT_Agriculture', ['metered', 'pumpset', 'LT IV(B)']),
    'LT Street Light'         : ('LT_StreetLight', ['street', 'all unit', 'gp', 'gram']),
    'HT Industry'             : ('HT_Industry',    ['general', 'HT I(A)', 'ht i(a)', 'industry - general']),
    'HT Commercial'           : ('HT_Commercial',  ['commercial', 'HT II']),
    'HT Agriculture'          : ('HT_Agriculture', ['pumpset', 'v(a)', 'V(A)']),
}

def extract_canonical(df):
    records = []
    for cat_name, (section, keywords) in CANONICAL_TARGETS.items():
        sub = df[df['Section'] == section].copy()
        if sub.empty: sub = df.copy()

        for fy in sorted(df['FY'].unique()):
            fy_sub = sub[sub['FY'] == fy]
            if fy_sub.empty: continue

            cat_lower = fy_sub['Category'].str.lower()
            scores = pd.Series(0, index=fy_sub.index)
            for kw in keywords:
                scores += cat_lower.str.contains(kw.lower(), na=False).astype(int)

            best_idx = scores.idxmax()
            if scores[best_idx] == 0: continue

            row = fy_sub.loc[best_idx]
            records.append({
                'Category': cat_name, 'FY': fy, 'Year': int(fy[:4]),
                'Energy_Charge': row['Energy_Charge']
            })

    canon = pd.DataFrame(records)
    canon = canon.sort_values('Energy_Charge').drop_duplicates(subset=['Category', 'FY'])
    canon = canon.sort_values(['Category', 'Year']).reset_index(drop=True)
    return canon

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — SCENARIO ML FORECASTING
# ══════════════════════════════════════════════════════════════════
FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029, 2030]

def forecast_category_scenarios(years_hist, values_hist, forecast_years):
    X = np.array(years_hist).reshape(-1, 1)
    y = np.array(values_hist)

    if len(X) < 2: return None, None, None, None

    # Base Linear Model (Expected)
    model = make_pipeline(PolynomialFeatures(1), LinearRegression())
    model.fit(X, y)

    r2 = r2_score(y, model.predict(X))
    X_fut = np.array(forecast_years).reshape(-1, 1)
    
    expected_preds = model.predict(X_fut)
    expected_preds = np.clip(expected_preds, 0.5, values_hist[-1] * 2.0)
    
    # Generate Scenarios (Compounding differences over time)
    years_ahead = np.array(forecast_years) - years_hist[-1]
    
    # Pessimistic: +3% additional compound inflation per year
    pessimistic_preds = expected_preds * ((1 + 0.03) ** years_ahead)
    
    # Optimistic: -2% reduction compared to expected trend per year
    optimistic_preds = expected_preds * ((1 - 0.02) ** years_ahead)

    return expected_preds, optimistic_preds, pessimistic_preds, r2

def build_forecasts(canon):
    forecasts = {}
    for cat in sorted(canon['Category'].unique()):
        sub = canon[canon['Category'] == cat].sort_values('Year')
        if len(sub) < 3: continue

        years = sub['Year'].tolist()
        values = sub['Energy_Charge'].tolist()

        expected, optimistic, pessimistic, r2 = forecast_category_scenarios(years, values, FORECAST_YEARS)
        if expected is not None:
            forecasts[cat] = {
                'hist_years': years, 'hist_vals': values,
                'fore_years': FORECAST_YEARS, 
                'expected': expected.tolist(),
                'optimistic': optimistic.tolist(),
                'pessimistic': pessimistic.tolist(),
                'r2': r2
            }
    return forecasts

def yoy_analysis(canon):
    canon_sorted = canon.sort_values(['Category', 'Year'])
    canon_sorted['YoY_pct'] = canon_sorted.groupby('Category')['Energy_Charge'].pct_change() * 100
    return canon_sorted

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — SEPARATED VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

def plot_historical(canon):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in sorted(canon['Category'].unique()):
        sub = canon[canon['Category'] == cat].sort_values('Year')
        if len(sub) < 2: continue
        ax.plot(sub['Year'], sub['Energy_Charge'], marker='o', markersize=6, 
                linewidth=2, color=CATEGORY_COLORS.get(cat, PALETTE['subtext']), label=cat)

    ax.set_title('1. Historical Energy Charges (2017-2025)', fontsize=14, fontweight='bold', color=PALETTE['text'], pad=15)
    ax.set_xlabel('Financial Year Start', fontsize=11)
    ax.set_ylabel('Energy Charge (Rs/kWh or Rs/kVAh)', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    _style_ax(ax)
    
    plt.savefig(IMG_DIR / '01_historical_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_forecast_scenarios(forecasts):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cat, fdata in forecasts.items():
        color = CATEGORY_COLORS.get(cat, PALETTE['subtext'])
        
        # Historical
        ax.plot(fdata['hist_years'], fdata['hist_vals'], marker='o', markersize=4, linewidth=2, color=color, alpha=0.9)
        
        # Bridge variables
        bridge_x = [fdata['hist_years'][-1]] + fdata['fore_years']
        bridge_y_exp = [fdata['hist_vals'][-1]] + fdata['expected']
        bridge_y_opt = [fdata['hist_vals'][-1]] + fdata['optimistic']
        bridge_y_pes = [fdata['hist_vals'][-1]] + fdata['pessimistic']
        
        # Plot Expected (Solid Dashed)
        ax.plot(bridge_x, bridge_y_exp, linestyle='--', linewidth=2, color=color, label=cat)
        
        # Fill Scenarios (Optimistic to Pessimistic bounds)
        ax.fill_between(bridge_x, bridge_y_opt, bridge_y_pes, color=color, alpha=0.1)

    ax.axvline(x=2024.5, color=PALETTE['gold'], linewidth=2, linestyle=':', label='← History | Scenario Forecast →')
    
    # Custom legend for scenarios
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    legend_elements = [
        mlines.Line2D([0], [0], color=PALETTE['text'], lw=2, linestyle='--', label='Expected ML Trend'),
        mpatches.Patch(color=PALETTE['text'], alpha=0.3, label='Scenario Bounds (Opt. to Pess.)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('2. Advanced AI Scenario Forecast to 2030', fontsize=14, fontweight='bold', color=PALETTE['text'], pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Predicted Rate (Rs/kWh)', fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    _style_ax(ax)
    
    plt.savefig(IMG_DIR / '02_ml_scenario_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_latest_bars(canon, prefix, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = [c for c in sorted(canon['Category'].unique()) if c.startswith(prefix)]
    
    all_fy = sorted(canon['FY'].unique())
    recent_fy = all_fy[-2:]
    if not recent_fy: return
    width = 0.35
    x = np.arange(len(cats))

    for fi, fy in enumerate(recent_fy):
        vals = [canon[(canon['FY'] == fy) & (canon['Category'] == cat)]['Energy_Charge'].values for cat in cats]
        vals = [v[0] if len(v) else 0 for v in vals]

        offset = (fi - 0.5) * width
        color = PALETTE['accent'] if fi == 1 else PALETTE['subtext']
        alpha = 1.0 if fi == 1 else 0.5
        bars = ax.bar(x + offset, vals, width, color=color, alpha=alpha, label=fy)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9, color=PALETTE['text'])

    short_labels = [c.replace(prefix+' ', '').replace(' (0-20kW)', '').replace(' (0-100u)', '').replace(' (Metered)', '') for c in cats]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha='center', fontsize=10)
    ax.set_ylabel('Rate (Rs/kWh)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', color=PALETTE['text'], pad=15)
    ax.legend(fontsize=10)
    _style_ax(ax)
    
    plt.savefig(IMG_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_yoy_heatmap(yoy_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = sorted(yoy_df['Category'].unique())
    fys  = sorted(yoy_df['FY'].unique())[1:]

    matrix = np.full((len(cats), len(fys)), np.nan)
    for ri, cat in enumerate(cats):
        for ci, fy in enumerate(fys):
            row = yoy_df[(yoy_df['Category'] == cat) & (yoy_df['FY'] == fy)]
            if not row.empty and not pd.isna(row['YoY_pct'].values[0]):
                matrix[ri, ci] = row['YoY_pct'].values[0]

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect='auto', cmap='RdYlGn_r', vmin=-10, vmax=15)

    for ri in range(len(cats)):
        for ci in range(len(fys)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                color = 'white' if abs(val) > 8 else PALETTE['bg']
                ax.text(ci, ri, f'{val:.1f}%', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xticks(range(len(fys)))
    ax.set_xticklabels(fys, fontsize=9, rotation=30, ha='right')
    ax.set_title('5. Year-over-Year Rate Hikes (%)', fontsize=14, fontweight='bold', color=PALETTE['text'], pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('YoY Change %', color=PALETTE['text'])
    cbar.ax.yaxis.set_tick_params(color=PALETTE['text'])
    
    plt.savefig(IMG_DIR / '05_yoy_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_summary_cagr(canon):
    fig, ax = plt.subplots(figsize=(8, 5))
    cagrs = {}
    for cat in sorted(canon['Category'].unique()):
        sub = canon[canon['Category'] == cat].sort_values('Year')
        if len(sub) < 2: continue
        n = sub['Year'].max() - sub['Year'].min()
        v0, vn = sub['Energy_Charge'].iloc[0], sub['Energy_Charge'].iloc[-1]
        if v0 > 0 and n > 0:
            cagrs[cat] = ((vn / v0) ** (1 / n) - 1) * 100

    cats_sorted = sorted(cagrs, key=cagrs.get)
    vals = [cagrs[c] for c in cats_sorted]
    colors = [PALETTE['red'] if v >= 0 else PALETTE['green'] for v in vals]

    bars = ax.barh(cats_sorted, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(val + (0.1 if val>=0 else -0.1), bar.get_y() + bar.get_height() / 2, 
                f'{val:.1f}%', va='center', ha='left' if val>=0 else 'right', 
                fontsize=9, color=PALETTE['text'])

    ax.set_title('6. Annual Inflation Rate (CAGR) by Category', fontsize=14, fontweight='bold', color=PALETTE['text'], pad=15)
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=11)
    ax.axvline(x=0, color=PALETTE['border'], linewidth=1)
    _style_ax(ax)
    
    plt.savefig(IMG_DIR / '06_cagr_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'═'*60}\n  MSEDCL TARIFF SCENARIO ANALYSIS\n{'═'*60}")

    df = load_and_clean(INPUT_CSV)
    canon = extract_canonical(df)
    yoy_df = yoy_analysis(canon)
    forecasts = build_forecasts(canon)

    print(f"\n  Generating Charts in Root Directory '{IMG_DIR.name}/' ...")
    plot_historical(canon)
    plot_forecast_scenarios(forecasts)
    plot_latest_bars(canon, 'LT', '3. Low Tension (LT) Tariffs (Latest 2 Years)', '03_lt_tariffs.png')
    plot_latest_bars(canon, 'HT', '4. High Tension (HT) Tariffs (Latest 2 Years)', '04_ht_tariffs.png')
    plot_yoy_heatmap(yoy_df)
    plot_summary_cagr(canon)

    # Save CSVs to the reports folder
    print(f"\n  Saving CSV Data to '{CSV_DIR.name}/' ...")
    canon.to_csv(CSV_DIR / "msedcl_clean_tariffs.csv", index=False)
    yoy_df.to_csv(CSV_DIR / "msedcl_yoy_changes.csv", index=False)

    rows = []
    for cat, fdata in forecasts.items():
        for i, yr in enumerate(fdata['fore_years']):
            rows.append({
                'Category': cat, 
                'Forecast_Year': yr, 
                'Expected_Charge': round(fdata['expected'][i], 3),
                'Optimistic_Charge': round(fdata['optimistic'][i], 3),
                'Pessimistic_Charge': round(fdata['pessimistic'][i], 3)
            })
            
    pd.DataFrame(rows).to_csv(CSV_DIR / "msedcl_scenario_forecast_2026_2030.csv", index=False)

    print(f"\n{'═'*60}\n  ✅ SCENARIO ANALYSIS COMPLETE\n{'═'*60}\n")

if __name__ == '__main__':
    main()
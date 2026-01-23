import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from streamlit_sortables import sort_items

# --- CONFIGURATION ---
DATE_COLUMN = 'Deal date'
VALUE_COLUMN = 'Amount raised (converted to GBP)'
ALT_DATE_COLUMN = ['Date the participant received the grant', 'Deal year', 'Date']
ALT_VALUE_COLUMN = ['Amount received (converted to GBP)', 'Average pre-money valuation (GBP)', 'Amount']

# --- USER COLOR PALETTE ---
PURPLE, DARK_PURPLE, LIGHT_PURPLE = '#6B67DA', '#38358E', '#BBBAF6'
WHITE_PURPLE, BLACK_PURPLE, YELLOW = '#EAEAFF', '#211E52', '#FFB914'

CATEGORY_COLORS = [PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE]
SPLIT_LINE_PALETTE = [PURPLE, DARK_PURPLE, BLACK_PURPLE, YELLOW]

PREDEFINED_COLORS = {
    'Purple': PURPLE, 'Dark Purple': DARK_PURPLE, 'Light Purple': LIGHT_PURPLE,
    'White Purple': WHITE_PURPLE, 'Black Purple': BLACK_PURPLE, 'Yellow': YELLOW
}

SINGLE_BAR_COLOR, PREDICTION_SHADE_COLOR = '#BBBAF6', WHITE_PURPLE
DEFAULT_LINE_COLOR, TITLE_COLOR, DEFAULT_TITLE = '#000000', '#000000', 'Grant Funding and Deal Count Over Time'

st.set_page_config(page_title="Time Series Chart Generator", layout="wide", initial_sidebar_state="expanded")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- HELPER FUNCTIONS ---

def format_currency(value):
    value = float(value)
    if value == 0: return "£0"
    neg, x_abs = value < 0, abs(value)
    if x_abs >= 1e9: unit, divisor = "b", 1e9
    elif x_abs >= 1e6: unit, divisor = "m", 1e6
    elif x_abs >= 1e3: unit, divisor = "k", 1e3
    else: unit, divisor = "", 1.0
    scaled = x_abs / divisor
    s = f"{scaled:.3g}"
    if float(s).is_integer(): s = str(int(float(s)))
    return f"{'-' if neg else ''}£{s}{unit}"

def is_dark_color(hex_color):
    try:
        r, g, b = to_rgb(hex_color)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5
    except: return False

@st.cache_data
def load_data(uploaded_file, sheet_name=None):
    # --- UPDATED: Sheet selection logic ---
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    
    data.columns = data.columns.str.strip()
    
    # Identify Date and Value columns
    for alt in ALT_DATE_COLUMN:
        if alt in data.columns: data.rename(columns={alt: DATE_COLUMN}, inplace=True); break
    for alt in ALT_VALUE_COLUMN:
        if alt in data.columns: data.rename(columns={alt: VALUE_COLUMN}, inplace=True); break

    try:
        # THE FIX FOR 1970 ISSUE: string conversion before parsing
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN].astype(str), format='mixed', errors='coerce')
        for col in data.columns:
            if col not in [DATE_COLUMN, VALUE_COLUMN]: data[col] = data[col].astype(str).str.strip()
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce').fillna(0)
    except Exception as e: return None, f"Error: {e}"
    
    if data.empty: return None, "No valid rows found."
    return data, None

def apply_filter(df, filter_configs):
    if not filter_configs: return df
    temp_df = df.copy()
    for config in filter_configs:
        col, values, include = config['column'], config['values'], config['include']
        if values: temp_df = temp_df[temp_df[col].isin(values)] if include else temp_df[~temp_df[col].isin(values)]
    return temp_df

def process_data(df, year_range, cat_col, line_cat_col, granularity, line_mode):
    df = df.copy()
    chart_data = df[df[DATE_COLUMN].dt.year.between(year_range[0], year_range[1], inclusive='both')].copy()
    if chart_data.empty: return None, "No data available."
    
    chart_data['time_period'] = chart_data[DATE_COLUMN].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else chart_data[DATE_COLUMN].dt.year
    chart_data = chart_data.sort_values(DATE_COLUMN)
    
    if cat_col != 'None':
        grouped = chart_data.groupby(['time_period', cat_col]).agg({VALUE_COLUMN: 'sum'}).reset_index()
        final_data = grouped.pivot(index='time_period', columns=cat_col, values=VALUE_COLUMN).fillna(0).reset_index()
    else: final_data = chart_data.groupby('time_period').agg({VALUE_COLUMN: 'sum'}).reset_index()

    if line_cat_col != 'None':
        metric_col, metric_agg = (VALUE_COLUMN, 'sum') if line_mode == 'Value' else (DATE_COLUMN, 'count')
        line_grouped = chart_data.groupby(['time_period', line_cat_col])[metric_col].agg(metric_agg).reset_index(name='metric')
        line_pivot = line_grouped.pivot(index='time_period', columns=line_cat_col, values='metric').fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        metric = chart_data.groupby('time_period')[VALUE_COLUMN].sum() if line_mode == 'Value' else chart_data.groupby('time_period').size()
        final_data = final_data.merge(metric.reset_index(name='line_metric'), on='time_period', how='left').fillna(0)
    return final_data, None

def generate_chart(final_data, cat_col, show_bars, show_line, title, colors, order, pred_y, line_cat_col, granularity, line_mode):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_pos, time_labels = np.arange(len(final_data)), final_data['time_period'].values
    is_pred = (time_labels.astype(int) >= pred_y) if granularity == 'Yearly' and pred_y else np.full(len(time_labels), False)
    font_size = int(max(8, min(22, 150 / len(final_data))))
    
    bar_cols = [c for c in final_data.columns if not str(c).startswith('line_split_') and c not in ['time_period', 'line_metric']]
    if order: bar_cols.sort(key=lambda x: order.get(x, 999))
    y_max = final_data[bar_cols].sum(axis=1).max() if bar_cols else final_data[VALUE_COLUMN].max()

    if cat_col != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(bar_cols):
            c = colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
            for i in range(len(final_data)):
                val = final_data[cat].iloc[i]
                if show_bars and val > 0:
                    bc = PREDICTION_SHADE_COLOR if is_pred[i] else c
                    ax1.bar(x_pos[i], val, 0.8, bottom=bottom[i], color=bc, hatch='xx' if is_pred[i] else None, edgecolor='black' if is_pred[i] else 'none', linewidth=0)
                    ax1.text(x_pos[i], (bottom[i] + y_max*0.01) if idx == 0 else (bottom[i] + val/2), format_currency(val), ha='center', va='bottom' if idx == 0 else 'center', fontsize=font_size, fontweight='bold', color='#FFFFFF' if is_dark_color(bc) else '#000000')
                bottom[i] += val
    elif show_bars:
        for i in range(len(final_data)):
            val, bc = final_data[VALUE_COLUMN].iloc[i], (PREDICTION_SHADE_COLOR if is_pred[i] else SINGLE_BAR_COLOR)
            ax1.bar(x_pos[i], val, 0.8, color=bc, hatch='xx' if is_pred[i] else None, edgecolor='black' if is_pred[i] else 'none', linewidth=0)
            if val > 0: ax1.text(x_pos[i], y_max*0.01, format_currency(val), ha='center', va='bottom', fontsize=font_size, fontweight='bold', color='#000000')

    ax1.set_xticks(x_pos); ax1.set_xticklabels(time_labels, fontsize=font_size); ax1.set_ylim(0, y_max * 1.15)
    ax1.tick_params(left=False, labelleft=False, length=0); [s.set_visible(False) for s in ax1.spines.values()]

    if show_line:
        ax2 = ax1.twinx()
        line_cols = [c for c in final_data.columns if c.startswith('line_split_')] if line_cat_col != 'None' else ['line_metric']
        for idx, l_col in enumerate(line_cols):
            lc = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_cat_col != 'None' else DEFAULT_LINE_COLOR
            ax2.plot(x_pos, final_data[l_col].values, color=lc, marker='o', linestyle='-', linewidth=2.5, markersize=8)
            for i, y in enumerate(final_data[l_col].values):
                ax2.text(x_pos[i], y + final_data[line_cols].values.max()*0.05, str(int(y)) if line_mode == 'Count' else format_currency(y), ha='center', va='bottom', fontsize=font_size, color=lc, fontweight='bold')
        ax2.axis('off'); ax2.set_ylim(0, final_data[line_cols].values.max() * 1.6 if final_data[line_cols].values.max() > 0 else 1)

    handles = []
    if show_bars:
        if cat_col != 'None': handles += [Line2D([0], [0], marker='s', color='w', markerfacecolor=colors.get(c, PURPLE), markersize=12, label=c) for c in bar_cols]
        else: handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=SINGLE_BAR_COLOR, markersize=12, label='Amount raised'))
    if show_line:
        if line_cat_col != 'None': handles += [Line2D([0], [0], color=SPLIT_LINE_PALETTE[i % len(SPLIT_LINE_PALETTE)], marker='o', label=f"{c.replace('line_split_', '')} ({line_mode})") for i, c in enumerate(line_cols)]
        else: handles.append(Line2D([0], [0], color=DEFAULT_LINE_COLOR, marker='o', label=f"{'Deals' if line_mode == 'Count' else 'Value'}"))
    if handles: ax1.legend(handles=handles, loc='upper left', frameon=False, prop={'size': 14}, ncol=2)
    plt.title(title, fontsize=22, fontweight='bold', pad=30); return fig

# --- APP LAYOUT ---
st.markdown(f'<h1 style="color:{BLACK_PURPLE};">Time Series Chart Generator</h1>', unsafe_allow_html=True)
st.markdown(f'<div style="background:{WHITE_PURPLE}; padding:20px; border-radius:10px; border-left:5px solid {YELLOW}; margin:15px 0;"><p style="margin:0; font-size:16px; color:#000;"><strong>Turn fundraising exports into time series charts – JT</strong></p><a href="https://platform.beauhurst.com/search/advancedsearch/" target="_blank" style="display:inline-block; background:#fff; padding:10px 16px; border-radius:6px; border:1px solid #ddd; color:{PURPLE}; font-weight:600; text-decoration:none; font-size:14px;">🔗 Beauhurst Advanced Search</a></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Data Source")
    file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'xls', 'csv'])
    sheet = None
    if file and file.name.endswith(('.xlsx', '.xls')):
        xl = pd.ExcelFile(file)
        sheet = st.selectbox("Select Sheet", xl.sheet_names)
    
    if file:
        df_base, err = load_data(file, sheet_name=sheet)
        if df_base is not None:
            st.header("2. Chart Title"); title = st.text_input("Title", DEFAULT_TITLE)
            st.header("3. Time Filters"); granularity = st.radio("Granularity", ['Yearly', 'Quarterly'])
            min_y, max_y = int(df_base[DATE_COLUMN].dt.year.min()), int(df_base[DATE_COLUMN].dt.year.max())
            s_y = st.selectbox("Start Year", list(range(min_y, max_y + 1)), index=0)
            e_y = st.selectbox("End Year", list(range(min_y, max_y + 1)), index=(max_y-min_y))
            st.header("4. Visual Elements"); show_bars = st.checkbox("Show Bars", True); show_line = st.checkbox("Show Line", True)
            line_mode, line_cat = ('Count', 'None')
            if show_line:
                line_mode = st.radio("Line Metric", ['Count', 'Value'])
                line_cat = st.selectbox("Split Line Category", ['None'] + [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
            pred_y = st.selectbox("Prediction Start", list(range(s_y, e_y + 1))) if granularity == 'Yearly' and st.checkbox("Enable Predictions") else None
            st.header("5. Stacked Bar"); stack_col, colors, order = ('None', {}, {})
            if st.checkbox('Enable Stacked Bar'):
                stack_col = st.selectbox("Column", [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                unique_cats = sorted(df_base[stack_col].unique()); sorted_cats = sort_items(unique_cats, key='sort_bars')
                colors = {c: st.selectbox(f"Color: {c}", list(PREDEFINED_COLORS.values()), index=i%6) for i, c in enumerate(sorted_cats)}; order = {c: i for i,c in enumerate(sorted_cats)}
            st.header("6. Data Filter"); configs = []
            if st.checkbox('Enable Filtering'):
                sel_f = st.multiselect("Columns", sorted([c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]]))
                for f in sel_f:
                    with st.expander(f"Filter: {f}", expanded=True):
                        vals = st.multiselect(f"Values", df_base[f].unique(), key=f"f_v_{f}")
                        configs.append({'column': f, 'values': vals, 'include': st.radio(f"Mode", ["Include", "Exclude"], key=f"f_m_{f}") == "Include"})
            st.header("7. Download"); buf_p, buf_s = BytesIO(), BytesIO()

if file and 'df_base' in locals() and df_base is not None:
    df_f = apply_filter(df_base, configs)
    final, err_p = process_data(df_f, (s_y, e_y), stack_col, line_cat, granularity, line_mode)
    if final is not None:
        col_chart, _ = st.columns([4, 1])
        with col_chart:
            fig = generate_chart(final, stack_col, show_bars, show_line, title, colors, order, pred_y, line_cat, granularity, line_mode)
            st.pyplot(fig)
        fig.savefig(buf_p, format='png', dpi=300, bbox_inches='tight'); fig.savefig(buf_s, format='svg', bbox_inches='tight')
        st.sidebar.download_button("Download PNG", buf_p.getvalue(), "chart.png", use_container_width=True)
        st.sidebar.download_button("Download Adobe SVG", buf_s.getvalue(), "chart.svg", use_container_width=True)
else: st.info("⬆️ Please upload your data file in the sidebar to begin.")

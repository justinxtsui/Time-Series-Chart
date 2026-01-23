import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from streamlit_sortables import sort_items

# --- CONFIGURATION ---
DATE_COLUMN = 'Deal date'
VALUE_COLUMN = 'Amount raised (converted to GBP)'
# Added 'Deal year' for compatibility with your valuation exports
ALT_DATE_COLUMN = ['Date the participant received the grant', 'Deal year']
ALT_VALUE_COLUMN = ['Amount received (converted to GBP)', 'Average pre-money valuation (GBP)']

# --- USER COLOR PALETTE ---
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
LIGHT_PURPLE = '#BBBAF6'
WHITE_PURPLE = '#EAEAFF'
BLACK_PURPLE = '#211E52'
YELLOW = '#FFB914' 

# Mapping palette for categorical data (Bars)
CATEGORY_COLORS = [PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE] 
# Palette for Split Lines prioritizing requested hex codes
SPLIT_LINE_PALETTE = [PURPLE, DARK_PURPLE, BLACK_PURPLE, YELLOW]

PREDEFINED_COLORS = {
    'Purple': PURPLE,
    'Dark Purple': DARK_PURPLE,
    'Light Purple': LIGHT_PURPLE,
    'White Purple': WHITE_PURPLE,
    'Black Purple': BLACK_PURPLE,
    'Yellow': YELLOW
}

SINGLE_BAR_COLOR = '#BBBAF6'
PREDICTION_SHADE_COLOR = WHITE_PURPLE 
PREDICTION_HATCH_COLOR = '#000000'
DEFAULT_LINE_COLOR = '#000000' 
TITLE_COLOR = '#000000'
APP_TITLE_COLOR = '#000000'
DEFAULT_TITLE = 'Grant Funding and Deal Count Over Time'

st.set_page_config(page_title="Time Series Chart Generator", layout="wide", initial_sidebar_state="expanded")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- HELPER FUNCTIONS ---

def format_currency(value):
    value = float(value)
    if value == 0:
        return "£0"
    neg = value < 0
    x_abs = abs(value)
    if x_abs >= 1e9:
        unit, divisor = "b", 1e9
    elif x_abs >= 1e6:
        unit, divisor = "m", 1e6
    elif x_abs >= 1e3:
        unit, divisor = "k", 1e3
    else:
        unit, divisor = "", 1.0
    scaled = x_abs / divisor
    s = f"{scaled:.3g}"
    try:
        if float(s).is_integer(): s = str(int(float(s)))
    except: pass
    sign = "-" if neg else ""
    return f"{sign}£{s}{unit}"

def is_dark_color(hex_color):
    try:
        r, g, b = to_rgb(hex_color)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
        return luminance < 0.5
    except ValueError: return False

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'): data = pd.read_csv(uploaded_file)
    else: data = pd.read_excel(uploaded_file, sheet_name=0)
    
    data.columns = data.columns.str.strip()
    original_value_column = 'raised'
    
    # Check and rename date column
    if DATE_COLUMN not in data.columns:
        found_date = False
        for alt in ALT_DATE_COLUMN:
            if alt in data.columns:
                data.rename(columns={alt: DATE_COLUMN}, inplace=True)
                found_date = True
                break
        if not found_date:
            return None, f"File must contain a date column.", None

    # Check and rename value column
    if VALUE_COLUMN not in data.columns:
        found_val = False
        for alt in ALT_VALUE_COLUMN:
            if alt in data.columns:
                data.rename(columns={alt: VALUE_COLUMN}, inplace=True)
                found_val = True
                break
        if not found_val:
            return None, f"File must contain a value column.", None

    try:
        # --- THE FIX FOR 1970 ISSUE ---
        # Converting to string first prevents Pandas from treating integers as nanosecond offsets
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN].astype(str), format='mixed', errors='coerce')
        # ------------------------------
        
        for col in data.columns:
            if col not in [DATE_COLUMN, VALUE_COLUMN]:
                data[col] = data[col].astype(str).str.strip() 
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce').fillna(0)
    except Exception as e:
        return None, f"Error: {e}", None
    
    if data.empty: return None, "No valid rows found.", None
    return data, None, original_value_column

@st.cache_data
def apply_filter(df, filter_configs):
    if not filter_configs: return df
    temp_df = df.copy()
    for config in filter_configs:
        col, values, is_include = config['column'], config['values'], config['include']
        if values:
            temp_df = temp_df[temp_df[col].isin(values)] if is_include else temp_df[~temp_df[col].isin(values)]
    return temp_df

@st.cache_data
def process_data(df, year_range, category_column, line_category_column='None', granularity='Yearly', line_mode='Count'):
    df = df.copy()
    start_year, end_year = year_range
    chart_data = df[df[DATE_COLUMN].dt.year.between(start_year, end_year, inclusive='both')].copy()
    if chart_data.empty: return None, "No data available."
    
    chart_data['time_period'] = chart_data[DATE_COLUMN].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else chart_data[DATE_COLUMN].dt.year
    chart_data = chart_data.sort_values(DATE_COLUMN)
    
    if category_column != 'None':
        grouped = chart_data.groupby(['time_period', category_column]).agg({VALUE_COLUMN: 'sum'}).reset_index()
        final_data = grouped.pivot(index='time_period', columns=category_column, values=VALUE_COLUMN).fillna(0).reset_index()
    else: final_data = chart_data.groupby('time_period').agg({VALUE_COLUMN: 'sum'}).reset_index()

    if line_category_column != 'None':
        line_grouped = chart_data.groupby(['time_period', line_category_column]).agg({VALUE_COLUMN: 'sum' if line_mode == 'Value' else 'size'}).reset_index()
        line_grouped.columns = ['time_period', line_category_column, 'metric']
        line_pivot = line_grouped.pivot(index='time_period', columns=line_category_column, values='metric').fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        metric = chart_data.groupby('time_period')[VALUE_COLUMN].sum() if line_mode == 'Value' else chart_data.groupby('time_period').size()
        final_data = final_data.merge(metric.reset_index(name='line_metric'), on='time_period', how='left').fillna(0)
    return final_data, None

def generate_chart(final_data, category_column, show_bars, show_line, chart_title, category_colors=None, category_order=None, prediction_start_year=None, line_category_column='None', granularity='Yearly', line_mode='Count'):
    chart_fig, chart_ax1 = plt.subplots(figsize=(20, 10))
    x_pos = np.arange(len(final_data))
    time_labels = final_data['time_period'].values
    is_predicted = (time_labels.astype(int) >= prediction_start_year) if granularity == 'Yearly' and prediction_start_year else np.full(len(time_labels), False)
    num_bars = len(final_data)
    DYNAMIC_FONT_SIZE = int(max(8, min(22, 150 / num_bars))) if num_bars > 0 else 12

    category_cols = [col for col in final_data.columns if not str(col).startswith('line_split_') and col not in ['time_period', 'line_metric']]
    if category_order: category_cols.sort(key=lambda x: category_order.get(x, 999))
    y_max = final_data[category_cols].sum(axis=1).max()
    
    if category_column != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            color = category_colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
            for i in range(len(final_data)):
                val = final_data[cat].iloc[i]
                if show_bars and val > 0:
                    bc = PREDICTION_SHADE_COLOR if is_predicted[i] else color
                    chart_ax1.bar(x_pos[i], val, 0.8, bottom=bottom[i], color=bc, hatch='xx' if is_predicted[i] else None, edgecolor='black' if is_predicted[i] else 'none', linewidth=0)
                    chart_ax1.text(x_pos[i], (bottom[i] + y_max*0.01) if idx == 0 else (bottom[i] + val/2), format_currency(val), ha='center', va='bottom' if idx == 0 else 'center', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color='#FFFFFF' if is_dark_color(bc) else '#000000')
                bottom[i] += val
    elif show_bars:
        for i in range(len(final_data)):
            val = final_data[VALUE_COLUMN].iloc[i]
            bc = PREDICTION_SHADE_COLOR if is_predicted[i] else SINGLE_BAR_COLOR
            chart_ax1.bar(x_pos[i], val, 0.8, color=bc, hatch='xx' if is_predicted[i] else None, edgecolor='black' if is_predicted[i] else 'none', linewidth=0)
            if val > 0: chart_ax1.text(x_pos[i], y_max*0.01, format_currency(val), ha='center', va='bottom', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color='#000000')

    chart_ax1.set_xticks(x_pos)
    chart_ax1.set_xticklabels(time_labels, fontsize=DYNAMIC_FONT_SIZE)
    chart_ax1.set_ylim(0, y_max * 1.1)
    chart_ax1.tick_params(axis='both', which='both', length=0, labelleft=False)
    for s in chart_ax1.spines.values(): s.set_visible(False)

    if show_line:
        chart_ax2 = chart_ax1.twinx()
        line_cols = [col for col in final_data.columns if col.startswith('line_split_')] if line_category_column != 'None' else ['line_metric']
        for idx, l_col in enumerate(line_cols):
            l_color = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_category_column != 'None' else DEFAULT_LINE_COLOR
            y_vals = final_data[l_col].values
            chart_ax2.plot(x_pos, y_vals, color=l_color, marker='o', linestyle='-', linewidth=2.5)
            for i, y in enumerate(y_vals):
                chart_ax2.text(x_pos[i], y + y_vals.max()*0.05, str(int(y)) if line_mode == 'Count' else format_currency(y), ha='center', va='bottom', fontsize=DYNAMIC_FONT_SIZE, color=l_color, fontweight='bold')
        chart_ax2.set_ylim(0, final_data[line_cols].values.max() * 1.5 if final_data[line_cols].values.max() > 0 else 10)
        chart_ax2.axis('off')

    plt.title(chart_title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    return chart_fig

# --- APP LAYOUT ---
st.markdown(f'<h1 style="color:{BLACK_PURPLE};">Time Series Chart Generator</h1>', unsafe_allow_html=True)
if 'buf_png' not in st.session_state:
    st.session_state.update({'year_range': (2016, 2025), 'granularity': 'Yearly', 'line_mode': 'Count', 'show_bars': True, 'show_line': True, 'chart_title': DEFAULT_TITLE})

with st.sidebar:
    st.header("1. Data Source")
    file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'xls', 'csv'])
    if file:
        df_base, err, orig_val = load_data(file)
        if df_base is not None:
            st.header("2. Settings")
            st.session_state['chart_title'] = st.text_input("Title", st.session_state['chart_title'])
            st.session_state['granularity'] = st.radio("Time Granularity", ['Yearly', 'Quarterly'])
            st.session_state['show_bars'] = st.checkbox("Show bars", value=True)
            st.session_state['show_line'] = st.checkbox("Show line", value=True)
            if st.session_state['show_line']:
                st.session_state['line_mode'] = st.radio("Line Mode", ['Count', 'Value'])
                line_cat = st.selectbox("Split Line?", ['None'] + [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
            
            if st.checkbox('Enable Stacked Bar?'):
                stack_col = st.selectbox("Stack?", [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                unique_cats = sorted(df_base[stack_col].unique())
                sorted_cats = sort_items(unique_cats, key='sort_cats')
                colors = {c: st.selectbox(f"Color: {c}", list(PREDEFINED_COLORS.values()), index=i%6) for i, c in enumerate(sorted_cats)}
            else: stack_col, sorted_cats, colors = 'None', [], {}

            if st.checkbox('Filter Data?'):
                f_cols = st.multiselect("Columns", [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                configs = [{'column': c, 'values': st.multiselect(f"Vals: {c}", df_base[c].unique()), 'include': st.radio(f"Mode: {c}", ["Include", "Exclude"]) == "Include"} for c in f_cols]
            else: configs = []

            if st.button("Generate Chart"):
                df_f = apply_filter(df_base, configs)
                final, err_p = process_data(df_f, st.session_state['year_range'], stack_col, line_cat if st.session_state['show_line'] else 'None', st.session_state['granularity'], st.session_state['line_mode'])
                if final is not None:
                    fig = generate_chart(final, stack_col, st.session_state['show_bars'], st.session_state['show_line'], st.session_state['chart_title'], colors, {c: i for i,c in enumerate(sorted_cats)}, None, line_cat, st.session_state['granularity'], st.session_state['line_mode'])
                    st.pyplot(fig)
                    buf_p, buf_s = BytesIO(), BytesIO()
                    fig.savefig(buf_p, format='png', dpi=300)
                    fig.savefig(buf_s, format='svg')
                    st.download_button("Download PNG", buf_p.getvalue(), "chart.png")
                    st.download_button("Download Adobe SVG", buf_s.getvalue(), "chart.svg")
        else: st.error(err)

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
ALT_DATE_COLUMN = ['Date the participant received the grant', 'Deal year', 'Date']
ALT_VALUE_COLUMN = ['Amount received (converted to GBP)', 'Average pre-money valuation (GBP)', 'Amount']

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
    if value == 0: return "£0"
    neg = value < 0
    x_abs = abs(value)
    if x_abs >= 1e9: unit, divisor = "b", 1e9
    elif x_abs >= 1e6: unit, divisor = "m", 1e6
    elif x_abs >= 1e3: unit, divisor = "k", 1e3
    else: unit, divisor = "", 1.0
    scaled = x_abs / divisor
    s = f"{scaled:.3g}"
    try:
        if float(s).is_integer(): s = str(int(float(s)))
    except: pass
    return f"{'-' if neg else ''}£{s}{unit}"

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
    
    if DATE_COLUMN not in data.columns:
        for alt in ALT_DATE_COLUMN:
            if alt in data.columns:
                data.rename(columns={alt: DATE_COLUMN}, inplace=True)
                break
    if VALUE_COLUMN not in data.columns:
        for alt in ALT_VALUE_COLUMN:
            if alt in data.columns:
                data.rename(columns={alt: VALUE_COLUMN}, inplace=True)
                break

    try:
        # THE FIX FOR 1970 ISSUE: string conversion before datetime parse
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN].astype(str), format='mixed', errors='coerce')
        for col in data.columns:
            if col not in [DATE_COLUMN, VALUE_COLUMN]:
                data[col] = data[col].astype(str).str.strip() 
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce').fillna(0)
    except Exception as e: return None, f"An error occurred: {e}", None
    
    if data.empty: return None, "No valid rows found.", None
    return data, None, 'raised'

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
        metric_col = VALUE_COLUMN if line_mode == 'Value' else DATE_COLUMN
        metric_agg = 'sum' if line_mode == 'Value' else 'count'
        line_grouped = chart_data.groupby(['time_period', line_category_column])[metric_col].agg(metric_agg).reset_index(name='metric')
        line_pivot = line_grouped.pivot(index='time_period', columns=line_category_column, values='metric').fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        metric = chart_data.groupby('time_period')[VALUE_COLUMN].sum() if line_mode == 'Value' else chart_data.groupby('time_period').size()
        final_data = final_data.merge(metric.reset_index(name='line_metric'), on='time_period', how='left').fillna(0)
    return final_data, None

def generate_chart(final_data, category_column, show_bars, show_line, chart_title, category_colors, category_order, prediction_start_year, line_category_column, granularity, line_mode):
    chart_fig, chart_ax1 = plt.subplots(figsize=(20, 10))
    x_pos = np.arange(len(final_data))
    time_labels = final_data['time_period'].values
    is_predicted = (time_labels.astype(int) >= prediction_start_year) if granularity == 'Yearly' and prediction_start_year else np.full(len(time_labels), False)
    DYNAMIC_FONT_SIZE = int(max(8, min(22, 150 / len(final_data))))
    
    category_cols = [col for col in final_data.columns if not str(col).startswith('line_split_') and col not in ['time_period', 'line_metric']]
    if category_order: category_cols.sort(key=lambda x: category_order.get(x, 999))
    y_max = final_data[category_cols].sum(axis=1).max() if category_cols else final_data[VALUE_COLUMN].max()

    if category_column != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            color = category_colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
            for i in range(len(final_data)):
                val = final_data[cat].iloc[i]
                if show_bars and val > 0:
                    bc = PREDICTION_SHADE_COLOR if is_predicted[i] else color
                    chart_ax1.bar(x_pos[i], val, 0.8, bottom=bottom[i], color=bc, hatch='xx' if is_predicted[i] else None, edgecolor='black' if is_predicted[i] else 'none', linewidth=0)
                    t_c = '#FFFFFF' if is_dark_color(bc) else '#000000'
                    chart_ax1.text(x_pos[i], (bottom[i] + y_max*0.01) if idx == 0 else (bottom[i] + val/2), format_currency(val), ha='center', va='bottom' if idx == 0 else 'center', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color=t_c)
                bottom[i] += val
    elif show_bars:
        for i in range(len(final_data)):
            val = final_data[VALUE_COLUMN].iloc[i]
            bc = PREDICTION_SHADE_COLOR if is_predicted[i] else SINGLE_BAR_COLOR
            chart_ax1.bar(x_pos[i], val, 0.8, color=bc, hatch='xx' if is_predicted[i] else None, edgecolor='black' if is_predicted[i] else 'none', linewidth=0)
            if val > 0: chart_ax1.text(x_pos[i], y_max*0.01, format_currency(val), ha='center', va='bottom', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color='#000000')

    chart_ax1.set_xticks(x_pos)
    chart_ax1.set_xticklabels(time_labels, fontsize=DYNAMIC_FONT_SIZE)
    chart_ax1.set_ylim(0, y_max * 1.15)
    chart_ax1.tick_params(left=False, labelleft=False, length=0)
    for s in chart_ax1.spines.values(): s.set_visible(False)

    if show_line:
        chart_ax2 = chart_ax1.twinx()
        line_cols = [c for c in final_data.columns if c.startswith('line_split_')] if line_category_column != 'None' else ['line_metric']
        for idx, l_col in enumerate(line_cols):
            l_color = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_category_column != 'None' else DEFAULT_LINE_COLOR
            y_vals = final_data[l_col].values
            chart_ax2.plot(x_pos, y_vals, color=l_color, marker='o', linestyle='-', linewidth=2.5, markersize=8)
            for i, y in enumerate(y_vals):
                label = str(int(y)) if line_mode == 'Count' else format_currency(y)
                chart_ax2.text(x_pos[i], y + y_vals.max()*0.05, label, ha='center', va='bottom', fontsize=DYNAMIC_FONT_SIZE, color=l_color, fontweight='bold')
        chart_ax2.axis('off')
        chart_ax2.set_ylim(0, final_data[line_cols].values.max() * 1.6 if final_data[line_cols].values.max() > 0 else 1)

    plt.title(chart_title, fontsize=22, fontweight='bold', pad=30)
    return chart_fig

# --- APP LAYOUT ---
st.markdown(f'<h1 style="color:{BLACK_PURPLE};">Time Series Chart Generator</h1>', unsafe_allow_html=True)
st.markdown(f"""<div style="background: {WHITE_PURPLE}; padding: 20px; border-radius: 10px; border-left: 5px solid {YELLOW}; margin: 15px 0;">
    <p style="margin: 0 0 10px 0; font-size: 16px; color: #000;"><strong>Turn any fundraising or grant export into a time series chart – JT</strong></p>
    <a href="https://platform.beauhurst.com/search/advancedsearch/" target="_blank" style="display: inline-block; background: #fff; padding: 10px 16px; border-radius: 6px; border: 1px solid #ddd; color: {PURPLE}; font-weight: 600; text-decoration: none; font-size: 14px;">🔗 Beauhurst Advanced Search</a>
    </div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Data Source")
    file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'xls', 'csv'])
    if file:
        df_base, err, _ = load_data(file)
        if df_base is not None:
            st.header("2. Chart Title")
            title = st.text_input("Title", DEFAULT_TITLE)
            st.header("3. Time Filters")
            granularity = st.radio("Time Granularity", ['Yearly', 'Quarterly'])
            min_y, max_y = int(df_base[DATE_COLUMN].dt.year.min()), int(df_base[DATE_COLUMN].dt.year.max())
            start_year = st.selectbox("Start Year", list(range(min_y, max_y + 1)), index=0)
            end_year = st.selectbox("End Year", list(range(min_y, max_y + 1)), index=(max_y-min_y))
            st.header("4. Visual Elements")
            show_bars = st.checkbox("Show Bars", True)
            show_line = st.checkbox("Show Line", True)
            if show_line:
                line_mode = st.radio("Line Metric", ['Count', 'Value'])
                line_cat = st.selectbox("Split Line by Category", ['None'] + [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
            else: line_mode, line_cat = 'Count', 'None'
            if granularity == 'Yearly':
                if st.checkbox("Enable Prediction Visuals"):
                    pred_y = st.selectbox("Prediction Start", list(range(start_year, end_year + 1)))
                else: pred_y = None
            else: pred_y = None
            st.header("5. Stacked Bar (Optional)")
            if st.checkbox('Enable Stacked Bar'):
                stack_col = st.selectbox("Stack Column", [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                unique_cats = sorted(df_base[stack_col].unique())
                sorted_cats = sort_items(unique_cats, key='sort_bars')
                colors = {c: st.selectbox(f"Color: {c}", list(PREDEFINED_COLORS.values()), index=i%6) for i, c in enumerate(sorted_cats)}
                order = {c: i for i,c in enumerate(sorted_cats)}
            else: stack_col, colors, order = 'None', {}, {}
            st.header("6. Data Filter")
            if st.checkbox('Enable Multi-Column Filtering'):
                all_cols = sorted([c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                sel_f = st.multiselect("Columns to filter", all_cols)
                configs = []
                for f in sel_f:
                    with st.expander(f"Filter: {f}", expanded=True):
                        vals = st.multiselect(f"Values for {f}", df_base[f].unique(), key=f"f_v_{f}")
                        mode = st.radio(f"Mode for {f}", ["Include", "Exclude"], key=f"f_m_{f}")
                        configs.append({'column': f, 'values': vals, 'include': mode == "Include"})
            else: configs = []
            st.header("7. Download")
            buf_p, buf_s = BytesIO(), BytesIO()

if file and 'df_base' in locals() and df_base is not None:
    df_f = apply_filter(df_base, configs)
    final, err_p = process_data(df_f, (start_year, end_year), stack_col, line_cat, granularity, line_mode)
    if final is not None:
        col_chart, _ = st.columns([4, 1])
        with col_chart:
            fig = generate_chart(final, stack_col, show_bars, show_line, title, colors, order, pred_y, line_cat, granularity, line_mode)
            st.pyplot(fig)
        fig.savefig(buf_p, format='png', dpi=300, bbox_inches='tight')
        fig.savefig(buf_s, format='svg', bbox_inches='tight')
        st.sidebar.download_button("Download PNG", buf_p.getvalue(), "chart.png", use_container_width=True)
        st.sidebar.download_button("Download Adobe SVG", buf_s.getvalue(), "chart.svg", use_container_width=True)
else: st.info("⬆️ Please upload your data file in the sidebar to begin.")

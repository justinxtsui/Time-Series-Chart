import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from streamlit_sortables import sort_items

# --- THE FIX FOR EDITABLE SVG TEXT ---
plt.rcParams['svg.fonttype'] = 'none' 

# --- CONFIGURATION & PALETTE ---
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

# --- CUSTOM CSS FOR SIDEBAR & BRANDING ---
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #f8f9fb;
        border-right: 1px solid #e6e9ef;
    }}
    [data-testid="stSidebar"] h2 {{
        color: {DARK_PURPLE};
        font-size: 1.2rem;
        border-bottom: 2px solid {LIGHT_PURPLE};
        padding-bottom: 5px;
        margin-top: 20px;
    }}
    .app-title {{
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
        color: {BLACK_PURPLE};
        margin-bottom: 0px;
        line-height: 1.1;
    }}
    .app-attribution {{
        font-size: 24px;
        font-weight: 600;
        color: {BLACK_PURPLE};
        margin-top: 0px;
        margin-bottom: 10px;
    }}
    .app-subtitle {{
        color: #000000;
        font-size: 18px;
        margin-bottom: 5px;
        font-weight: normal;
    }}
    .bold-divider {{
        height: 3px;
        background-color: #e6e9ef;
        border: none;
        margin-top: 10px;
        margin-bottom: 25px;
    }}
    </style>
    """, unsafe_allow_html=True)

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
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    return data

def apply_filter(df, filter_configs):
    if not filter_configs: return df
    temp_df = df.copy()
    for config in filter_configs:
        col, values, include = config['column'], config['values'], config['include']
        if values: temp_df = temp_df[temp_df[col].isin(values)] if include else temp_df[~temp_df[col].isin(values)]
    return temp_df

def process_data(df, date_col, value_col, year_range, cat_col, line_cat_col, granularity, line_mode):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='mixed', errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
    
    chart_data = df[df[date_col].dt.year.between(year_range[0], year_range[1], inclusive='both')].copy()
    if chart_data.empty: return None, "No data available."
    
    chart_data['time_period'] = chart_data[date_col].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else chart_data[date_col].dt.year
    chart_data = chart_data.sort_values(date_col)
    
    if cat_col != 'None':
        grouped = chart_data.groupby(['time_period', cat_col]).agg({value_col: 'sum'}).reset_index()
        final_data = grouped.pivot(index='time_period', columns=cat_col, values=value_col).fillna(0).reset_index()
    else: 
        final_data = chart_data.groupby('time_period').agg({value_col: 'sum'}).reset_index()

    if line_cat_col != 'None':
        metric_col, metric_agg = (value_col, 'sum') if line_mode == 'Value' else (date_col, 'count')
        line_grouped = chart_data.groupby(['time_period', line_cat_col])[metric_col].agg(metric_agg).reset_index(name='metric')
        line_pivot = line_grouped.pivot(index='time_period', columns=line_cat_col, values='metric').fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        metric = chart_data.groupby('time_period')[value_col].sum() if line_mode == 'Value' else chart_data.groupby('time_period').size()
        final_data = final_data.merge(metric.reset_index(name='line_metric'), on='time_period', how='left').fillna(0)
    return final_data, None

def generate_chart(final_data, value_col, cat_col, show_bars, show_line, title, y_axis_title, colors, order, pred_y, line_cat_col, granularity, line_mode):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_pos, time_labels = np.arange(len(final_data)), final_data['time_period'].values
    is_pred = (time_labels.astype(int) >= pred_y) if granularity == 'Yearly' and pred_y else np.full(len(time_labels), False)
    font_size = int(max(8, min(22, 150 / len(final_data))))
    
    bar_cols = [c for c in final_data.columns if not str(c).startswith('line_split_') and c not in ['time_period', 'line_metric']]
    if order: bar_cols.sort(key=lambda x: order.get(x, 999))
    y_max = final_data[bar_cols].sum(axis=1).max() if bar_cols else final_data[value_col].max()

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
            val, bc = final_data[value_col].iloc[i], (PREDICTION_SHADE_COLOR if is_pred[i] else SINGLE_BAR_COLOR)
            ax1.bar(x_pos[i], val, 0.8, color=bc, hatch='xx' if is_pred[i] else None, edgecolor='black' if is_pred[i] else 'none', linewidth=0)
            if val > 0: ax1.text(x_pos[i], y_max*0.01, format_currency(val), ha='center', va='bottom', fontsize=font_size, fontweight='bold', color='#000000')

    ax1.set_xticks(x_pos); ax1.set_xticklabels(time_labels, fontsize=font_size); ax1.set_ylim(0, y_max * 1.15)
    ax1.set_ylabel(y_axis_title, fontsize=16, fontweight='bold')
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
        else: handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=SINGLE_BAR_COLOR, markersize=12, label='Value'))
    if show_line:
        if line_cat_col != 'None': handles += [Line2D([0], [0], color=SPLIT_LINE_PALETTE[i % len(SPLIT_LINE_PALETTE)], marker='o', label=f"{c.replace('line_split_', '')} ({line_mode})") for i, c in enumerate(line_cols)]
        else: handles.append(Line2D([0], [0], color=DEFAULT_LINE_COLOR, marker='o', label=f"{'Deals' if line_mode == 'Count' else 'Value'}"))
    if handles: ax1.legend(handles=handles, loc='upper left', frameon=False, prop={'size': 14}, ncol=2)
    plt.title(title, fontsize=22, fontweight='bold', pad=30); return fig

# --- APP HEADER AREA ---
st.image("https://github.com/justinxtsui/Time-Series-Chart/blob/main/Screenshot%202026-02-06%20at%2016.51.25.png?raw=true", width=250) 
st.markdown('<div class="app-title">Line-us (˶ > ₃ < ˶) & Bar-tholomew (≖_≖ ) </div>', unsafe_allow_html=True)
st.markdown('<div class="app-attribution">by JT</div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Create any line or bar chart or both(Do not share the bot externally ⚠️)</p>', unsafe_allow_html=True)
st.markdown('<hr class="bold-divider">', unsafe_allow_html=True)

# --- SIDEBAR REORGANIZATION ---
with st.sidebar:
    # 1. UPLOAD DATA
    st.header("1. Upload Data")
    file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'xls', 'csv'])
    sheet = None
    if file and file.name.endswith(('.xlsx', '.xls')):
        xl = pd.ExcelFile(file)
        sheet = st.selectbox("Select Sheet", xl.sheet_names)
    
    if file:
        df_base = load_data(file, sheet_name=sheet)
        
        # 2. SELECT DATA TO ANALYSIS
        st.header("2. Select data to analysis")
        date_col = st.selectbox("Select Time Column", df_base.columns)
        granularity = st.radio("Time Period Type", ['Yearly', 'Quarterly'])
        
        try:
            temp_dates = pd.to_datetime(df_base[date_col].astype(str), format='mixed', errors='coerce').dropna()
            min_y, max_y = int(temp_dates.dt.year.min()), int(temp_dates.dt.year.max())
            year_range = st.slider("Select Year Range", min_y, max_y, (min_y, max_y))
            s_y, e_y = year_range
        except:
            st.info("Please ensure the date column is valid.")
            st.stop()

        # 3. SELECT VALUE TO PLOT
        st.header("3. Select value to plot")
        show_bars = st.checkbox("Show Bars", True)
        show_line = st.checkbox("Show Line", True)
        
        # Logic to pick columns based on user selection
        value_col = st.selectbox("Select column to plot as Bar/Line value", df_base.columns)
        
        line_mode = 'Count'
        if show_line:
            line_mode = st.radio("Line Metric Mode", ['Count', 'Value'])

        # 4. SPLIT BY CATEGORY
        st.header("4. Categorization (Split)")
        stack_col, colors, order = ('None', {}, {})
        line_cat = 'None'
        
        if st.checkbox("Enable Split by Category Column"):
            cat_target = st.multiselect("Apply split to:", ["Bars", "Line"])
            split_col = st.selectbox("Category Column to split by", [c for c in df_base.columns if c not in [date_col, value_col]])
            
            if "Bars" in cat_target:
                stack_col = split_col
                unique_cats = sorted([str(c) for c in df_base[stack_col].unique() if pd.notna(c)])
                if unique_cats:
                    st.write("Sort/Rearrange categories:")
                    sorted_cats = sort_items(unique_cats, key='sort_bars_new')
                    order = {c: i for i,c in enumerate(sorted_cats)}
                    # Colors handled in Labels section for unified key
            
            if "Line" in cat_target:
                line_cat = split_col

        # 5. LABEL & KEY
        st.header("5. Labels & Colour Key")
        title = st.text_input("Main Chart Title", DEFAULT_TITLE)
        y_axis_title = st.text_input("Y Axis Title", "Value")
        
        # Color mapping logic
        if stack_col != 'None':
            st.write("**Colour Key for Bars:**")
            cats_for_color = sorted([str(c) for c in df_base[stack_col].unique() if pd.notna(c)])
            colors = {c: st.selectbox(f"Color for {c}", list(PREDEFINED_COLORS.values()), index=i%6, key=f"col_{c}") for i, c in enumerate(cats_for_color)}

        # 6. EXPORT
        st.header("6. Export")
        export_format = st.selectbox("Format", options=['PNG', 'SVG (Vectorized)'])
        buf_p, buf_s = BytesIO(), BytesIO()

# --- MAIN LOGIC & RENDERING ---
if file and 'df_base' in locals() and df_base is not None:
    # Handle simple filtering if needed or just use current configs
    df_f = df_base.copy() 
    final, err_p = process_data(df_f, date_col, value_col, (s_y, e_y), stack_col, line_cat, granularity, line_mode)
    
    if final is not None:
        fig = generate_chart(final, value_col, stack_col, show_bars, show_line, title, y_axis_title, colors, order, None, line_cat, granularity, line_mode)
        st.pyplot(fig)
        
        fig.savefig(buf_p, format='png', dpi=300, bbox_inches='tight')
        fig.savefig(buf_s, format='svg', bbox_inches='tight')
        
        if export_format == 'PNG':
            st.sidebar.download_button("Download PNG", buf_p.getvalue(), "chart.png", use_container_width=True)
        else:
            st.sidebar.download_button("Download SVG", buf_s.getvalue(), "chart.svg", use_container_width=True)
else: 
    st.info("⬆️ Please upload your data file in the sidebar to begin.")

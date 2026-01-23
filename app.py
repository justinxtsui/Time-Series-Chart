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
# Recognizes common date and value variations
ALT_DATE_COLUMNS = ['Date the participant received the grant', 'Deal year', 'Date']
ALT_VALUE_COLUMNS = ['Amount received (converted to GBP)', 'Average pre-money valuation (GBP)', 'Amount']

# --- COLORS ---
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
LIGHT_PURPLE = '#BBBAF6'
WHITE_PURPLE = '#EAEAFF'
BLACK_PURPLE = '#211E52'
YELLOW = '#FFB914' 

CATEGORY_COLORS = [PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE] 
SPLIT_LINE_PALETTE = [PURPLE, DARK_PURPLE, BLACK_PURPLE, YELLOW]

PREDEFINED_COLORS = {
    'Purple': PURPLE, 'Dark Purple': DARK_PURPLE, 'Light Purple': LIGHT_PURPLE,
    'White Purple': WHITE_PURPLE, 'Black Purple': BLACK_PURPLE, 'Yellow': YELLOW
}

SINGLE_BAR_COLOR = '#BBBAF6'
PREDICTION_SHADE_COLOR = WHITE_PURPLE 
DEFAULT_LINE_COLOR = '#000000' 
DEFAULT_TITLE = 'Fundraising and Deal Count Over Time'

st.set_page_config(page_title="Time Series Chart Generator", layout="wide", initial_sidebar_state="expanded")

# --- HELPER FUNCTIONS ---

def format_currency(value):
    """Formats numeric values into currency with units (k, m, b)"""
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
    if float(s).is_integer(): s = str(int(float(s)))
    return f"{'-' if neg else ''}£{s}{unit}"

def is_dark_color(hex_color):
    """Checks if a color is dark to determine text contrast"""
    try:
        r, g, b = to_rgb(hex_color)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5
    except: return False

@st.cache_data
def load_data(uploaded_file):
    """Loads and cleans the uploaded data file"""
    if uploaded_file.name.endswith('.csv'): data = pd.read_csv(uploaded_file)
    else: data = pd.read_excel(uploaded_file, sheet_name=0)
    
    data.columns = data.columns.str.strip()
    
    # Identify Date and Value columns automatically
    if DATE_COLUMN not in data.columns:
        for alt in ALT_DATE_COLUMNS:
            if alt in data.columns:
                data.rename(columns={alt: DATE_COLUMN}, inplace=True)
                break
    
    if VALUE_COLUMN not in data.columns:
        for alt in ALT_VALUE_COLUMNS:
            if alt in data.columns:
                data.rename(columns={alt: VALUE_COLUMN}, inplace=True)
                break

    if DATE_COLUMN not in data.columns or VALUE_COLUMN not in data.columns:
        return None, "Required columns (Date and Value) not found.", None

    try:
        # THE 1970 FIX: Convert date to string before parsing
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN].astype(str), format='mixed', errors='coerce')
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce').fillna(0)
        for col in data.columns:
            if col not in [DATE_COLUMN, VALUE_COLUMN]:
                data[col] = data[col].astype(str).str.strip() 
    except Exception as e:
        return None, f"Processing Error: {e}", None
    
    return data, None, 'raised'

def generate_chart(final_data, category_column, show_bars, show_line, chart_title, category_colors, line_mode, granularity, line_category_column):
    """Main plotting function"""
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_pos = np.arange(len(final_data))
    time_labels = final_data['time_period'].values
    
    num_bars = len(final_data)
    DYNAMIC_FONT_SIZE = int(max(8, min(20, 150 / num_bars))) if num_bars > 0 else 12

    # Bar Logic
    category_cols = [c for c in final_data.columns if not str(c).startswith('line_split_') and c not in ['time_period', 'line_metric']]
    y_max = final_data[category_cols].sum(axis=1).max() if category_cols else (final_data[VALUE_COLUMN].max() if VALUE_COLUMN in final_data.columns else 1)

    if category_column != 'None' and category_cols:
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            color = category_colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
            ax1.bar(x_pos, final_data[cat], 0.8, bottom=bottom, color=color, linewidth=0)
            for i in range(len(final_data)):
                val = final_data[cat].iloc[i]
                if val > 0:
                    y_text = bottom[i] + (val / 2)
                    ax1.text(x_pos[i], y_text, format_currency(val), ha='center', va='center', 
                             fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color='white' if is_dark_color(color) else 'black')
            bottom += final_data[cat].values
    elif show_bars:
        ax1.bar(x_pos, final_data[VALUE_COLUMN], 0.8, color=SINGLE_BAR_COLOR, linewidth=0)
        for i, val in enumerate(final_data[VALUE_COLUMN]):
            if val > 0:
                ax1.text(x_pos[i], val * 0.01, format_currency(val), ha='center', va='bottom', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(time_labels, fontsize=DYNAMIC_FONT_SIZE)
    ax1.set_ylim(0, y_max * 1.2)
    for s in ax1.spines.values(): s.set_visible(False)
    ax1.tick_params(left=False, labelleft=False)

    # Line Logic
    if show_line:
        ax2 = ax1.twinx()
        line_cols = [c for c in final_data.columns if c.startswith('line_split_')] if line_category_column != 'None' else ['line_metric']
        for idx, l_col in enumerate(line_cols):
            l_color = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_category_column != 'None' else DEFAULT_LINE_COLOR
            ax2.plot(x_pos, final_data[l_col], color=l_color, marker='o', linewidth=3, markersize=10)
            for i, y in enumerate(final_data[l_col]):
                label = format_currency(y) if line_mode == 'Value' else str(int(y))
                ax2.text(x_pos[i], y + (final_data[line_cols].values.max() * 0.05), label, ha='center', color=l_color, fontweight='bold', fontsize=DYNAMIC_FONT_SIZE)
        ax2.axis('off')
        ax2.set_ylim(0, final_data[line_cols].values.max() * 1.6 if final_data[line_cols].values.max() > 0 else 1)

    plt.title(chart_title, fontsize=22, fontweight='bold', pad=30)
    return fig

# --- APPLICATION FLOW ---
st.markdown(f'<h1 style="color:{BLACK_PURPLE};">Time Series Chart Generator</h1>', unsafe_allow_html=True)

# Main UI split: Controls on Sidebar, Display in Main Area
st.sidebar.header("1. Data Source")
file = st.sidebar.file_uploader("Upload CSV/Excel", type=['xlsx', 'xls', 'csv'])

if file:
    df_base, err, _ = load_data(file)
    if df_base is not None:
        # Sidebar Controls
        title = st.sidebar.text_input("Chart Title", DEFAULT_TITLE)
        granularity = st.sidebar.radio("Time Granularity", ['Yearly', 'Quarterly'])
        
        c_v1, c_v2 = st.sidebar.columns(2)
        show_bars = c_v1.checkbox("Show Bars", True)
        show_line = c_v2.checkbox("Show Line", True)
        
        line_mode, line_cat = 'Count', 'None'
        if show_line:
            line_mode = st.sidebar.radio("Line Mode", ['Count', 'Value'])
            line_cat = st.sidebar.selectbox("Split Line Category", ['None'] + [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
            
        stack_col, sorted_cats, colors = 'None', [], {}
        if st.sidebar.checkbox("Stack Bars?"):
            stack_col = st.sidebar.selectbox("Stack Category", [c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
            unique_cats = sorted(df_base[stack_col].unique())
            sorted_cats = sort_items(unique_cats, key='sort_bars')
            colors = {c: st.sidebar.selectbox(f"Color: {c}", list(PREDEFINED_COLORS.values()), index=i % 6) for i, c in enumerate(sorted_cats)}

        # Data Processing
        df_base['time_period'] = df_base[DATE_COLUMN].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else df_base[DATE_COLUMN].dt.year
        df_base = df_base.sort_values(DATE_COLUMN)
        
        if stack_col != 'None':
            final = df_base.groupby(['time_period', stack_col])[VALUE_COLUMN].sum().unstack().fillna(0).reset_index()
        else:
            final = df_base.groupby('time_period')[VALUE_COLUMN].sum().reset_index()

        if line_cat != 'None':
            l_metric = df_base.groupby(['time_period', line_cat])[VALUE_COLUMN].sum().unstack() if line_mode == 'Value' else df_base.groupby(['time_period', line_cat]).size().unstack()
            l_metric.columns = [f"line_split_{c}" for c in l_metric.columns]
            final = final.merge(l_metric.fillna(0), on='time_period', how='left')
        else:
            final['line_metric'] = df_base.groupby('time_period')[VALUE_COLUMN].sum().values if line_mode == 'Value' else df_base.groupby('time_period').size().values

        # --- CENTER-LEFT DISPLAY ---
        col_chart, col_spacer = st.columns([4, 1]) # Allocates more space to the chart on the left
        
        with col_chart:
            fig = generate_chart(final, stack_col, show_bars, show_line, title, colors, line_mode, granularity, line_cat)
            st.pyplot(fig)
        
        # Download options kept in the sidebar for clean UI
        buf_p = BytesIO()
        fig.savefig(buf_p, format='png', dpi=300)
        st.sidebar.download_button("Download PNG", buf_p.getvalue(), "chart.png", "image/png")
        
        buf_s = BytesIO()
        fig.savefig(buf_s, format='svg')
        st.sidebar.download_button("Download Adobe SVG", buf_s.getvalue(), "chart.svg", "image/svg+xml")
        
    else: st.error(err)
else: st.info("Please upload a file to begin.")

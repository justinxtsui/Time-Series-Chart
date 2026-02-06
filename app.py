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

def process_data(df, date_col, bar_val_col, line_val_col, year_range, cat_col, line_cat_col, granularity):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='mixed', errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    
    # Handle "Row Count" logic
    if bar_val_col == "Row Count":
        df["bar_internal_val"] = 1
        bar_col_to_use = "bar_internal_val"
    else:
        df[bar_val_col] = pd.to_numeric(df[bar_val_col], errors='coerce').fillna(0)
        bar_col_to_use = bar_val_col

    if line_val_col == "Row Count":
        df["line_internal_val"] = 1
        line_col_to_use = "line_internal_val"
    else:
        df[line_val_col] = pd.to_numeric(df[line_val_col], errors='coerce').fillna(0)
        line_col_to_use = line_val_col
    
    chart_data = df[df[date_col].dt.year.between(year_range[0], year_range[1], inclusive='both')].copy()
    if chart_data.empty: return None, "No data available."
    
    chart_data['time_period'] = chart_data[date_col].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else chart_data[date_col].dt.year
    chart_data = chart_data.sort_values(date_col)
    
    # 1. Process Bars
    if cat_col != 'None':
        grouped_bars = chart_data.groupby(['time_period', cat_col]).agg({bar_col_to_use: 'sum'}).reset_index()
        final_data = grouped_bars.pivot(index='time_period', columns=cat_col, values=bar_col_to_use).fillna(0).reset_index()
    else: 
        final_data = chart_data.groupby('time_period').agg({bar_col_to_use: 'sum'}).reset_index()
        final_data.rename(columns={bar_col_to_use: 'bar_total'}, inplace=True)

    # 2. Process Lines
    if line_cat_col != 'None':
        line_grouped = chart_data.groupby(['time_period', line_cat_col])[line_col_to_use].sum().reset_index()
        line_pivot = line_grouped.pivot(index='time_period', columns=line_cat_col, values=line_col_to_use).fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        line_metric = chart_data.groupby('time_period')[line_col_to_use].sum().reset_index(name='line_metric')
        final_data = final_data.merge(line_metric, on='time_period', how='left').fillna(0)
        
    return final_data, None

def generate_chart(final_data, bar_val_col, cat_col, show_bars, show_line, title, y_axis_title, colors, order, pred_y, line_cat_col, granularity):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_pos, time_labels = np.arange(len(final_data)), final_data['time_period'].values
    is_pred = (time_labels.astype(int) >= pred_y) if granularity == 'Yearly' and pred_y else np.full(len(time_labels), False)
    font_size = int(max(8, min(22, 150 / len(final_data))))
    
    # Determine bar columns
    bar_cols = [c for c in final_data.columns if not str(c).startswith('line_split_') and c not in ['time_period', 'line_metric']]
    if order: bar_cols.sort(key=lambda x: order.get(x, 999))
    y_max = final_data[bar_cols].sum(axis=1).max() if bar_cols else 1

    if show_bars:
        if cat_col != 'None':
            bottom = np.zeros(len(final_data))
            for idx, cat in enumerate(bar_cols):
                c = colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
                for i in range(len(final_data)):
                    val = final_data[cat].iloc[i]
                    if val > 0:
                        bc = PREDICTION_SHADE_COLOR if is_pred[i] else c
                        ax1.bar(x_pos[i], val, 0.8, bottom=bottom[i], color=bc, hatch='xx' if is_pred[i] else None, edgecolor='black' if is_pred[i] else 'none', linewidth=0)
                        label_text = str(int(val)) if bar_val_col == "Row Count" else format_currency(val)
                        ax1.text(x_pos[i], (bottom[i] + y_max*0.01) if idx == 0 else (bottom[i] + val/2), label_text, ha='center', va='bottom' if idx == 0 else 'center', fontsize=font_size, fontweight='bold', color='#FFFFFF' if is_dark_color(bc) else '#000000')
                    bottom[i] += val
        else:
            for i in range(len(final_data)):
                val, bc = final_data['bar_total'].iloc[i], (PREDICTION_SHADE_COLOR if is_pred[i] else SINGLE_BAR_COLOR)
                ax1.bar(x_pos[i], val, 0.8, color=bc, hatch='xx' if is_pred[i] else None, edgecolor='black' if is_pred[i] else 'none', linewidth=0)
                if val > 0:
                    label_text = str(int(val)) if bar_val_col == "Row Count" else format_currency(val)
                    ax1.text(x_pos[i], y_max*0.01, label_text, ha='center', va='bottom', fontsize=font_size, fontweight='bold', color='#000000')

    ax1.set_xticks(x_pos); ax1.set_xticklabels(time_labels, fontsize=font_size); ax1.set_ylim(0, y_max * 1.15)
    ax1.set_ylabel(y_axis_title, fontsize=16, fontweight='bold')
    ax1.tick_params(left=False, labelleft=False, length=0); [s.set_visible(False) for s in ax1.spines.values()]

    if show_line:
        ax2 = ax1.twinx()
        line_cols = [c for c in final_data.columns if c.startswith('line_split_')] if line_cat_col != 'None' else ['line_metric']
        l_max = final_data[line_cols].values.max() if final_data[line_cols].values.size > 0 else 1
        for idx, l_col in enumerate(line_cols):
            lc = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_cat_col != 'None' else DEFAULT_LINE_COLOR
            ax2.plot(x_pos, final_data[l_col].values, color=lc, marker='o', linestyle='-', linewidth=2.5, markersize=8)
            for i, y in enumerate(final_data[l_col].values):
                label_text = str(int(y)) # Lines usually represent counts in this UI logic
                ax2.text(x_pos[i], y + l_max*0.05, label_text, ha='center', va='bottom', fontsize=font_size, color=lc, fontweight='bold')
        ax2.axis('off'); ax2.set_ylim(0, l_max * 1.6)

    handles = []
    if show_bars:
        if cat_col != 'None': handles += [Line2D([0], [0], marker='s', color='w', markerfacecolor=colors.get(c, PURPLE), markersize=12, label=c) for c in bar_cols]
        else: handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=SINGLE_BAR_COLOR, markersize=12, label='Value'))
    if show_line:
        if line_cat_col != 'None': handles += [Line2D([0], [0], color=SPLIT_LINE_PALETTE[i % len(SPLIT_LINE_PALETTE)], marker='o', label=f"{c.replace('line_split_', '')}") for i, c in enumerate(line_cols)]
        else: handles.append(Line2D([0], [0], color=DEFAULT_LINE_COLOR, marker='o', label='Line Metric'))
    if handles: ax1.legend(handles=handles, loc='upper left', frameon=False, prop={'size': 14}, ncol=2)
    plt.title(title, fontsize=22, fontweight='bold', pad=30); return fig

# --- APP HEADER AREA ---
st.image("https://github.com/justinxtsui/Time-Series-Chart/blob/main/Screenshot%202026-02-06%20at%2016.51.25.png?raw=true", width=250) 
st.markdown('<div class="app-title">Line-us (˶ > ₃ < ˶) & Bar-tholomew (≖_≖ ) </div>', unsafe_allow_html=True)
st.markdown('<div class="app-attribution">by JT</div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Create any line or bar chart or both(Do not share the bot externally ⚠️)</p>', unsafe_allow_html=True)
st.markdown('<hr class="bold-divider">', unsafe_allow_html=True)

# --- SIDEBAR LOGIC FLOW ---
if 'generate_clicked' not in st.session_state:
    st.session_state.generate_clicked = False

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
        
        configs = []
        if st.checkbox("Filter any data?"):
            sel_f = st.multiselect("Pick column to filter", sorted([c for c in df_base.columns]))
            for f in sel_f:
                with st.expander(f"Filter settings: {f}", expanded=True):
                    vals = st.multiselect(f"Pick categories", df_base[f].unique(), key=f"f_v_{f}")
                    mode = st.radio(f"Mode for {f}", ["Include", "Exclude"], key=f"f_m_{f}")
                    configs.append({'column': f, 'values': vals, 'include': mode == "Include"})

        # 2. SELECT DATA TO ANALYSIS
        st.header("2. Select data to analysis")
        date_col = st.selectbox("Select Time Column", options=[None] + list(df_base.columns), index=0)
        granularity = st.radio("Time Period Type", ['Yearly', 'Quarterly'])
        
        if date_col:
            temp_dates = pd.to_datetime(df_base[date_col].astype(str), format='mixed', errors='coerce').dropna()
            min_y, max_y = int(temp_dates.dt.year.min()), int(temp_dates.dt.year.max())
            col_s, col_e = st.columns(2)
            with col_s: s_y = st.selectbox("Start Year", list(range(min_y, max_y + 1)), index=0)
            with col_e: e_y = st.selectbox("End Year", list(range(min_y, max_y + 1)), index=(max_y-min_y))
            if s_y > e_y: st.warning("Start > End"); st.stop()
        else: st.stop()

        # 3. SELECT VALUE TO PLOT
        st.header("3. Select value to plot")
        show_bars = st.checkbox("Show Bars", True)
        bar_val_col = None
        if show_bars:
            bar_val_col = st.selectbox("Select column for Bars", options=[None, "Row Count"] + list(df_base.columns), index=0)

        show_line = st.checkbox("Show Line", True)
        line_val_col = None
        if show_line:
            line_val_col = st.selectbox("Select column for Line", options=[None, "Row Count"] + list(df_base.columns), index=0)

        if (show_bars and not bar_val_col) or (show_line and not line_val_col):
            st.stop()

        # 4. SPLIT LINES OR STACKED BAR?
        st.header("4. Split lines or stacked bar?")
        stack_col, colors, order = ('None', {}, {})
        line_cat = 'None'
        
        if st.checkbox("Enable Split/Stack Settings"):
            cat_target = st.multiselect("Apply split to:", ["Bars", "Line"])
            split_col = st.selectbox("Pick column to split by", options=[None] + [c for c in df_base.columns if c not in [date_col, bar_val_col, line_val_col]], index=0)
            
            if split_col:
                if "Bars" in cat_target:
                    stack_col = split_col
                    unique_cats = sorted([str(c) for c in df_base[stack_col].unique() if pd.notna(c)])
                    if unique_cats:
                        st.write("Rearrange categories for Stacked Bars:")
                        sorted_cats = sort_items(unique_cats, key='sort_bars_new')
                        order = {c: i for i,c in enumerate(sorted_cats)}
                if "Line" in cat_target: line_cat = split_col
                if st.button("Generate"): st.session_state.generate_clicked = True
            else: st.session_state.generate_clicked = False
        else: st.session_state.generate_clicked = True

        # 5. LABEL & KEY
        st.header("5. Labels & Colour Key")
        title = st.text_input("Main Chart Title", DEFAULT_TITLE)
        y_axis_title = st.text_input("Y Axis Title", "Value")
        if stack_col != 'None':
            st.write("**Colour Key for Bars:**")
            cats_for_color = sorted([str(c) for c in df_base[stack_col].unique() if pd.notna(c)])
            colors = {c: st.selectbox(f"Color for {c}", list(PREDEFINED_COLORS.values()), index=i%6, key=f"col_{c}") for i, c in enumerate(cats_for_color)}

        # 6. EXPORT
        st.header("6. Export")
        export_format = st.selectbox("Format", options=['PNG', 'SVG (Vectorized)'])
        buf_p, buf_s = BytesIO(), BytesIO()

# --- MAIN LOGIC & RENDERING ---
if file and st.session_state.generate_clicked:
    df_f = apply_filter(df_base, configs)
    final, err_p = process_data(df_f, date_col, bar_val_col, line_val_col, (s_y, e_y), stack_col, line_cat, granularity)
    
    if final is not None:
        fig = generate_chart(final, bar_val_col, stack_col, show_bars, show_line, title, y_axis_title, colors, order, None, line_cat, granularity)
        st.pyplot(fig)
        
        fig.savefig(buf_p, format='png', dpi=300, bbox_inches='tight')
        fig.savefig(buf_s, format='svg', bbox_inches='tight')
        
        btn_label = f"Download {export_format}"
        if export_format == 'PNG': st.sidebar.download_button(btn_label, buf_p.getvalue(), "chart.png", use_container_width=True)
        else: st.sidebar.download_button(btn_label, buf_s.getvalue(), "chart.svg", use_container_width=True)
elif file and not st.session_state.generate_clicked:
    st.info("Please configure your split settings and click 'Generate' to see the chart.")
else: 
    st.info("⬆️ Please upload your data file in the sidebar to begin.")

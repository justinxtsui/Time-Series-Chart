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
ALT_DATE_COLUMN = 'Date the participant received the grant'
ALT_VALUE_COLUMN = 'Amount received (converted to GBP)'

# --- USER COLOR PALETTE ---
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
LIGHT_PURPLE = '#BBBAF6'
WHITE_PURPLE = '#EAEAFF'
BLACK_PURPLE = '#211E52'
YELLOW = '#FFB914' 

# Mapping palette for categorical data (Bars)
CATEGORY_COLORS = [PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE] 
# New Palette for Split Lines (Yellow prioritized)
SPLIT_LINE_PALETTE = [YELLOW, PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE]

PREDEFINED_COLORS = {
    'Purple': PURPLE,
    'Dark Purple': DARK_PURPLE,
    'Light Purple': LIGHT_PURPLE,
    'White Purple': WHITE_PURPLE,
    'Black Purple': BLACK_PURPLE,
    'Yellow': YELLOW
}

SINGLE_BAR_COLOR = '#BBBAF6' # Original default
PREDICTION_SHADE_COLOR = WHITE_PURPLE 
PREDICTION_HATCH_COLOR = '#000000'
DEFAULT_LINE_COLOR = '#000000' # Original default black
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
        unit = "b"
        divisor = 1e9
    elif x_abs >= 1e6:
        unit = "m"
        divisor = 1e6
    elif x_abs >= 1e3:
        unit = "k"
        divisor = 1e3
    else:
        unit = ""
        divisor = 1.0

    scaled = x_abs / divisor
    s = f"{scaled:.3g}"
    
    try:
        if float(s).is_integer():
            s = str(int(float(s)))
    except:
        pass

    sign = "-" if neg else ""
    return f"{sign}£{s}{unit}"

def is_dark_color(hex_color):
    try:
        r, g, b = to_rgb(hex_color)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
        return luminance < 0.5
    except ValueError:
        return False

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name=0)
    
    data.columns = data.columns.str.strip()
    original_value_column = None
    
    if DATE_COLUMN not in data.columns:
        if ALT_DATE_COLUMN in data.columns:
            data.rename(columns={ALT_DATE_COLUMN: DATE_COLUMN}, inplace=True)
        else:
            return None, f"File must contain a date column named **`{DATE_COLUMN}`** or **`{ALT_DATE_COLUMN}`**.", None

    if VALUE_COLUMN not in data.columns:
        if ALT_VALUE_COLUMN in data.columns:
            original_value_column = 'received' 
            data.rename(columns={ALT_VALUE_COLUMN: VALUE_COLUMN}, inplace=True)
        else:
            return None, f"File must contain a value column named **`{VALUE_COLUMN}`** or **`{ALT_VALUE_COLUMN}`**.", None
    else:
        original_value_column = 'raised'

    try:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], format='mixed', errors='coerce')
        for col in data.columns:
            if col not in [DATE_COLUMN, VALUE_COLUMN]:
                data[col] = data[col].astype(str).str.strip() 
        
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce')
        data[VALUE_COLUMN] = data[VALUE_COLUMN].fillna(0)

    except Exception as e:
        return None, f"An error occurred during data conversion: {e}", None
    
    if data.empty:
        return None, "File loaded but contained no valid rows after processing.", None

    return data, None, original_value_column

@st.cache_data
def apply_filter(df, filter_config):
    if not filter_config['enabled'] or filter_config['column'] == 'None':
        return df
    col = filter_config['column']
    values = filter_config['values']
    is_include = filter_config['include']
    if values:
        return df[df[col].isin(values)] if is_include else df[~df[col].isin(values)]
    return df

@st.cache_data
def process_data(df, year_range, category_column, line_category_column='None', granularity='Yearly'):
    df = df.copy()
    start_year, end_year = year_range
    chart_data = df[df[DATE_COLUMN].dt.year.between(start_year, end_year, inclusive='both')].copy()
    
    if chart_data.empty:
        return None, "No data available for the selected range."
    
    if granularity == 'Quarterly':
        chart_data['time_period'] = chart_data[DATE_COLUMN].dt.to_period('Q').astype(str)
        chart_data = chart_data.sort_values(DATE_COLUMN)
    else:
        chart_data['time_period'] = chart_data[DATE_COLUMN].dt.year
    
    if category_column != 'None':
        grouped = chart_data.groupby(['time_period', category_column]).agg({VALUE_COLUMN: 'sum'}).reset_index()
        final_data = grouped.pivot(index='time_period', columns=category_column, values=VALUE_COLUMN).fillna(0).reset_index()
    else:
        final_data = chart_data.groupby('time_period').agg({VALUE_COLUMN: 'sum'}).reset_index()

    if line_category_column != 'None':
        line_grouped = chart_data.groupby(['time_period', line_category_column]).size().reset_index(name='count')
        line_pivot = line_grouped.pivot(index='time_period', columns=line_category_column, values='count').fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
        final_data = final_data.merge(row_counts, on='time_period', how='left').fillna(0)
    
    return final_data, None


def generate_chart(final_data, category_column, show_bars, show_line, chart_title, 
                   original_value_column='raised', category_colors=None, category_order=None, 
                   prediction_start_year=None, line_category_column='None', granularity='Yearly'):
    
    chart_fig, chart_ax1 = plt.subplots(figsize=(20, 10))
    bar_width = 0.8
    x_pos = np.arange(len(final_data))
    time_labels = final_data['time_period'].values
    
    if granularity == 'Yearly' and prediction_start_year is not None:
        is_predicted = (time_labels.astype(int) >= prediction_start_year)
    else:
        is_predicted = np.full(len(time_labels), False)

    bar_legend_label = 'Total amount received' if original_value_column == 'received' else 'Amount raised'
    num_bars = len(final_data)
    DYNAMIC_FONT_SIZE = int(max(8, min(22, 150 / num_bars))) if num_bars > 0 else 12

    category_cols = []
    if category_column != 'None':
        category_cols = [col for col in final_data.columns if not str(col).startswith('line_split_') and col not in ['time_period', 'row_count']]
        if category_order:
            category_cols.sort(key=lambda x: category_order.get(x, 999))

    y_max = final_data[category_cols].sum(axis=1).max() if category_column != 'None' else final_data[VALUE_COLUMN].max()
    vertical_offset = y_max * 0.01
    
    # --- BAR CHART ---
    if category_column != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            color = category_colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
            for i in range(len(final_data)):
                val = final_data[cat].iloc[i]
                if show_bars and val > 0:
                    bar_color = PREDICTION_SHADE_COLOR if is_predicted[i] else color
                    h_style = 'xx' if is_predicted[i] else None
                    e_color = PREDICTION_HATCH_COLOR if is_predicted[i] else 'none'
                    
                    chart_ax1.bar(x_pos[i], val, bar_width, bottom=bottom[i], color=bar_color, 
                                  hatch=h_style, edgecolor=e_color, linewidth=0)
                    
                    text_color = '#FFFFFF' if is_dark_color(bar_color) else '#000000'
                    y_text = (bottom[i] + vertical_offset) if idx == 0 else (bottom[i] + val / 2)
                    chart_ax1.text(x_pos[i], y_text, format_currency(val), ha='center', 
                                   va='bottom' if idx == 0 else 'center', fontsize=DYNAMIC_FONT_SIZE, 
                                   fontweight='bold', color=text_color)
                bottom[i] += val
    else:
        if show_bars:
            for i in range(len(final_data)):
                val = final_data[VALUE_COLUMN].iloc[i]
                bar_color = PREDICTION_SHADE_COLOR if is_predicted[i] else SINGLE_BAR_COLOR
                chart_ax1.bar(x_pos[i], val, bar_width, color=bar_color, 
                              hatch='xx' if is_predicted[i] else None, 
                              edgecolor=PREDICTION_HATCH_COLOR if is_predicted[i] else 'none', linewidth=0)
                if val > 0:
                    chart_ax1.text(x_pos[i], vertical_offset, format_currency(val), ha='center', 
                                   va='bottom', fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color='#000000')

    chart_ax1.set_xticks(x_pos)
    chart_ax1.set_xticklabels(time_labels, fontsize=DYNAMIC_FONT_SIZE)
    chart_ax1.set_ylim(0, y_max * 1.1)
    chart_ax1.tick_params(axis='both', which='both', length=0, labelleft=False)
    for spine in chart_ax1.spines.values(): spine.set_visible(False)

    # --- LINE CHART ---
    if show_line:
        chart_ax2 = chart_ax1.twinx()
        line_cols = [col for col in final_data.columns if col.startswith('line_split_')] if line_category_column != 'None' else ['row_count']
        
        for idx, l_col in enumerate(line_cols):
            display_name = l_col.replace('line_split_', '') if line_category_column != 'None' else 'Number of deals'
            
            # Use original black for single line, use NEW PALETTE for split line
            if line_category_column != 'None':
                l_color = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)]
            else:
                l_color = DEFAULT_LINE_COLOR
            
            y_vals = final_data[l_col].values
            
            actual_mask = ~is_predicted
            if any(actual_mask):
                chart_ax2.plot(x_pos[actual_mask], y_vals[actual_mask], color=l_color, marker='o', 
                               linestyle='-', linewidth=2.5, markersize=8, label=display_name)
            
            if any(is_predicted):
                p_idx = np.where(is_predicted)[0]
                a_idx = np.where(~is_predicted)[0]
                if len(a_idx) > 0:
                    conn_x = np.concatenate(([x_pos[a_idx[-1]]], x_pos[p_idx]))
                    conn_y = np.concatenate(([y_vals[a_idx[-1]]], y_vals[p_idx]))
                else:
                    conn_x, conn_y = x_pos[p_idx], y_vals[p_idx]
                chart_ax2.plot(conn_x, conn_y, color=l_color, marker='o', linestyle='--', linewidth=2.5, markersize=8)

            y_range = y_vals.max() * 0.05 if y_vals.max() > 0 else 1
            for i, y in enumerate(y_vals):
                chart_ax2.text(x_pos[i], y + y_range, str(int(y)), ha='center', va='bottom', 
                               fontsize=DYNAMIC_FONT_SIZE, color=l_color, fontweight='bold')

        chart_ax2.set_ylim(0, final_data[line_cols].values.max() * 1.5 if final_data[line_cols].values.max() > 0 else 10)
        chart_ax2.axis('off')

    # --- LEGEND ---
    legend_elements = []
    if show_bars:
        if category_column != 'None':
            for idx, cat in enumerate(category_cols):
                c = category_colors.get(cat, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
                legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=12, label=cat))
        else:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=SINGLE_BAR_COLOR, markersize=12, label=bar_legend_label))
    
    if show_line:
        if line_category_column != 'None':
            for idx, l_col in enumerate(line_cols):
                display_name = l_col.replace('line_split_', '')
                l_c = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)]
                legend_elements.append(Line2D([0], [0], color=l_c, marker='o', label=f"{display_name} (Deals)"))
        else:
            legend_elements.append(Line2D([0], [0], color=DEFAULT_LINE_COLOR, marker='o', label='Number of deals'))

    chart_ax1.legend(handles=legend_elements, loc='upper left', frameon=False, prop={'size': 14}, ncol=2)
    plt.title(chart_title, fontsize=18, fontweight='bold', pad=20, color=TITLE_COLOR)
    plt.tight_layout()
    return chart_fig

# --- STREAMLIT APP LAYOUT ---
st.markdown(f'<h1 style="color:{APP_TITLE_COLOR};">Time Series Chart Generator</h1>', unsafe_allow_html=True)
st.markdown(f"""<div style="background: {WHITE_PURPLE}; padding: 20px; border-radius: 10px; border-left: 5px solid {YELLOW}; margin: 15px 0;">
    <p style="margin: 0 0 10px 0; font-size: 16px; color: #000;"><strong>Turn any fundraising or grant export into a time series chart – JT</strong></p>
    <a href="https://platform.beauhurst.com/search/advancedsearch/" target="_blank" style="display: inline-block; background: #fff; padding: 10px 16px; border-radius: 6px; border: 1px solid #ddd; color: {PURPLE}; font-weight: 600; text-decoration: none; font-size: 14px;">🔗 Beauhurst Advanced Search</a>
    </div>""", unsafe_allow_html=True)

if 'buf_png' not in st.session_state:
    st.session_state.update({'year_range': (1900, 2100), 'category_column': 'None', 'line_category_column': 'None',
                             'show_bars': True, 'show_line': True, 'chart_title': DEFAULT_TITLE,
                             'buf_png': BytesIO(), 'buf_svg': BytesIO(), 'prediction_start_year': None,
                             'granularity': 'Yearly'})

with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'xls', 'csv'])

    df_base = None
    if uploaded_file:
        df_base, error_msg, orig_val_col = load_data(uploaded_file)
        if df_base is not None:
            st.session_state['original_value_column'] = orig_val_col
            min_y, max_y = int(df_base[DATE_COLUMN].dt.year.min()), int(df_base[DATE_COLUMN].dt.year.max())
            all_years = list(range(min_y, max_y + 1))
            
            st.header("2. Chart Title")
            st.session_state['chart_title'] = st.text_input("Title", value=st.session_state['chart_title'])
            
            st.header("3. Time Filters")
            st.session_state['granularity'] = st.radio("Time Granularity", ['Yearly', 'Quarterly'], index=0 if st.session_state['granularity']=='Yearly' else 1)
            c1, c2 = st.columns(2)
            start_year = c1.selectbox("Start Year", all_years, index=0)
            end_year = c2.selectbox("End Year", all_years, index=len(all_years)-1)
            st.session_state['year_range'] = (start_year, end_year)
            
            st.header("4. Visual Elements")
            st.session_state['show_bars'] = st.checkbox("Show value bars", value=True)
            st.session_state['show_line'] = st.checkbox("Show deal count line", value=True)
            
            if st.session_state['show_line']:
                line_cols = ['None'] + sorted([c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                st.session_state['line_category_column'] = st.selectbox("Split line by category", line_cols, index=0)
            
            if st.session_state['granularity'] == 'Yearly':
                st.subheader("Prediction Visuals")
                if st.checkbox("Enable prediction"):
                    pred_years = ['None'] + list(range(start_year, end_year + 1))
                    sel_pred = st.selectbox("Prediction Start Year", pred_years)
                    st.session_state['prediction_start_year'] = int(sel_pred) if sel_pred != 'None' else None

            st.header("5. Stacked Bar (Optional)")
            if st.checkbox('Enable Stacked Bar', value=False):
                cat_cols = ['None'] + sorted([c for c in df_base.columns if c not in [DATE_COLUMN, VALUE_COLUMN]])
                category_column = st.selectbox("Stack Column", cat_cols)
                st.session_state['category_column'] = category_column
                if category_column != 'None':
                    unique_cats = sorted(df_base[category_column].unique())
                    sorted_cats = sort_items(unique_cats, direction='vertical', key='sort_cats')
                    st.session_state['category_order'] = {c: i for i, c in enumerate(sorted_cats)}
                    st.session_state['category_colors'] = {c: st.selectbox(f"Color: {c}", list(PREDEFINED_COLORS.values()), index=i % len(PREDEFINED_COLORS)) for i, c in enumerate(sorted_cats)}
            else:
                st.session_state['category_column'] = 'None'

            st.header("6. Data Filter")
            if st.checkbox('Enable Filtering'):
                f_col = st.selectbox("Filter Col", ['None'] + sorted(df_base.columns))
                if f_col != 'None':
                    vals = st.multiselect("Values", df_base[f_col].unique())
                    st.session_state['filter_config'] = {'enabled': True, 'column': f_col, 'include': True, 'values': vals}
            else:
                st.session_state['filter_config'] = {'enabled': False, 'column': 'None', 'include': True, 'values': []}

            st.header("7. Download")
            st.download_button("Download PNG", st.session_state['buf_png'], "chart.png", "image/png", use_container_width=True)
        else:
            st.error(error_msg)

if df_base is not None:
    df_filtered = apply_filter(df_base, st.session_state.get('filter_config', {'enabled': False}))
    final_data, err = process_data(df_filtered, st.session_state['year_range'], 
                                   st.session_state['category_column'], 
                                   st.session_state['line_category_column'],
                                   st.session_state['granularity'])
    
    if final_data is not None:
        fig = generate_chart(final_data, st.session_state['category_column'], 
                             st.session_state['show_bars'], st.session_state['show_line'], 
                             st.session_state['chart_title'], st.session_state['original_value_column'],
                             st.session_state.get('category_colors', {}), 
                             st.session_state.get('category_order', {}),
                             st.session_state['prediction_start_year'],
                             st.session_state['line_category_column'],
                             st.session_state['granularity'])
        
        st.pyplot(fig, use_container_width=True)
        
        buf_p = BytesIO()
        fig.savefig(buf_p, format='png', dpi=300, bbox_inches='tight')
        st.session_state['buf_png'] = buf_p
else:
    st.info("⬆️ Please upload your data file in the sidebar to begin.")

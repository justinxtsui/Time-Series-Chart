import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

# --- CONFIGURATION ---
# Define required column names
DATE_COLUMN = 'Date the participant received the grant'
VALUE_COLUMN = 'Amount received (converted to GBP)'
# Define the color palette for categories
CATEGORY_COLORS = ['#302A7E', '#8884B3', '#D0CCE5', '#5C5799', '#B4B1CE', '#E0DEE9']
# Define the default single bar color (third color in the palette for a lighter tone)
SINGLE_BAR_COLOR = CATEGORY_COLORS[2] 
# Define the line chart color
LINE_COLOR = '#000000' # Black for high contrast

# Set page config and general styles
st.set_page_config(page_title="Dynamic Grant Chart Generator", layout="wide")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- HELPER FUNCTIONS ---

def format_currency(value):
    """
    Format a numeric value as money with £ and units (k, m, b),
    to 3 significant figures.
    """
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
    """Check if a hex color is dark. Returns True if dark, False if light."""
    try:
        r, g, b = to_rgb(hex_color)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
        return luminance < 0.5
    except ValueError:
        return False

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded file."""
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        # Load the first sheet
        data = pd.read_excel(uploaded_file, sheet_name=0)
        
    if DATE_COLUMN not in data.columns or VALUE_COLUMN not in data.columns:
        return None, f"File must contain columns: **`{DATE_COLUMN}`** and **`{VALUE_COLUMN}`**."

    try:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors='coerce')
        data.dropna(subset=[DATE_COLUMN], inplace=True)
    except Exception:
        return None, f"Could not convert **`{DATE_COLUMN}`** to datetime format."

    return data, None

@st.cache_data
def process_data(df, year_range, category_column):
    """Filters and aggregates the data for charting."""
    df = df.copy()
    start_year, end_year = year_range
    
    chart_data = df[df[DATE_COLUMN].dt.year.between(start_year, end_year, inclusive='both')].copy()
    
    if chart_data.empty:
        return None, "No data available for the selected year range."
    
    chart_data['time_period'] = chart_data[DATE_COLUMN].dt.year
    
    if category_column != 'None':
        grouped = chart_data.groupby(['time_period', category_column]).agg({
            VALUE_COLUMN: 'sum'
        }).reset_index()
        row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
        pivot_data = grouped.pivot(index='time_period', columns=category_column, values=VALUE_COLUMN).fillna(0)
        final_data = pivot_data.reset_index().merge(row_counts, on='time_period')
    else:
        grouped = chart_data.groupby('time_period').agg({
            VALUE_COLUMN: 'sum'
        }).reset_index()
        row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
        final_data = grouped.merge(row_counts, on='time_period')
    
    return final_data, None


def generate_chart(final_data, category_column, show_bars, show_line):
    """Generates the dual-axis Matplotlib chart."""
    chart_fig, chart_ax1 = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8
    x_pos = np.arange(len(final_data))
    dynamic_font_size = max(8, min(14, int(50 / len(final_data)) * 3))
    
    category_cols = []
    if category_column != 'None':
        category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]

    if category_column == 'None':
        y_max = final_data[VALUE_COLUMN].max()
    else:
        y_max = final_data[category_cols].sum(axis=1).max()

    vertical_offset = y_max * 0.01 
    
    # --- AXIS 1 (Bar Chart - Value) ---
    if category_column != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
            if show_bars:
                chart_ax1.bar(x_pos, final_data[cat], bar_width, bottom=bottom, 
                              label=cat, color=color, alpha=1.0)
            
            for i, x in enumerate(x_pos):
                val = final_data[cat].iloc[i]
                if val > 0 and show_bars:
                    label_text = format_currency(val)
                    current_color = color
                    text_color = '#FFFFFF' if is_dark_color(current_color) else '#000000'
                    
                    if idx == 0:
                        y_pos = vertical_offset
                        va = 'bottom'
                    else:
                        y_pos = bottom[i] + val / 2
                        va = 'center'
                        
                    chart_ax1.text(x, y_pos, label_text, ha='center', va=va,
                                   fontsize=dynamic_font_size, fontweight='bold', color=text_color)
            bottom += final_data[cat].values
    else:
        if show_bars:
            chart_ax1.bar(x_pos, final_data[VALUE_COLUMN], bar_width, 
                          label='Total Amount', color=SINGLE_BAR_COLOR, alpha=1.0)
        
            for i, x in enumerate(x_pos):
                val = final_data[VALUE_COLUMN].iloc[i]
                if val > 0:
                    label_text = format_currency(val)
                    text_color = '#FFFFFF' if is_dark_color(SINGLE_BAR_COLOR) else '#000000'
                    chart_ax1.text(x, vertical_offset, label_text, ha='center', va='bottom',
                                   fontsize=dynamic_font_size, fontweight='bold', color=text_color)
    
    chart_ax1.set_xticks(x_pos)
    chart_ax1.set_xticklabels(final_data['time_period'])
    
    plt.setp(chart_ax1.get_xticklabels(), fontsize=dynamic_font_size + 1, fontweight='normal')
    
    chart_ax1.set_ylim(0, y_max * 1.1)
    chart_ax1.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False, length=0)
    chart_ax1.tick_params(axis='x', bottom=False, length=0, pad=6)
    for spine in chart_ax1.spines.values():
        spine.set_visible(False)
    chart_ax1.grid(False)

    # --- AXIS 2 (Line Chart - Count) ---
    if show_line:
        chart_ax2 = chart_ax1.twinx()
        line_data = final_data['row_count']
        max_count = line_data.max()
        
        chart_ax2.plot(x_pos, line_data, color=LINE_COLOR, marker='o', linewidth=1.5, markersize=6, label='Number of Deals')
        
        chart_ax2.set_ylim(0, max_count * 1.5)
        chart_ax2.tick_params(axis='y', right=False, labelright=False, left=False, labelleft=False, length=0)
        for spine in chart_ax2.spines.values():
            spine.set_visible(False)
            
        y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
        base_offset = y_range * 0.015
        
        for i, (x, y) in enumerate(zip(x_pos, line_data)):
            place_below = (i % 2 == 0)
            va = 'top' if place_below else 'bottom'
            y_pos = y - base_offset if place_below else y + base_offset
            
            chart_ax2.text(x, y_pos, str(int(y)), ha='center', va=va, fontsize=dynamic_font_size, 
                           color=LINE_COLOR, fontweight='bold')
    
    # --- LEGEND & TITLE ---
    legend_elements = []
    
    if show_bars:
        if category_column != 'None':
            for idx, cat in enumerate(category_cols):
                color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=10, label=cat))
        else:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=SINGLE_BAR_COLOR, markersize=10, label='Total Amount'))
            
    if show_line:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=LINE_COLOR, markersize=10, label='Number of Deals'))
        
    chart_ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False, 
                     prop={'weight': 'normal'}, labelspacing=1.0)
    
    plt.title('Data Visualization', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return chart_fig

# --- STREAMLIT APP LAYOUT ---

st.title("📊 Dynamic Grant Funding Chart Generator")
st.markdown("---")

# Initialize buffers and session state
buf_png = BytesIO()
buf_svg = BytesIO()
if 'year_range' not in st.session_state:
    st.session_state['year_range'] = (1900, 2100)
    st.session_state['category_column'] = 'None'
    st.session_state['show_bars'] = True
    st.session_state['show_line'] = True
    st.session_state['buf_png'] = BytesIO()
    st.session_state['buf_svg'] = BytesIO()

# Use a sidebar for controls
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    df = None
    if uploaded_file:
        df, error_msg = load_data(uploaded_file)
        if df is None:
            st.error(error_msg)
            st.stop()
        
        st.caption(f"Loaded **{df.shape[0]}** rows for processing.")
        
    if df is not None:
        st.markdown("---")
        st.header("2. Configure Visualization")
        
        # --- New Year Selection (Start/End Select Boxes) ---
        min_year = int(df[DATE_COLUMN].dt.year.min())
        max_year = int(df[DATE_COLUMN].dt.year.max())
        all_years = list(range(min_year, max_year + 1))
        
        # Determine initial selection defaults
        default_start = min_year
        default_end = max_year
        
        # Retrieve or set current session state values
        current_start, current_end = st.session_state.get('year_range', (default_start, default_end))
        
        col_start, col_end = st.columns(2)
        
        with col_start:
            # Set the index based on the current session state value
            start_year = st.selectbox(
                "Select Start Year",
                options=all_years,
                index=all_years.index(current_start) if current_start in all_years else 0,
                key='start_year_selector'
            )
            
        with col_end:
            # Set the index based on the current session state value
            end_year = st.selectbox(
                "Select End Year",
                options=all_years,
                index=all_years.index(current_end) if current_end in all_years else len(all_years) - 1,
                key='end_year_selector'
            )
            
        # Validate selection
        if start_year > end_year:
            st.error("Start Year must be less than or equal to End Year.")
            st.stop()
            
        year_range = (start_year, end_year)
        
        # --- Category Column Selection ---
        category_columns = ['None'] + sorted([col for col in df.columns if col not in [DATE_COLUMN, VALUE_COLUMN]])
        category_column = st.selectbox(
            "Select Category Column (Splits bars)", 
            category_columns,
            index=category_columns.index(st.session_state.get('category_column', 'None')),
            key='category_col_selector'
        )

        # --- Display Options ---
        st.subheader("Chart Elements")
        show_bars = st.checkbox(
            "Show Total Grant Amount Bars", 
            value=st.session_state.get('show_bars', True), 
            key='show_bars_selector'
        )
        show_line = st.checkbox(
            "Show Deal Count Line", 
            value=st.session_state.get('show_line', True), 
            key='show_line_selector'
        )
        
        if not show_bars and not show_line:
            st.warning("Please select at least one element (Bars or Line) to display the chart.")
            st.stop()
        
        # Update session state with new values
        st.session_state['year_range'] = year_range
        st.session_state['category_column'] = category_column
        st.session_state['show_bars'] = show_bars
        st.session_state['show_line'] = show_line
        
        # --- DOWNLOAD SECTION (Sidebar) ---
        st.markdown("---")
        st.header("3. Download Chart ⬇️")
        
        st.download_button(
            label="Download Chart as **PNG** 🖼️",
            data=st.session_state.get('buf_png', BytesIO()),
            file_name="grant_funding_chart.png",
            mime="image/png",
            key="download_png",
            use_container_width=True
        )
        st.download_button(
            label="Download Chart as **SVG** 📐",
            data=st.session_state.get('buf_svg', BytesIO()),
            file_name="grant_funding_chart.svg",
            mime="image/svg+xml",
            key="download_svg",
            use_container_width=True
        )


# --- MAIN AREA: CHART GENERATION ---

if df is not None:
    
    # Retrieve parameters from session state
    year_range = st.session_state['year_range']
    category_column = st.session_state['category_column']
    show_bars = st.session_state['show_bars']
    show_line = st.session_state['show_line']
    
    # Process the data
    final_data, process_error = process_data(df, year_range, category_column)
    
    if final_data is None:
        st.error(process_error)
        st.stop()
    
    # Generate the chart
    chart_fig = generate_chart(final_data, category_column, show_bars, show_line)

    st.pyplot(chart_fig, use_container_width=True)
    
    # --- Export Figure to Buffers (to update sidebar download buttons) ---
    
    # PNG
    buf_png = BytesIO()
    chart_fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    st.session_state['buf_png'] = buf_png

    # SVG
    buf_svg = BytesIO()
    chart_fig.savefig(buf_svg, format='svg', bbox_inches='tight')
    buf_svg.seek(0)
    st.session_state['buf_svg'] = buf_svg

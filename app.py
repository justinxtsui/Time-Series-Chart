import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from streamlit_sortables import sort_items

# --- CONFIGURATION ---
# Define required column names
DATE_COLUMN = 'Deal date' 
VALUE_COLUMN = 'Amount raised (converted to GBP)' 
# Alternative Column Names (Original Names for Backwards Compatibility)
ALT_DATE_COLUMN = 'Date the participant received the grant'
ALT_VALUE_COLUMN = 'Amount received (converted to GBP)'
# Define the color palette for categories
CATEGORY_COLORS = ['#302A7E', '#D0CCE5']  # Dark Purple and Light Lavender only

# Predefined color palette for user selection (3 purple/lavender shades)
PREDEFINED_COLORS = {
    'Dark Purple': '#302A7E',
    'Medium Purple': '#8884B3',
    'Light Lavender': '#D0CCE5'
}
# Define the default single bar color (third color in the palette for a lighter tone)
SINGLE_BAR_COLOR = '#BBBAF6'
# Define the line chart color
LINE_COLOR = '#000000' # Black for high contrast
# Define the chart title color
TITLE_COLOR = '#000000' # Matplotlib Chart Title Color is Black
# Define the Application Title Color (Black)
APP_TITLE_COLOR = '#000000' 
# Default Title
DEFAULT_TITLE = 'Grant Funding and Deal Count Over Time'

# Set page config and general styles
st.set_page_config(page_title="Time Series Chart Generator", layout="wide", initial_sidebar_state="expanded")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# Minimal LinkedIn-style CSS
st.markdown("""
    <style>
    /* Clean typography */
    html, body, [class*="css"] {
        font-family: -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main content background */
    .main {
        background-color: #f3f2ef;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: #000;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Code blocks */
    code {
        background-color: #f3f6f8;
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 0.9em;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

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
        # Calculate luminance
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
        return luminance < 0.5
    except ValueError:
        return False

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded file, handling dual column names."""
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        # Load the first sheet
        data = pd.read_excel(uploaded_file, sheet_name=0)
        
    # 1. Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()
    
    # Track original value column name for legend
    original_value_column = None
    
    # 2. Check and rename date column
    if DATE_COLUMN not in data.columns:
        if ALT_DATE_COLUMN in data.columns:
            data.rename(columns={ALT_DATE_COLUMN: DATE_COLUMN}, inplace=True)
        else:
            return None, f"File must contain a date column named **`{DATE_COLUMN}`** or **`{ALT_DATE_COLUMN}`**.", None

    # 3. Check and rename value column
    if VALUE_COLUMN not in data.columns:
        if ALT_VALUE_COLUMN in data.columns:
            original_value_column = 'received'  # Track that it was "received"
            data.rename(columns={ALT_VALUE_COLUMN: VALUE_COLUMN}, inplace=True)
        else:
            return None, f"File must contain a value column named **`{VALUE_COLUMN}`** or **`{ALT_VALUE_COLUMN}`**.", None
    else:
        original_value_column = 'raised'  # Track that it was "raised"

    try:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors='coerce')
        data.dropna(subset=[DATE_COLUMN], inplace=True)
    except Exception:
        return None, f"Could not convert **`{DATE_COLUMN}`** to datetime format.", None

    return data, None, original_value_column

@st.cache_data
def apply_filter(df, filter_config):
    """Applies dynamic filters to the DataFrame."""
    if not filter_config['enabled'] or filter_config['column'] == 'None':
        return df

    col = filter_config['column']
    values = filter_config['values']
    is_include = filter_config['include']

    if values:
        if is_include:
            return df[df[col].isin(values)]
        else:
            return df[~df[col].isin(values)]
    return df

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


def generate_chart(final_data, category_column, show_bars, show_line, chart_title, original_value_column='raised', category_colors=None, category_order=None):
    """Generates the dual-axis Matplotlib chart."""
    # Matplotlib Figure Size (Increased for resolution)
    chart_fig, chart_ax1 = plt.subplots(figsize=(20, 10)) 
    
    bar_width = 0.8
    x_pos = np.arange(len(final_data))
    
    # --- DYNAMIC FONT SIZE CALCULATION ---
    
    num_bars = len(final_data)
    min_size = 8    # Minimum acceptable font size
    max_size = 22   # Maximum acceptable font size
    
    if num_bars > 0:
        # Scaling numerator INCREASED to 150 for greater sensitivity.
        scale_factor = 150 / num_bars 
        
        # Apply both minimum and maximum caps
        DYNAMIC_FONT_SIZE = int(max(min_size, min(max_size, scale_factor)))
    else:
        DYNAMIC_FONT_SIZE = 12
    # -------------------------------------------------------------
    
    
    category_cols = []
    if category_column != 'None':
        category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]
        
        # Sort categories by user-defined order if provided
        if category_order:
            # Create a list of (category, order) tuples
            category_order_list = [(cat, category_order.get(cat, 999)) for cat in category_cols]
            # Sort by order value
            category_order_list.sort(key=lambda x: x[1])
            # Extract sorted category names
            category_cols = [cat for cat, _ in category_order_list]

    if category_column == 'None':
        y_max = final_data[VALUE_COLUMN].max()
    else:
        y_max = final_data[category_cols].sum(axis=1).max()

    # Use vertical_offset for placement near the base of the bar
    vertical_offset = y_max * 0.01 
    
    # --- AXIS 1 (Bar Chart - Value) ---
    if category_column != 'None':
        bottom = np.zeros(len(final_data))
        for idx, cat in enumerate(category_cols):
            # Use custom color if available, otherwise use default palette
            if category_colors and cat in category_colors:
                color = category_colors[cat]
            else:
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
                    
                    # Vertical positioning logic (near the base / center):
                    if idx == 0:
                        # Bottom segment: placed just above the x-axis
                        y_pos = vertical_offset
                        va = 'bottom'
                    else:
                        # Upper segments: placed in the vertical middle of the segment
                        y_pos = bottom[i] + val / 2
                        va = 'center'
                        
                    chart_ax1.text(x, y_pos, label_text, ha='center', va=va,
                                     fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color=text_color)
            bottom += final_data[cat].values
    else:
        if show_bars:
            # Bar chart label is used in the legend
            chart_ax1.bar(x_pos, final_data[VALUE_COLUMN], bar_width, 
                          label='Total amount received', color=SINGLE_BAR_COLOR, alpha=1.0) 
        
            for i, x in enumerate(x_pos):
                val = final_data[VALUE_COLUMN].iloc[i]
                if val > 0:
                    label_text = format_currency(val)
                    text_color = '#FFFFFF' if is_dark_color(SINGLE_BAR_COLOR) else '#000000'

                    # Vertical positioning logic (near the base):
                    # Placed just above the x-axis
                    y_pos = vertical_offset
                    va = 'bottom'
                        
                    chart_ax1.text(x, y_pos, label_text, ha='center', va=va,
                                     fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color=text_color)
    
    chart_ax1.set_xticks(x_pos)
    plt.setp(chart_ax1.get_xticklabels(), fontsize=DYNAMIC_FONT_SIZE, fontweight='normal') # Use DYNAMIC_FONT_SIZE for x-ticks
    chart_ax1.set_xticklabels(final_data['time_period'])
    
    chart_ax1.set_ylim(0, y_max * 1.1)
    chart_ax1.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False, length=0)
    chart_ax1.tick_params(axis='x', bottom=False, length=0, pad=6)
    for spine in chart_ax1.spines.values():
        spine.set_visible(False)
    chart_ax1.grid(False)

    # --- AXIS 2 (Line Chart - Count) ---
    if show_line:
        chart_ax2 = chart_ax1.twinx()
        line_data = final_data['row_count'].values
        
        # Line chart label is used in the legend
        chart_ax2.plot(x_pos, line_data, color=LINE_COLOR, marker='o', linewidth=1.5, markersize=6, label='Number of deals') 
        
        # Calculate max_count after plotting to get accurate current limits
        max_count = line_data.max()
        chart_ax2.set_ylim(0, max_count * 1.5)
        
        chart_ax2.tick_params(axis='y', right=False, labelright=False, left=False, labelleft=False, length=0)
        for spine in chart_ax2.spines.values():
            spine.set_visible(False)
            
        y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
        base_offset = y_range * 0.025 
        
        # --- PEAK/VALLEY/SLOPE PLACEMENT LOGIC ---
        num_points = len(line_data)
        
        for i, y in enumerate(line_data):
            x = x_pos[i]
            
            # Default placement is ABOVE (va='bottom')
            place_above = True
            
            if num_points == 1:
                place_above = True
            elif i == 0:
                # First point: Check slope to the next point
                if line_data[i+1] > y:
                    place_above = True
                elif line_data[i+1] < y:
                    place_above = False
            elif i == num_points - 1:
                # Last point: Check slope from the previous point
                if line_data[i-1] < y:
                    place_above = True
                elif line_data[i-1] > y:
                    place_above = False
            else:
                # Middle points: Check incoming and outgoing slopes
                y_prev = line_data[i-1]
                y_next = line_data[i+1]
                
                is_peak = (y > y_prev) and (y > y_next)
                is_valley = (y < y_prev) and (y < y_next)

                if is_peak:
                    place_above = True 
                elif is_valley:
                    place_above = False
                elif y > y_prev and y < y_next:
                    place_above = True
                elif y < y_prev and y > y_next:
                    place_above = False
                elif y_prev == y and y_next > y:
                    place_above = True
                elif y_prev == y and y_next < y:
                    place_above = False
                elif y_prev < y and y_next == y:
                    place_above = True
                elif y_prev > y and y_next == y:
                    place_above = False
                else:
                    place_above = True
                        
            # Determine final vertical alignment and position
            if place_above:
                # Place ABOVE the dot (va='bottom', text sits on top of y_pos)
                va = 'bottom'
                y_pos = y + base_offset
            else:
                # Place BELOW the dot (va='top', text hangs below y_pos)
                va = 'top' 
                y_pos = y - base_offset
            
            chart_ax2.text(x, y_pos, str(int(y)), ha='center', va=va, 
                            fontsize=DYNAMIC_FONT_SIZE, # <-- APPLY DYNAMIC FONT SIZE
                            color=LINE_COLOR, fontweight='bold')
    
    # --- LEGEND & TITLE ---
    legend_elements = []
    
    # Define large font size for legend
    LEGEND_FONT_SIZE = 18  # Legend font size
    # Keep marker size fixed at 16 points
    LEGEND_MARKER_SIZE = 16
    
    # Set legend label based on original column type
    if original_value_column == 'received':
        bar_legend_label = 'Total amount received'
    else:  # 'raised'
        bar_legend_label = 'Amount raised'
    
    if show_bars:
        if category_column != 'None':
            for idx, cat in enumerate(category_cols):
                # Use custom color if available, otherwise use default palette
                if category_colors and cat in category_colors:
                    color = category_colors[cat]
                else:
                    color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
                # Use proportional marker size
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=LEGEND_MARKER_SIZE, label=cat)) 
        else:
            # Use dynamic legend label
            # Use proportional marker size
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=SINGLE_BAR_COLOR, markersize=LEGEND_MARKER_SIZE, label=bar_legend_label)) 
            
    if show_line:
        # UPDATED LEGEND LABEL
        # Use proportional marker size
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=LINE_COLOR, markersize=LEGEND_MARKER_SIZE, label='Number of deals')) 
        
    # Legend with increased font size and proportional markers
    chart_ax1.legend(handles=legend_elements, loc='upper left', 
                     prop={'size': LEGEND_FONT_SIZE, 'weight': 'normal'}, 
                     frameon=False, labelspacing=1.0)
    
    # Matplotlib Chart Title: Color is TITLE_COLOR (Black)
    plt.title(chart_title, fontsize=18, fontweight='bold', pad=20, color=TITLE_COLOR)
    plt.tight_layout()
    
    return chart_fig

# --- STREAMLIT APP LAYOUT ---

# 1. MAIN APPLICATION TITLE
st.markdown("""
    <div style="background: white; 
                padding: 24px 0; 
                border-bottom: 1px solid #e0e0e0;
                margin-bottom: 24px;">
        <h1 style="color: #000; margin: 0 0 8px 0; font-size: 32px; font-weight: 600;">
            Time Series Chart Generator
        </h1>
        <p style="color: #666; margin: 0 0 12px 0; font-size: 15px;">
            Turn any fundraising or grant export into a time series chart – JT
        </p>
        <a href="https://platform.beauhurst.com/search/advancedsearch/?avs_json=eyJiYXNlIjoiY29tcGFueSIsImNvbWJpbmUiOiJhbmQiLCJjaGlsZHJlbiI6W119" 
           target="_blank" 
           style="color: #0073b1; text-decoration: none; font-size: 14px; font-weight: 500;">
           Beauhurst Advanced Search →
        </a>
    </div>
""", unsafe_allow_html=True)

# Initialize buffers and session state
if 'year_range' not in st.session_state:
    st.session_state['year_range'] = (1900, 2100)
    st.session_state['category_column'] = 'None'
    st.session_state['show_bars'] = True
    st.session_state['show_line'] = True
    st.session_state['chart_title'] = DEFAULT_TITLE
    st.session_state['buf_png'] = BytesIO()
    st.session_state['buf_svg'] = BytesIO()
    st.session_state['filter_enabled'] = False
    st.session_state['filter_column'] = 'None'
    st.session_state['filter_include'] = True
    st.session_state['filter_values'] = []
    st.session_state['original_value_column'] = 'raised'  # Default
    st.session_state['stacked_enabled'] = False  # Default
    st.session_state['category_colors'] = {}  # Default
    st.session_state['category_order'] = {}  # Default


# --- SIDEBAR (All Controls) ---
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'xls', 'csv'], 
                                     help="The file must contain a date column and a value column.")
    st.markdown("---")

    # Initialize df_base to None
    df_base = None 
    
    if uploaded_file:
        df_base, error_msg, original_value_column = load_data(uploaded_file)
        if df_base is None:
            st.error(error_msg)
            st.stop()
        
        st.caption(f"Loaded **{df_base.shape[0]}** rows for processing.")
        # Store original_value_column in session state
        st.session_state['original_value_column'] = original_value_column
        
    if df_base is not None:
        
        # --- 2. CHART TITLE ---
        st.markdown("---")
        st.header("2. Chart Title")
        
        custom_title = st.text_input(
            "Chart Title", 
            value=st.session_state.get('chart_title', DEFAULT_TITLE),
            key='chart_title_input',
            help="Customize the title shown above the chart."
        )
        st.session_state['chart_title'] = custom_title
        
        # --- 3. TIME FILTERS ---
        st.markdown("---")
        st.header("3. Time Filters")
        
        # FIX: Using df_base inside the conditional block
        min_year = int(df_base[DATE_COLUMN].dt.year.min())
        max_year = int(df_base[DATE_COLUMN].dt.year.max())
        all_years = list(range(min_year, max_year + 1))
        
        default_start = min_year
        default_end = max_year
        
        current_start, current_end = st.session_state.get('year_range', (default_start, default_end))
        
        col_start, col_end = st.columns(2)
        
        with col_start:
            start_year = st.selectbox(
                "Start Year",
                options=all_years,
                index=all_years.index(current_start) if current_start in all_years else 0,
                key='start_year_selector',
                help="First year of data to include."
            )
            
        with col_end:
            end_year = st.selectbox(
                "End Year",
                options=all_years,
                index=all_years.index(current_end) if current_end in all_years else len(all_years) - 1,
                key='end_year_selector',
                help="Last year of data to include."
            )
            
        if start_year > end_year:
            st.error("Start Year must be <= End Year.")
            st.stop()
            
        year_range = (start_year, end_year)
        
        # --- 4. VISUAL ELEMENTS ---
        st.markdown("---")
        st.header("4. Visual Elements")
        
        col_elem_1, col_elem_2 = st.columns(2)
        
        with col_elem_1:
            show_bars = st.checkbox(
                "Show bar for deal value", 
                value=st.session_state.get('show_bars', True), 
                key='show_bars_selector'
            )
        with col_elem_2:
            show_line = st.checkbox(
                "Show line for number of deals", 
                value=st.session_state.get('show_line', True), 
                key='show_line_selector'
            )
        
        if not show_bars and not show_line:
            st.warning("Select at least one element.")
            st.stop()
        
        # Update session state
        st.session_state['year_range'] = year_range
        st.session_state['show_bars'] = show_bars
        st.session_state['show_line'] = show_line
        
        # --- 5. STACKED BAR (OPTIONAL) ---
        st.markdown("---")
        st.header("5. Stacked bar? (Optional)")

        stacked_enabled = st.checkbox('Enable Stacked Bar', value=st.session_state.get('stacked_enabled', False))
        st.session_state['stacked_enabled'] = stacked_enabled

        if stacked_enabled:
            config_columns = [col for col in df_base.columns if col not in [DATE_COLUMN, VALUE_COLUMN]]
            category_columns = ['None'] + sorted(config_columns)
            
            category_column = st.selectbox(
                "Select Column for Stacking", 
                category_columns,
                index=category_columns.index(st.session_state.get('category_column', 'None')),
                key='category_col_selector',
                help="Select a column to stack and color-code the bars."
            )
            st.session_state['category_column'] = category_column
            
            # Color picker for each category  
            if category_column != 'None':
                st.subheader("Category Order & Colors")
                
                # Enhanced CSS for modern, clean design
                st.markdown("""
                    <style>
                    /* Modern sortable styling */
                    .sortable-item {
                        background: white !important;
                        border: 2px dashed #d0d0d0 !important;
                        border-radius: 8px !important;
                        padding: 14px 16px !important;
                        margin: 10px 0 !important;
                        cursor: grab !important;
                        transition: all 0.2s ease !important;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
                    }
                    .sortable-item:hover {
                        background: #fafafa !important;
                        border-color: #8884B3 !important;
                        border-style: solid !important;
                        box-shadow: 0 2px 8px rgba(136,132,179,0.15) !important;
                        transform: translateY(-1px) !important;
                    }
                    .sortable-item:active {
                        cursor: grabbing !important;
                    }
                    .sortable-ghost {
                        opacity: 0.4 !important;
                        background: #f0f0f0 !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Get unique categories from the selected column
                unique_categories = sorted(df_base[category_column].dropna().unique())
                
                # Initialize category_colors and category_order in session state if not exists
                if 'category_colors' not in st.session_state:
                    st.session_state['category_colors'] = {}
                if 'category_order' not in st.session_state:
                    st.session_state['category_order'] = {}
                
                # Initialize sorted category list if not exists or if categories changed
                if 'sorted_categories' not in st.session_state or set(st.session_state.get('sorted_categories', [])) != set(unique_categories):
                    st.session_state['sorted_categories'] = list(reversed(unique_categories))  # Reversed so top = top
                
                # Pre-assign colors before showing drag interface
                for idx, category in enumerate(st.session_state['sorted_categories']):
                    if category not in st.session_state['category_colors']:
                        default_color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
                        st.session_state['category_colors'][category] = default_color
                
                # Drag section
                st.markdown("**Drag to Reorder**")
                
                # Simple drag interface
                sorted_categories = sort_items(
                    st.session_state['sorted_categories'],
                    direction='vertical',
                    key='category_sorter'
                )
                
                # Update sorted categories in session state
                st.session_state['sorted_categories'] = sorted_categories
                
                # Update category order based on sorted list (higher number = higher in stack)
                num_categories = len(sorted_categories)
                for idx, category in enumerate(sorted_categories):
                    st.session_state['category_order'][category] = num_categories - idx
                
                st.markdown("---")
                
                # Color selection section
                st.markdown("**Assign Colors**")
                
                for idx, category in enumerate(sorted_categories):
                    current_color = st.session_state['category_colors'].get(category, CATEGORY_COLORS[idx % len(CATEGORY_COLORS)])
                    
                    # Create row with category, dropdown, and color box
                    col1, col2, col3 = st.columns([1, 1.5, 0.5])
                    
                    with col1:
                        # Category name
                        st.markdown(f"<div style='padding-top: 8px; font-size: 16px;'><strong>{category}</strong></div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Dropdown with just hex codes (no emojis)
                        color_options = list(PREDEFINED_COLORS.values())
                        
                        selected_hex = st.selectbox(
                            f"Color for {category}",
                            options=color_options,
                            index=color_options.index(current_color) if current_color in color_options else 0,
                            key=f'color_select_{category}',
                            label_visibility='collapsed'
                        )
                        
                        st.session_state['category_colors'][category] = selected_hex
                    
                    with col3:
                        # Colored square box showing selected color
                        st.markdown(
                            f'<div style="background-color: {selected_hex}; height: 38px; width: 100%; '
                            f'border-radius: 4px; border: 2px solid #ddd; margin-top: 0px;"></div>',
                            unsafe_allow_html=True
                        )
        else:
            st.session_state['category_column'] = 'None'
            st.session_state['category_colors'] = {}
            st.session_state['category_order'] = {}
            if 'sorted_categories' in st.session_state:
                del st.session_state['sorted_categories']

        # --- 6. DATA FILTER ---
        st.markdown("---")
        st.header("6. Data Filter")

        filter_enabled = st.checkbox('Enable Data Filtering', value=st.session_state['filter_enabled'])
        st.session_state['filter_enabled'] = filter_enabled

        if filter_enabled:
            
            filter_columns = [c for c in df_base.columns if df_base[c].dtype in ['object', 'category'] and c not in [DATE_COLUMN]]
            filter_columns = ['None'] + sorted(filter_columns)
            
            filter_column = st.selectbox(
                "Select Column to Filter",
                filter_columns,
                index=filter_columns.index(st.session_state['filter_column']) if st.session_state['filter_column'] in filter_columns else 0,
                key='filter_col_selector'
            )
            st.session_state['filter_column'] = filter_column

            if filter_column != 'None':
                
                # Fetch unique values for the selected column
                unique_values = df_base[filter_column].astype(str).unique().tolist()
                
                filter_mode = st.radio(
                    "Filter Mode",
                    options=["Include selected values", "Exclude selected values"],
                    index=0 if st.session_state['filter_include'] else 1,
                    key='filter_mode_radio'
                )
                
                st.session_state['filter_include'] = (filter_mode == "Include selected values")
                
                # Use default from session state or all unique values if first run
                default_selection = st.session_state['filter_values'] if st.session_state['filter_values'] else unique_values
                
                selected_values = st.multiselect(
                    f"Select values in '{filter_column}'",
                    options=unique_values,
                    default=[v for v in default_selection if v in unique_values], # Ensure defaults are valid options
                    key='filter_values_selector'
                )
                st.session_state['filter_values'] = selected_values
            else:
                 st.session_state['filter_values'] = []

        # --- 7. DOWNLOAD SECTION ---
        st.markdown("---")
        st.header("7. Download Chart")
        
        with st.expander("Download Options", expanded=True):
            st.caption("Download your generated chart file.")
            st.download_button(
                label="Download as **PNG** (High-Res)",
                data=st.session_state.get('buf_png', BytesIO()),
                file_name=f"{custom_title.replace(' ', '_').lower()}_chart.png",
                mime="image/png",
                key="download_png",
                use_container_width=True
            )
            st.download_button(
                label="Download as **SVG** (Vector)",
                data=st.session_state.get('buf_svg', BytesIO()),
                file_name=f"{custom_title.replace(' ', '_').lower()}_chart.svg",
                mime="image/svg+xml",
                key="download_svg",
                use_container_width=True
            )


# --- MAIN AREA: CHART DISPLAY ONLY ---

if 'df_base' in locals() and df_base is not None:
    
    # Apply dynamic filter first
    filter_config = {
        'enabled': st.session_state['filter_enabled'],
        'column': st.session_state['filter_column'],
        'include': st.session_state['filter_include'],
        'values': st.session_state['filter_values']
    }
    
    df_filtered = apply_filter(df_base, filter_config)
    
    if df_filtered.empty:
        st.error("The selected filters resulted in no data. Please adjust your configuration.")
        st.stop()
        
    # Process the data
    final_data, process_error = process_data(df_filtered, st.session_state['year_range'], st.session_state['category_column'])
    
    if final_data is None:
        st.error(process_error)
        st.stop()
    
    # Generate the chart
    chart_fig = generate_chart(final_data, st.session_state['category_column'], 
                               st.session_state['show_bars'], st.session_state['show_line'], 
                               st.session_state['chart_title'], 
                               st.session_state.get('original_value_column', 'raised'),
                               st.session_state.get('category_colors', {}),
                               st.session_state.get('category_order', {}))

    # --- CHART CENTERING IMPROVEMENT ---
    # Centering and sizing adjustment: Minimized side margins ([0.05, 7, 0.05])
    col_left, col_chart, col_right = st.columns([0.05, 7, 0.05])
    
    with col_chart:
        # Display the chart. use_container_width=True to fill the allocated column space.
        st.pyplot(chart_fig, use_container_width=True) 
    
    # --- Export Figure to Buffers (for download buttons) ---
    
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

else:
    # Message for initial load - Simple style
    st.markdown("""
        <div style="background: #f3f6f8; 
                    padding: 16px; 
                    border-radius: 4px;
                    margin: 20px 0;">
            <p style="margin: 0; font-size: 14px; color: #333;">
                Please upload your data file using the controls in the sidebar (Section 1) to begin chart configuration.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Expected Data Format Card
    st.markdown("""
        <div style="background: white; 
                    padding: 20px; 
                    border: 1px solid #e0e0e0; 
                    border-radius: 4px;
                    margin: 20px 0;">
            <h3 style="color: #000; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">
                Expected Data Format
            </h3>
            <p style="color: #666; font-size: 14px; line-height: 1.5; margin: 0;">
                Your file must contain, at minimum, a date column (either <code style="background: #f3f6f8; padding: 2px 6px; border-radius: 2px; font-size: 13px;">Deal date</code> or <code style="background: #f3f6f8; padding: 2px 6px; border-radius: 2px; font-size: 13px;">Date the participant received the grant</code>) and a value column (either <code style="background: #f3f6f8; padding: 2px 6px; border-radius: 2px; font-size: 13px;">Amount raised (converted to GBP)</code> or <code style="background: #f3f6f8; padding: 2px 6px; border-radius: 2px; font-size: 13px;">Amount received (converted to GBP)</code>).
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # How It Works Card
    st.markdown("""
        <div style="background: white; 
                    padding: 20px; 
                    border: 1px solid #e0e0e0; 
                    border-radius: 4px;
                    margin: 20px 0;">
            <h3 style="color: #000; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">
                How It Works
            </h3>
            <p style="color: #666; font-size: 14px; line-height: 1.5; margin: 0 0 16px 0;">
                This generator creates professional time series charts visualizing value (bars) and count (line) over time.
            </p>
            <div style="color: #666; font-size: 14px; line-height: 1.8;">
                <div style="margin-bottom: 12px;">
                    <strong style="color: #000;">1. Upload:</strong> Provide your data file in the sidebar.
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #000;">2. Configure:</strong> Use the controls in the sidebar sections to:
                    <ul style="margin: 8px 0 0 20px;">
                        <li>Set your chart title (Section 2)</li>
                        <li>Filter the time range (Section 3)</li>
                        <li>Choose visual elements (Section 4)</li>
                        <li>Enable stacked bars (Section 5)</li>
                        <li>Apply data filters (Section 6)</li>
                    </ul>
                </div>
                <div>
                    <strong style="color: #000;">3. View & Download:</strong> The generated chart will appear instantly here, ready for high-resolution download in Section 7 of the sidebar.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

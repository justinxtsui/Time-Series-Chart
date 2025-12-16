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
# Define required column names
DATE_COLUMN = 'Deal date'
VALUE_COLUMN = 'Amount raised (converted to GBP)'
# Alternative Column Names (Original Names for Backwards Compatibility)
ALT_DATE_COLUMN = 'Date the participant received the grant'
ALT_VALUE_COLUMN = 'Amount received (converted to GBP)'
# Define the color palette for categories
CATEGORY_COLORS = ['#302A7E', '#D0CCE5'] # Dark Purple and Light Lavender only

# Predefined color palette for user selection (3 purple/lavender shades)
PREDEFINED_COLORS = {
    'Dark Purple': '#302A7E',
    'Medium Purple': '#8884B3',
    'Light Lavender': '#D0CCE5'
}
# Define the default single bar color (third color in the palette for a lighter tone)
SINGLE_BAR_COLOR = '#BBBAF6'
# Define the prediction shade color (LIGHT GREY FILL for predicted bars)
PREDICTION_SHADE_COLOR = '#F0F0F0' 
# Define the prediction hatching color (BLACK HATCH for maximum contrast over light grey)
PREDICTION_HATCH_COLOR = '#000000' 
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
            original_value_column = 'received' # Track that it was "received"
            data.rename(columns={ALT_VALUE_COLUMN: VALUE_COLUMN}, inplace=True)
        else:
            return None, f"File must contain a value column named **`{VALUE_COLUMN}`** or **`{ALT_VALUE_COLUMN}`**.", None
    else:
        original_value_column = 'raised' # Track that it was "raised"

    try:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], format='%d/%m/%Y', errors='coerce')
        data.dropna(subset=[DATE_COLUMN], inplace=True)
        # Convert value column to numeric, setting errors='coerce' to turn bad values to NaN
        data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors='coerce')
        data.dropna(subset=[VALUE_COLUMN], inplace=True)

    except Exception as e:
        return None, f"An error occurred during data conversion: {e}", None
    
    # *** FIX: Check if the DataFrame is empty after cleanup/conversion ***
    if data.empty:
        return None, "File loaded but contained no valid rows after processing (missing date or value).", None

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


def generate_chart(final_data, category_column, show_bars, show_line, chart_title, original_value_column='raised', category_colors=None, category_order=None, prediction_start_year=None):
    """Generates the dual-axis Matplotlib chart with prediction styling."""
    # Matplotlib Figure Size (Increased for resolution)
    chart_fig, chart_ax1 = plt.subplots(figsize=(20, 10))
    
    bar_width = 0.8
    x_pos = np.arange(len(final_data))
    years = final_data['time_period'].values
    
    # Determine which bars/points are for predicted data
    is_predicted = (years >= prediction_start_year) if prediction_start_year is not None else np.full(len(years), False)

    # Define bar_legend_label early to prevent UnboundLocalError
    if original_value_column == 'received':
        bar_legend_label = 'Total amount received'
    else:
        bar_legend_label = 'Amount raised'
    
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
            category_order_list = [(cat, category_order.get(cat, 999)) for cat in category_cols]
            category_order_list.sort(key=lambda x: x[1])
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
            
            for i in range(len(final_data)):
                x = x_pos[i]
                val = final_data[cat].iloc[i]
                
                if show_bars and val > 0:
                    
                    # --- SHADING REVERT: Predicted bar uses PREDICTION_SHADE_COLOR (Light Grey) ---
                    bar_color = PREDICTION_SHADE_COLOR if is_predicted[i] else color
                    
                    # Hatching: Always use 'xx' if predicted, with black/dark hatching color
                    hatch_style = 'xx' if is_predicted[i] else None
                    edge_color = PREDICTION_HATCH_COLOR if is_predicted[i] else 'none'
                    # ---------------------------------------------------------------------------------

                    alpha_val = 1.0 
                    
                    # Plot the bar
                    label_str = cat if i == 0 else '_nolegend_'
                    
                    chart_ax1.bar(x, val, bar_width, bottom=bottom[i],
                                  label=label_str, 
                                  color=bar_color, 
                                  alpha=alpha_val, 
                                  hatch=hatch_style,
                                  edgecolor=edge_color,
                                  linewidth=0) # Remove border around bar
                    
                    # Data label logic
                    label_text = format_currency(val)
                    # Text color logic: Always Black for light bars (light grey and light lavender), White for dark purple
                    
                    # Determine base color for dark check (if it's predicted, the base is PREDICTION_SHADE_COLOR)
                    check_color = PREDICTION_SHADE_COLOR if is_predicted[i] else color
                    text_color = '#FFFFFF' if is_dark_color(check_color) else '#000000'
                    
                    # Vertical positioning logic (near the base / center):
                    if idx == 0:
                        y_pos = bottom[i] + vertical_offset
                        va = 'bottom'
                    else:
                        y_pos = bottom[i] + val / 2
                        va = 'center'
                        
                    chart_ax1.text(x, y_pos, label_text, ha='center', va=va,
                                   fontsize=DYNAMIC_FONT_SIZE, fontweight='bold', color=text_color)

                # Update bottom for stacking
                bottom[i] += final_data[cat].iloc[i]

    else:
        # Non-stacked bar chart
        if show_bars:
            for i in range(len(final_data)):
                x = x_pos[i]
                val = final_data[VALUE_COLUMN].iloc[i]
                
                # --- SHADING REVERT: Predicted bar uses PREDICTION_SHADE_COLOR (Light Grey) ---
                bar_color = PREDICTION_SHADE_COLOR if is_predicted[i] else SINGLE_BAR_COLOR
                
                # Hatching: Always use 'xx' if predicted, with black/dark hatching color
                hatch_style = 'xx' if is_predicted[i] else None
                edge_color = PREDICTION_HATCH_COLOR if is_predicted[i] else 'none'
                # ---------------------------------------------------------------------------------

                alpha_val = 1.0 
                
                # Only use label in legend for the first category instance (i==0)
                label_str = bar_legend_label if i == 0 else '_nolegend_'
                
                chart_ax1.bar(x, val, bar_width,
                              label=label_str,
                              color=bar_color, 
                              alpha=alpha_val, 
                              hatch=hatch_style,
                              edgecolor=edge_color,
                              linewidth=0) # Remove border around bar
        
                if val > 0:
                    label_text = format_currency(val)
                    # Text color logic: Black for the light bar color (SINGLE_BAR_COLOR is light purple, PREDICTION_SHADE_COLOR is light grey)
                    text_color = '#000000'

                    # Vertical positioning logic (near the base):
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
        
        # Split data into actual (solid) and predicted (dotted) sections
        actual_x = x_pos[~is_predicted]
        actual_y = line_data[~is_predicted]
        predicted_x = x_pos[is_predicted]
        predicted_y = line_data[is_predicted]
        
        # 1. Plot Actual (Solid Line)
        if len(actual_x) > 0:
            # Only add label if not in prediction mode or if this is the only line shown
            label_str = 'Number of deals' if prediction_start_year is None else '_nolegend_'
            chart_ax2.plot(actual_x, actual_y, color=LINE_COLOR, marker='o', linestyle='-', linewidth=1.5, markersize=6, label=label_str)

        # 2. Plot Predicted (Dotted Line) - Use '_nolegend_' for the prediction line
        if len(predicted_x) > 0:
            if len(actual_x) > 0 and predicted_x[0] == actual_x[-1] + 1:
                connection_x = np.concatenate(([actual_x[-1]], predicted_x))
                connection_y = np.concatenate(([actual_y[-1]], predicted_y))
            else:
                connection_x = predicted_x
                connection_y = predicted_y

            chart_ax2.plot(connection_x, connection_y, color=LINE_COLOR, marker='o', linestyle='--', linewidth=1.5, markersize=6, label='_nolegend_')
        
        # Calculate max_count after plotting to get accurate current limits
        max_count = line_data.max()
        chart_ax2.set_ylim(0, max_count * 1.5)
        
        chart_ax2.tick_params(axis='y', right=False, labelright=False, left=False, labelleft=False, length=0)
        for spine in chart_ax2.spines.values():
            spine.set_visible(False)
            
        y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
        base_offset = y_range * 0.025
        
        # --- LINE DATA LABEL PLACEMENT LOGIC ---
        num_points = len(line_data)
        
        for i, y in enumerate(line_data):
            x = x_pos[i]
            
            # Placement logic remains the same (checking peaks/valleys)
            place_above = True
            if num_points > 1:
                if i == 0:
                    place_above = line_data[i+1] >= y
                elif i == num_points - 1:
                    place_above = line_data[i-1] <= y
                else:
                    is_peak = (y >= line_data[i-1]) and (y >= line_data[i+1])
                    is_valley = (y < line_data[i-1]) and (y < line_data[i+1])
                    place_above = is_peak or (y > line_data[i-1] and y < line_data[i+1])
                    if is_valley:
                        place_above = False

            # Determine final vertical alignment and position
            va = 'bottom' if place_above else 'top'
            y_pos = y + base_offset if place_above else y - base_offset
            
            chart_ax2.text(x, y_pos, str(int(y)), ha='center', va=va,
                           fontsize=DYNAMIC_FONT_SIZE, # <-- APPLY DYNAMIC FONT SIZE
                           color=LINE_COLOR, fontweight='bold')
    
    # --- LEGEND & TITLE ---
    legend_elements = []
    
    # Define large font size for legend
    LEGEND_FONT_SIZE = 18 # Legend font size
    LEGEND_MARKER_SIZE = 16
    
    # --- BAR LEGEND ENTRIES (Actual data only) ---
    if show_bars:
        if category_column != 'None':
            # Add all categories using their defined color (Non-predicted style)
            for idx, cat in enumerate(category_cols):
                if category_colors and cat in category_colors:
                    color = category_colors[cat]
                else:
                    color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
                legend_elements.append(Line2D([0], [0], marker='o', linestyle='',
                                              markerfacecolor=color, markersize=LEGEND_MARKER_SIZE * 0.7, 
                                              markeredgecolor='none', label=cat))
        else:
            # Single bar
            legend_elements.append(Line2D([0], [0], marker='o', linestyle='',
                                          markerfacecolor=SINGLE_BAR_COLOR, markersize=LEGEND_MARKER_SIZE * 0.7, 
                                          markeredgecolor='none', label=bar_legend_label))


    # --- LINE LEGEND ENTRY (Single Entry for all "Number of Deals") ---
    if show_line:
        # Add a single entry for the line count
        legend_elements.append(Line2D([0], [0], color=LINE_COLOR, marker='o', linestyle='-', linewidth=1.5, markersize=6, label='Number of deals'))


    # Filter to unique labels (important for cases where prediction is OFF)
    final_legend_elements = []
    seen_labels = set()
    for element in legend_elements:
        label = element.get_label()
        if label not in seen_labels and label != '_nolegend_':
            final_legend_elements.append(element)
            seen_labels.add(label)

    # Legend with increased font size and proportional markers
    chart_ax1.legend(handles=final_legend_elements, loc='upper left',
                     prop={'size': LEGEND_FONT_SIZE, 'weight': 'normal'},
                     frameon=False, labelspacing=1.0, ncol=2)
    
    # Matplotlib Chart Title: Color is TITLE_COLOR (Black)
    plt.title(chart_title, fontsize=18, fontweight='bold', pad=20, color=TITLE_COLOR)
    plt.tight_layout()
    
    return chart_fig

# --- STREAMLIT APP LAYOUT ---

# 1. MAIN APPLICATION TITLE
st.markdown(f'<h1 style="color:{APP_TITLE_COLOR};">Time Series Chart Generator</h1>', unsafe_allow_html=True)

# Styled description box
st.markdown("""
    <div style="background: #f5f7fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #302A7E;
                margin: 15px 0;">
        <p style="margin: 0 0 10px 0; font-size: 16px; color: #333;">
            <strong>Turn any fundraising or grant export into a time series chart – JT</strong>
        </p>
        <a href="https://platform.beauhurst.com/search/advancedsearch/?avs_json=eyJiYXNlIjoiY29tcGFueSIsImNvbWJpbmUiOiJhbmQiLCJjaGlsZHJlbiI6W119"
            target="_blank"
            style="display: inline-block; background: #fff; padding: 10px 16px; border-radius: 6px;
                    border: 1px solid #ddd; color: #302A7E; font-weight: 600; text-decoration: none;
                    font-size: 14px; transition: all 0.2s ease;">
            🔗 Beauhurst Advanced Search
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

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
    st.session_state['prediction_start_year'] = None # New default

# --- SIDEBAR (All Controls) ---
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'xls', 'csv'],
                                     help="The file must contain a date column and a value column.")

    # Initialize df_base to None
    df_base = None
    
    # --- Load Data and Set Default Years ---
    if uploaded_file:
        df_base, error_msg, original_value_column = load_data(uploaded_file)
        
        # Check if df_base was successfully loaded
        if df_base is not None:
            st.caption(f"Loaded **{df_base.shape[0]}** rows for processing.")
            # Store original_value_column in session state
            st.session_state['original_value_column'] = original_value_column
            
            # **FIX: df_base is now guaranteed to be a non-empty DataFrame here.**
            # Calculate min/max years based on loaded data
            min_year = int(df_base[DATE_COLUMN].dt.year.min())
            max_year = int(df_base[DATE_COLUMN].dt.year.max())
            all_years = list(range(min_year, max_year + 1))
            
            default_start = min_year
            default_end = max_year
            
            # --- Continue with the rest of the configuration blocks only if data is loaded ---
            
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
            
            current_start, current_end = st.session_state.get('year_range', (default_start, default_end))
            
            # Ensure current_start/end are within the actual min/max range
            current_start = max(min_year, min(max_year, current_start))
            current_end = max(min_year, min(max_year, current_end))
            
            col_start, col_end = st.columns(2)
            
            with col_start:
                # Calculate default index for start year safely
                try:
                    start_index = all_years.index(current_start)
                except ValueError:
                    start_index = 0
                    
                start_year = st.selectbox(
                    "Start Year",
                    options=all_years,
                    index=start_index,
                    key='start_year_selector',
                    help="First year of data to include."
                )
                
            with col_end:
                # Calculate default index for end year safely
                try:
                    end_index = all_years.index(current_end)
                except ValueError:
                    end_index = len(all_years) - 1
                    
                end_year = st.selectbox(
                    "End Year",
                    options=all_years,
                    index=end_index,
                    key='end_year_selector',
                    help="Last year of data to include."
                )
                
            if start_year > end_year:
                st.error("Start Year must be <= End Year. Please adjust.")
                # We stop here to prevent running the chart generation with an invalid range
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
                st.warning("Select at least one element (Bar or Line) to display a chart.")
                st.stop()
                
            # Update session state
            st.session_state['year_range'] = year_range
            st.session_state['show_bars'] = show_bars
            st.session_state['show_line'] = show_line
            
            # --- PREDICTION TOGGLE AND YEAR SELECT ---
            st.subheader("Prediction Visuals (Dotted Line / Hatched Bar)")
            enable_prediction = st.checkbox("Enable prediction mode", key='enable_prediction_checkbox')
            
            prediction_start_year = None
            
            # Check if the currently filtered time range has any years
            filtered_years = list(range(start_year, end_year + 1))
            
            if enable_prediction and filtered_years:
                # Only allow selection of years that are visible in the chart
                prediction_options = ['None'] + filtered_years
                
                # Find the index of the current or last year
                default_year_to_select = st.session_state['prediction_start_year']
                if default_year_to_select not in prediction_options:
                    # If the previous selection is outside the new range, default to the last year of the range
                    default_year_to_select = filtered_years[-1] if filtered_years else 'None'
                
                default_index = prediction_options.index(default_year_to_select) if default_year_to_select != 'None' else 0
                
                selected_prediction_year = st.selectbox(
                    "Start Year for Prediction/Shading",
                    options=prediction_options,
                    index=default_index,
                    key='prediction_year_selector',
                    help="Data from this year (inclusive) will be rendered as predicted (dotted line/hatched bars)."
                )
                
                if selected_prediction_year != 'None':
                    prediction_start_year = int(selected_prediction_year)
                    # Ensure the prediction year is within the selected filter range
                    if prediction_start_year < start_year or prediction_start_year > end_year:
                        # Don't stop, but warn and disable the feature for this run
                        st.warning(f"Prediction start year {prediction_start_year} is outside the time filter range ({start_year}-{end_year}). Prediction styling will not be shown.")
                        prediction_start_year = None
                
            st.session_state['prediction_start_year'] = prediction_start_year
            
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
                    index=category_columns.index(st.session_state.get('category_column', 'None')) if st.session_state.get('category_column', 'None') in category_columns else 0,
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
                    unique_categories = sorted(df_base[category_column].dropna().astype(str).unique())
                    
                    # Initialize category_colors and category_order in session state if not exists
                    if 'category_colors' not in st.session_state:
                        st.session_state['category_colors'] = {}
                    if 'category_order' not in st.session_state:
                        st.session_state['category_order'] = {}
                    
                    # Initialize sorted category list if not exists or if categories changed
                    current_sorted = st.session_state.get('sorted_categories', [])
                    if set(current_sorted) != set(unique_categories):
                        # Use new unique categories, maintaining order for existing ones
                        st.session_state['sorted_categories'] = [cat for cat in current_sorted if cat in unique_categories]
                        st.session_state['sorted_categories'].extend([cat for cat in unique_categories if cat not in current_sorted])
                        st.session_state['sorted_categories'] = list(reversed(st.session_state['sorted_categories'])) # Reversed so top = top
                    
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

            filter_enabled = st.checkbox('Enable Data Filtering', value=st.session_state.get('filter_enabled', False))
            st.session_state['filter_enabled'] = filter_enabled

            if filter_enabled:
                
                filter_columns = [c for c in df_base.columns if df_base[c].dtype in ['object', 'category'] and c not in [DATE_COLUMN]]
                filter_columns = ['None'] + sorted(filter_columns)
                
                filter_column = st.selectbox(
                    "Select Column to Filter",
                    filter_columns,
                    index=filter_columns.index(st.session_state.get('filter_column', 'None')) if st.session_state.get('filter_column', 'None') in filter_columns else 0,
                    key='filter_col_selector'
                )
                st.session_state['filter_column'] = filter_column

                if filter_column != 'None':
                    
                    # Fetch unique values for the selected column, coercing to string to handle all types
                    unique_values = df_base[filter_column].astype(str).unique().tolist()
                    
                    filter_mode = st.radio(
                        "Filter Mode",
                        options=["Include selected values", "Exclude selected values"],
                        index=0 if st.session_state.get('filter_include', True) else 1,
                        key='filter_mode_radio'
                    )
                    
                    st.session_state['filter_include'] = (filter_mode == "Include selected values")
                    
                    # Use default from session state or all unique values if first run
                    default_selection = st.session_state.get('filter_values', unique_values)
                    
                    selected_values = st.multiselect(
                        f"Select values in '{filter_column}'",
                        options=unique_values,
                        default=[v for v in default_selection if v in unique_values], # Ensure defaults are valid options
                        key='filter_values_selector'
                    )
                    st.session_state['filter_values'] = selected_values
                else:
                    st.session_state['filter_values'] = []
                    st.session_state['filter_column'] = 'None' # Reset column if filter is active but column is 'None'

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
        else:
            # This handles the case where uploaded_file is present but load_data failed
            st.error(error_msg)
            # Ensure df_base is None so the main block displays the help message
            df_base = None

    # This check ensures that the main body only runs if data was loaded successfully
    if df_base is None:
        # If no file uploaded or file load failed, clear out the configuration states 
        # that depend on the data structure, to avoid stale values
        st.session_state['category_column'] = 'None'
        st.session_state['filter_column'] = 'None'
        st.session_state['filter_values'] = []
        if 'sorted_categories' in st.session_state:
             del st.session_state['sorted_categories']


# --- MAIN AREA: CHART DISPLAY ONLY ---

# Check both local scope and session state for the presence of df_base if necessary,
# but relying on df_base in local scope (set in sidebar) is better.

if 'df_base' in locals() and df_base is not None:
    
    # Apply dynamic filter first
    filter_config = {
        'enabled': st.session_state.get('filter_enabled', False),
        'column': st.session_state.get('filter_column', 'None'),
        'include': st.session_state.get('filter_include', True),
        'values': st.session_state.get('filter_values', [])
    }
    
    df_filtered = apply_filter(df_base, filter_config)
    
    if df_filtered.empty:
        st.error("The selected filters resulted in no data. Please adjust your configuration.")
        # Need to re-stop if data is filtered out
        st.stop()
        
    # Process the data
    final_data, process_error = process_data(df_filtered, st.session_state['year_range'], st.session_state['category_column'])
    
    if final_data is None:
        st.error(process_error)
        st.stop()
    
    # Get prediction year from session state
    prediction_start_year = st.session_state['prediction_start_year']
    
    # Generate the chart, passing the new parameter
    chart_fig = generate_chart(final_data, st.session_state['category_column'],
                               st.session_state['show_bars'], st.session_state['show_line'],
                               st.session_state['chart_title'],
                               st.session_state.get('original_value_column', 'raised'),
                               st.session_state.get('category_colors', {}),
                               st.session_state.get('category_order', {}),
                               prediction_start_year=prediction_start_year)

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
    # Message for initial load
    st.info("⬆️ **Please upload your data file using the controls in the sidebar (Section 1) to begin chart configuration.**")
    st.markdown("---")
    
    st.subheader("Expected Data Format")
    st.markdown(f"""
    Your file must contain, at minimum, a date column (either **`{DATE_COLUMN}`** or **`{ALT_DATE_COLUMN}`**) and a value column (either **`{VALUE_COLUMN}`** or **`{ALT_VALUE_COLUMN}`**).
    """)
    
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    This generator creates professional time series charts visualizing value (bars) and count (line) over time.

    1.  **Upload:** Provide your data file in the sidebar.
    2.  **Configure:** Use the controls in the sidebar sections to:
        - Set your chart title (Section 2)
        - Filter the time range (Section 3)
        - Choose visual elements, including **prediction visuals** (Section 4)
        - Enable stacked bars (Section 5)
        - Apply data filters (Section 6)
    3.  **View & Download:** The generated chart will appear instantly here, ready for high-resolution download in Section 7 of the sidebar.
    """)

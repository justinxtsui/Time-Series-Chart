import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D

# Set page config
st.set_page_config(page_title="Chart Generator", layout="wide")

# Title
st.title("Interactive Chart Generator")

# File upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    # Load the data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # Fixed column names
    date_column = 'Date the participant received the grant'
    value_column = 'Amount received (converted to GBP)'
    
    # Check if required columns exist
    if date_column not in data.columns or value_column not in data.columns:
        st.error(f"File must contain columns: '{date_column}' and '{value_column}'")
        st.stop()
        
    # Ensure date column is datetime for filtering and plotting
    try:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        # Drop rows where date conversion failed to maintain data integrity
        data.dropna(subset=[date_column], inplace=True) 
    except Exception:
        st.error(f"Could not convert '{date_column}' to datetime format.")
        st.stop()
        
    # Process dates to get min/max year for slider
    min_year = data[date_column].dt.year.min()
    max_year = data[date_column].dt.year.max()
    
    st.success(f"File uploaded successfully! X-axis: '{date_column}', Bar value: '{value_column}'. Shape: {data.shape}")
    st.dataframe(data.head())
    
    # Column selection for category (optional) and Year Range
    st.subheader("Configure Chart Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category column selection (optional)
        category_columns = ['None'] + data.columns.tolist()
        category_column = st.selectbox("Select Category Column (Bar colors - optional)", category_columns)
        
    with col2:
        # Year Range Selection
        if pd.isna(min_year) or pd.isna(max_year):
             st.warning("No valid dates found for year selection.")
             year_range = (2000, 2024) # Default fallback
        else:
            year_range = st.slider(
                "Select Year Range",
                min_value=int(min_year),
                max_value=int(max_year),
                value=(int(min_year), int(max_year)),
                step=1
            )
            
    # Display options
    st.subheader("Display Options")
    col3, col4 = st.columns(2)
    
    with col3:
        # Re-introducing show_bars for user control
        show_bars = st.checkbox("Show Bars", value=True) 

    with col4:
        show_line = st.checkbox("Show Line (Number of deals)", value=True)
    
    # Function to format currency values
    def format_currency(value):
        if value >= 1e9:
            val = value / 1e9
            if val >= 100:
                return f'£{int(val)}b'
            elif val >= 10:
                formatted = f'£{val:.1f}b'.rstrip('0').rstrip('.')
                return formatted
            else:
                formatted = f'£{val:.2f}b'.rstrip('0').rstrip('.')
                return formatted
        elif value >= 1e6:
            val = value / 1e6
            if val >= 100:
                return f'£{int(val)}m'
            elif val >= 10:
                formatted = f'£{val:.1f}m'
                if formatted.endswith('.0m'):
                    formatted = formatted.replace('.0m', 'm')
                return formatted
            else:
                formatted = f'£{val:.2f}m'
                if formatted.endswith('.0m'):
                    formatted = formatted.replace('.0m', 'm')
                else:
                    formatted = formatted.rstrip('0').rstrip('.')
                return formatted
        elif value >= 1e3:
            val = value / 1e3
            if val >= 100:
                return f'£{int(val)}k'
            elif val >= 10:
                formatted = f'£{val:.1f}k'
                if formatted.endswith('.0k'):
                    formatted = formatted.replace('.0k', 'k')
                return formatted
            else:
                formatted = f'£{val:.2f}k'
                if formatted.endswith('.0k'):
                    formatted = formatted.replace('.0k', 'k')
                else:
                    formatted = formatted.rstrip('0').rstrip('.')
                return formatted
        else:
            return f'£{value:.2f}'
    
    # Function to check if a color is dark (needs light text) or light (needs dark text)
    def is_dark_color(hex_color):
        """Check if a hex color is dark. Returns True if dark, False if light."""
        # Remove the # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        # If luminance is less than 0.5, it's dark
        return luminance < 0.5

    # Generate chart button
    if st.button("Generate Chart", type="primary"):
        # Filter data based on selected year range
        start_year, end_year = year_range
        chart_data = data[data[date_column].dt.year.between(start_year, end_year, inclusive='both')].copy()
        
        if chart_data.empty:
            st.error(f"No data available for the selected year range: {start_year} - {end_year}")
            st.stop()
            
        # Extract year or appropriate time period from date
        chart_data['time_period'] = chart_data[date_column].dt.year
        
        # Group data based on whether category is selected
        if category_column != 'None':
            # Group by time period and category
            grouped = chart_data.groupby(['time_period', category_column]).agg({
                value_column: 'sum'
            }).reset_index()
            
            # Count rows per time period
            row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
            
            # Pivot to get categories as columns
            pivot_data = grouped.pivot(index='time_period', columns=category_column, values=value_column).fillna(0)
            
            # Merge with row counts
            final_data = pivot_data.reset_index()
            final_data = final_data.merge(row_counts, on='time_period')
            
        else:
            # Group by time period only
            grouped = chart_data.groupby('time_period').agg({
                value_column: 'sum'
            }).reset_index()
            
            # Count rows per time period
            row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
            
            # Merge
            final_data = grouped.merge(row_counts, on='time_period')
        
        st.success("Data processed successfully!")
        st.dataframe(final_data.head())
        
        # Create the chart
        chart_fig, chart_ax1 = plt.subplots(figsize=(10, 8))
        
        # Bar width 0.8
        bar_width = 0.8
        x_pos = np.arange(len(final_data))
        
        # Calculate dynamic font size based on bar width
        fig_width = chart_fig.get_figwidth()
        ax_bbox = chart_ax1.get_position()
        ax_width_inches = fig_width * ax_bbox.width
        bar_width_inches = (ax_width_inches / len(final_data)) * bar_width
        
        # Stronger scaling factor (16) for dynamic font size
        dynamic_font_size = max(9, min(24, int(bar_width_inches * 16))) 
        
        # Identify category columns if they exist (used for y_max and color logic later)
        category_cols = []
        if category_column != 'None':
            category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]

        # Calculate y_max for plot limits and vertical offset
        if category_column == 'None':
            y_max = final_data[value_column].max()
        else:
            # Sum the new category columns to get the total height of the tallest bar
            y_max = final_data[category_cols].sum(axis=1).max()

        # Define the fixed vertical offset for bar labels - close to the bottom
        vertical_offset = y_max * 0.008  # Very close to the base
        
        # Determine if we have categories or not
        if category_column != 'None':
            # Define colors for stacked bars (starting with light purple)
            colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
            
            if show_bars:
                # Create stacked bars
                bottom = np.zeros(len(final_data))
                for idx, cat in enumerate(category_cols):
                    color = colors[idx % len(colors)]
                    chart_ax1.bar(x_pos, final_data[cat], bar_width, bottom=bottom, 
                                label=cat, color=color, alpha=1.0)
                    
                    # Add labels to each bar segment
                    for i, x in enumerate(x_pos):
                        val = final_data[cat].iloc[i]
                        if val > 0:
                            label_text = format_currency(val)
                            
                            # Better contrast logic using is_dark_color function
                            current_color = colors[idx % len(colors)]
                            if is_dark_color(current_color):
                                text_color = '#FFFFFF'  # Pure white for dark backgrounds
                            else:
                                text_color = '#000000'  # Pure black for light backgrounds
                            
                            # Special positioning for bottom segment (first category)
                            if idx == 0:
                                # Bottom bar: position at bottom center with small vertical offset
                                y_pos = vertical_offset
                                chart_ax1.text(x, y_pos, label_text, ha='center', va='bottom',
                                        fontsize=dynamic_font_size, fontfamily='Public Sans', fontweight=600, color=text_color)
                            else:
                                # Other segments: center positioning
                                y_pos = bottom[i] + val / 2
                                chart_ax1.text(x, y_pos, label_text, ha='center', va='center',
                                        fontsize=dynamic_font_size, fontfamily='Public Sans', fontweight=600, color=text_color)
                    
                    bottom += final_data[cat].values
        else:
            # Single bar without categories (use light purple as default)
            if show_bars:
                chart_ax1.bar(x_pos, final_data[value_column], bar_width, 
                            label='Amount raised', color='#EDD9E4', alpha=1.0)
                
                # Add labels to bars
                # Baseline position is the fixed offset for 'just slightly above the bottom'
                baseline_position = vertical_offset 
                for i, x in enumerate(x_pos):
                    val = final_data[value_column].iloc[i]
                    if val > 0:
                        label_text = format_currency(val)
                        # Check contrast for light purple background
                        if is_dark_color('#EDD9E4'):
                            text_color = '#FFFFFF'
                        else:
                            text_color = '#000000'
                        # Alignment is ha='center' 
                        # Font weight is 600 (Semi Bold)
                        chart_ax1.text(x, baseline_position, label_text, ha='center', va='bottom',
                                fontsize=dynamic_font_size, fontfamily='Public Sans', fontweight=600, color=text_color)
        
        # Set up x-axis
        chart_ax1.set_xticks(x_pos)
        
        # Ensure year labels use dynamic font size
        chart_ax1.set_xticklabels(final_data['time_period'])
        
        # Set font properties for X-axis ticks (Year). Size is dynamic_font_size, Weight is 'normal'.
        plt.setp(chart_ax1.get_xticklabels(),
                 fontsize=dynamic_font_size, # Ensures size matches the values/numbers
                 fontfamily='Public Sans',
                 fontweight='normal') # Set to normal weight (not bold)
        
        chart_ax1.tick_params(axis='y', labelsize=10, left=False, labelleft=False, 
                            right=False, labelright=False, length=0)
                            
        # Pad value to position year labels slightly closer to the bars
        chart_ax1.tick_params(axis='x', bottom=False, length=0, pad=6)
        
        # Remove spines
        chart_ax1.spines['top'].set_visible(False)
        chart_ax1.spines['right'].set_visible(False)
        chart_ax1.spines['left'].set_visible(False)
        chart_ax1.spines['bottom'].set_visible(False)
        chart_ax1.grid(False)
        
        # Create second y-axis for row count line
        if show_line:
            chart_ax2 = chart_ax1.twinx()
            chart_ax2.plot(x_pos, final_data['row_count'], color='black', 
                        marker='o', linewidth=1.0, markersize=5, label='Number of deals')
            chart_ax2.tick_params(axis='y', labelsize=10, right=False, labelright=False, 
                                left=False, labelleft=False, length=0)
            chart_ax2.set_ylim(0, final_data['row_count'].max() * 1.5)
            
            # Define colors array for line label contrast checking
            if category_column != 'None':
                colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
            
            # Add labels to line points
            for i, (x, y) in enumerate(zip(x_pos, final_data['row_count'])):
                # Determine if label should go above or below based on neighboring points
                place_below = False
                
                # Check if there are adjacent points that are higher
                if i < len(final_data) - 1:
                    if final_data['row_count'].iloc[i+1] > y:
                        place_below = True
                if i > 0:
                    if final_data['row_count'].iloc[i-1] > y:
                        place_below = True
                
                # Determine text color based on position
                text_color = '#000000'  # Default for labels above (on white background)
                
                # Calculate base offset - even closer to the dot
                y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
                base_offset = y_range * 0.015  # Very small offset for tight spacing
                
                # Check for potential overlap with bar labels when placing below
                if place_below and category_column != 'None':
                    # Get the bar values at this x position to check for overlaps
                    bar_ax1_max = chart_ax1.get_ylim()[1]
                    line_ax2_max = chart_ax2.get_ylim()[1]
                    
                    # Convert line position to bar chart scale
                    line_fraction = y / line_ax2_max
                    approx_bar_y = line_fraction * bar_ax1_max
                    
                    # Calculate where the label will be positioned
                    label_position = y - base_offset
                    label_position_bar_scale = (label_position / line_ax2_max) * bar_ax1_max
                    
                    # Check each bar segment to see if label would overlap
                    cumulative_height = 0
                    needs_extra_offset = False
                    for seg_idx, cat in enumerate(category_cols):
                        segment_value = final_data[cat].iloc[i]
                        segment_top = cumulative_height + segment_value
                        segment_middle = cumulative_height + segment_value / 2
                        
                        # Check if line label would be too close to a bar label
                        # Bar labels are at segment_middle (or at bottom for first segment)
                        if seg_idx == 0:
                            bar_label_position = vertical_offset
                        else:
                            bar_label_position = segment_middle
                        
                        # Convert to line chart scale for comparison
                        bar_label_position_line_scale = (bar_label_position / bar_ax1_max) * line_ax2_max
                        
                        # If within danger zone, need extra offset
                        danger_zone = y_range * 0.12  # Labels should be at least this far apart
                        if abs(label_position - bar_label_position_line_scale) < danger_zone:
                            needs_extra_offset = True
                            # Increase offset to avoid collision
                            base_offset = y_range * 0.06
                            break
                        
                        # Determine segment color for contrast
                        if label_position_bar_scale >= cumulative_height and label_position_bar_scale <= segment_top:
                            segment_color = colors[seg_idx % len(colors)]
                            if is_dark_color(segment_color):
                                text_color = '#FFFFFF'
                            else:
                                text_color = '#000000'
                        
                        cumulative_height += segment_value
                
                elif place_below and category_column == 'None':
                    # Single bar - check if it's dark
                    if is_dark_color('#EDD9E4'):
                        text_color = '#FFFFFF'
                    else:
                        text_color = '#000000'
                
                # Font weight is 600 (Semi Bold)
                if place_below:
                    chart_ax2.text(x, y - base_offset, str(int(y)), ha='center', va='top', fontsize=dynamic_font_size, 
                            fontfamily='Public Sans', color=text_color, fontweight=600)
                else:
                    chart_ax2.text(x, y + base_offset, str(int(y)), ha='center', va='bottom', fontsize=dynamic_font_size, 
                            fontfamily='Public Sans', color=text_color, fontweight=600)
            
            # Remove spines for second axis
            chart_ax2.spines['top'].set_visible(False)
            chart_ax2.spines['right'].set_visible(False)
            chart_ax2.spines['left'].set_visible(False)
            chart_ax2.spines['bottom'].set_visible(False)
        
        # Create custom legend with circles
        legend_elements = []
        if show_bars:
            if category_column != 'None':
                category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]
                colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
                for idx, cat in enumerate(category_cols):
                    color = colors[idx % len(colors)]
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=cat))
            else:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='#EDD9E4', markersize=10, label='Amount raised'))
        
        if show_line:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='black', markersize=10, label='Number of deals'))
        
        # Legend is not semi-bold
        chart_ax1.legend(handles=legend_elements, loc='upper left', fontsize=18, frameon=False, 
                    prop={'family': 'Public Sans', 'weight': 'normal'}, labelspacing=1.2)
        
        # Title uses fontweight=600 (Semi Bold)
        plt.title('Data Visualization', fontsize=14, fontweight=600, pad=20, fontfamily='Public Sans')
        plt.tight_layout()
        
        # Display the chart in Streamlit
        st.pyplot(chart_fig)
        
        # Add download buttons for both PNG and SVG
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # PNG download
            buf_png = BytesIO()
            chart_fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
            buf_png.seek(0)
            
            st.download_button(
                label="Download Chart as PNG",
                data=buf_png,
                file_name="chart.png",
                mime="image/png"
            )
        
        with col_download2:
            # SVG download
            buf_svg = BytesIO()
            chart_fig.savefig(buf_svg, format='svg', bbox_inches='tight')
            buf_svg.seek(0)
            
            st.download_button(
                label="Download Chart as SVG",
                data=buf_svg,
                file_name="chart.svg",
                mime="image/svg+xml"
            )

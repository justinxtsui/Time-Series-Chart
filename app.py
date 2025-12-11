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
    
    # Hardcode column names as per user request
    date_column = "deal date"
    value_column = "amount raised"

    # Check if the required columns exist
    if date_column not in data.columns or value_column not in data.columns:
        st.error(f"Required columns '{date_column}' and '{value_column}' must be present in the uploaded file.")
        st.stop()
        
    st.success(f"File uploaded successfully! X-axis set to '{date_column}', Bar value set to '{value_column}'. Shape: {data.shape}")
    st.dataframe(data.head())
    
    # Column selection
    st.subheader("Configure Chart Settings")
    
    # Use a single column for the only remaining selection (Category)
    col_cat = st.columns(1)[0]
    
    with col_cat:
        # Category column selection (optional)
        category_columns = ['None'] + data.columns.tolist()
        category_column = st.selectbox("Select Category Column (Bar colors - optional)", category_columns)
    
    # Display options
    st.subheader("Display Options")
    col4, col5 = st.columns(2)
    
    with col4:
        show_bars = st.checkbox("Show Bars", value=True)
    
    with col5:
        show_line = st.checkbox("Show Line (Number of deals)", value=True)
    
    # Function to calculate dynamic font size
    def calculate_dynamic_font_size(num_elements):
        # Base size for few elements
        base_size = 12
        # Determine the reduction factor (e.g., reduce size by 1 for every 5 elements over 5)
        reduction = max(0, (num_elements - 5) // 5) 
        # Minimum font size
        min_size = 8
        return max(min_size, base_size - reduction)

    # Generate chart button
    if st.button("Generate Chart", type="primary"):
        # Process the data
        chart_data = data.copy()
        
        # Convert date column to datetime if not already
        try:
            chart_data[date_column] = pd.to_datetime(chart_data[date_column])
        except:
            st.error(f"Could not convert {date_column} to datetime format")
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
        
        # Calculate dynamic font size
        dynamic_bar_font_size = calculate_dynamic_font_size(len(final_data))
        
        # Function to format currency values
        def format_currency(value):
            # Assumes value is in GBP as requested by user
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
        
        # Create the chart with reduced width to bring bars closer
        chart_fig, chart_ax1 = plt.subplots(figsize=(10, 8))
        
        bar_width = 0.6
        x_pos = np.arange(len(final_data))
        
        # Determine if we have categories or not
        if category_column != 'None':
            # Get category columns (all columns except time_period and row_count)
            category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]
            
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
                            y_pos = bottom[i] + val / 2
                            # Use light grey text for dark colors, black for light colors
                            text_color = '#D3D3D3' if idx % 2 == 1 else 'black'
                            chart_ax1.text(x, y_pos, label_text, ha='center', va='center',
                                    fontsize=dynamic_bar_font_size, fontfamily='Public Sans', fontweight='semibold', color=text_color)
                    
                    bottom += final_data[cat].values
        else:
            # Single bar without categories (use light purple as default)
            if show_bars:
                chart_ax1.bar(x_pos, final_data[value_column], bar_width, 
                            label='Amount raised', color='#EDD9E4', alpha=1.0)
                
                # Add labels to bars
                baseline_position = final_data[value_column].iloc[0] * 0.05 if len(final_data) > 0 else 0
                for i, x in enumerate(x_pos):
                    val = final_data[value_column].iloc[i]
                    if val > 0:
                        label_text = format_currency(val)
                        chart_ax1.text(x, baseline_position, label_text, ha='center', va='bottom',
                                fontsize=dynamic_bar_font_size, fontfamily='Public Sans', fontweight='semibold', color='black')
        
        # Set up x-axis
        chart_ax1.set_xticks(x_pos)
        chart_ax1.set_xticklabels(final_data['time_period'], fontfamily='Public Sans', fontsize=12, fontweight='semibold')
        chart_ax1.tick_params(axis='y', labelsize=10, left=False, labelleft=False, 
                            right=False, labelright=False, length=0)
        chart_ax1.tick_params(axis='x', labelsize=12, bottom=False, length=0)
        
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
            
            # Add labels to line points
            for i, (x, y) in enumerate(zip(x_pos, final_data['row_count'])):
                # Determine if label should go above or below
                place_below = False
                if i < len(final_data) - 1:
                    if final_data['row_count'].iloc[i+1] > y:
                        place_below = True
                if i > 0:
                    if final_data['row_count'].iloc[i-1] > y:
                        place_below = True
                
                # Determine text color
                text_color = 'black'
                
                # Calculate offset
                y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
                offset = y_range * 0.02
                
                if place_below:
                    chart_ax2.text(x, y - offset, str(y), ha='center', va='top', fontsize=dynamic_bar_font_size, 
                            fontfamily='Public Sans', color=text_color, fontweight='semibold')
                else:
                    chart_ax2.text(x, y + offset, str(y), ha='center', va='bottom', fontsize=dynamic_bar_font_size, 
                            fontfamily='Public Sans', color=text_color, fontweight='semibold')
            
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
        
        chart_ax1.legend(handles=legend_elements, loc='upper left', fontsize=18, frameon=False, 
                    prop={'family': 'Public Sans', 'weight': 'semibold'}, labelspacing=1.2)
        
        plt.title('Data Visualization', fontsize=14, fontweight='bold', pad=20, fontfamily='Public Sans')
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

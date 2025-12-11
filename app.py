import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Chart Generator", layout="wide")

# --- 2. FORMATTING FUNCTIONS ---

# Function to format currency values for y-axis
def currency_formatter(x, pos):
    if x >= 1e9:
        val = x / 1e9
        return f'£{val:.1f}B'
    elif x >= 1e6:
        val = x / 1e6
        return f'£{val:.1f}M'
    elif x >= 1e3:
        val = x / 1e3
        return f'£{val:.0f}K'
    else:
        return f'£{int(x)}'

# Function to format currency values for bar labels
def format_currency(value):
    if value >= 1e9:
        val = value / 1e9
        if val >= 100:
            return f'£{int(val)}b'
        elif val >= 10:
            return f'£{val:.1f}b'.rstrip('0').rstrip('.')
        else:
            return f'£{val:.2f}b'.rstrip('0').rstrip('.')
    elif value >= 1e6:
        val = value / 1e6
        if val >= 100:
            return f'£{int(val)}m'
        elif val >= 10:
            formatted = f'£{val:.1f}m'
            return formatted.replace('.0m', 'm')
        else:
            formatted = f'£{val:.2f}m'
            return formatted.rstrip('0').rstrip('.')
    elif value >= 1e3:
        val = value / 1e3
        if val >= 100:
            return f'£{int(val)}k'
        elif val >= 10:
            formatted = f'£{val:.1f}k'
            return formatted.replace('.0k', 'k')
        else:
            formatted = f'£{val:.2f}k'
            return formatted.rstrip('0').rstrip('.')
    else:
        return f'£{int(value)}'


# --- 3. MAIN APP LOGIC ---

st.title("Interactive Chart Generator")

# File upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

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
        # Ensure value column is numeric
        data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
        data.dropna(subset=[value_column], inplace=True)
    except Exception as e:
        st.error(f"Could not process required columns. Details: {e}")
        st.stop()
        
    # Process dates to get min/max year for slider
    min_year = data[date_column].dt.year.min()
    max_year = data[date_column].dt.year.max()
    
    st.success(f"File uploaded successfully! X-axis: '{date_column}', Bar value: '{value_column}'. Shape: {data.shape}")
    
    # --- 4. CONFIGURATION WIDGETS ---
    
    st.subheader("Configure Chart Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category column selection (optional)
        category_columns = ['None'] + [col for col in data.columns if data[col].dtype == 'object' or data[col].nunique() < 50]
        category_column = st.selectbox("Select Category Column (Bar colors - optional)", category_columns)
        
    with col2:
        # Year Range Selection
        if pd.isna(min_year) or pd.isna(max_year) or min_year > max_year:
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
        show_bars = st.checkbox("Show Bars (Total Amount)", value=True) 

    with col4:
        show_line = st.checkbox("Show Line (Number of Deals)", value=True)
    
    # --- 5. CHART GENERATION LOGIC ---

    if st.button("Generate Chart", type="primary"):
        st.write("Data processed successfully!")
        
        start_year, end_year = year_range
        # Filter data based on selected year range
        chart_data = data[data[date_column].dt.year.between(start_year, end_year, inclusive='both')].copy()

        if chart_data.empty:
            st.warning("No data available for the selected year range.")
        else:
            # Group data by year
            chart_data['Year'] = chart_data[date_column].dt.year
            
            # Prepare summary for bars (Primary Y-axis)
            if category_column != 'None' and show_bars:
                # Group by Year and Category
                grouped_bars = chart_data.groupby(['Year', category_column])[value_column].sum().unstack(fill_value=0)
                bar_labels = grouped_bars.index.astype(int).tolist()
                categories = grouped_bars.columns.tolist()
            elif show_bars:
                # Group only by Year
                grouped_bars = chart_data.groupby('Year')[value_column].sum()
                bar_labels = grouped_bars.index.astype(int).tolist()
                categories = ['Total']
            else:
                bar_labels = []
                categories = []

            # Prepare summary for line (Secondary Y-axis)
            if show_line:
                grouped_line = chart_data.groupby('Year').size()
                line_labels = grouped_line.index.astype(int).tolist()
                line_values = grouped_line.values
            else:
                line_labels = []
                line_values = []
                
            # Combine all unique years for the x-axis ticks
            all_years = sorted(list(set(bar_labels) | set(line_labels)))
            
            # Create Plot
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # --- Primary Y-axis (Bars: Amount Received) ---
            if show_bars:
                ax1.set_title(f'Total {value_column} by Year ({start_year}-{end_year})', fontsize=16, pad=15)
                ax1.set_xlabel('Year', fontsize=12)
                ax1.set_ylabel(f'Total {value_column}', color='blue', fontsize=12)
                
                # Plot Stacked/Grouped Bars
                if category_column != 'None':
                    bottom_data = np.zeros(len(bar_labels))
                    bar_width = 0.8 / len(categories) # Adjust width for grouped effect if preferred, but stacking is easier
                    
                    # Use a colormap for distinct colors
                    cmap = plt.get_cmap('Paired')
                    colors = [cmap(i / len(categories)) for i in range(len(categories))]

                    # Stacked Bars Logic
                    bar_data = []
                    for i, cat in enumerate(categories):
                        values = grouped_bars[cat].reindex(all_years, fill_value=0).values
                        
                        # Plot the current category's bars
                        rects = ax1.bar(all_years, values, bottom=bottom_data, label=cat, color=colors[i], width=0.8)
                        
                        # Add value labels on top of the stack (only for the last category)
                        if i == len(categories) - 1:
                            for rect, total_value in zip(rects, grouped_bars.sum(axis=1).reindex(all_years, fill_value=0).values):
                                if total_value > 0:
                                    ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + bottom_data[rects.index(rect)] + ax1.get_ylim()[1] * 0.02, 
                                             format_currency(total_value), ha='center', va='bottom', fontsize=8, rotation=90)
                        
                        bottom_data += values
                    
                    # Add legend for categories
                    ax1.legend(loc='upper left', title=category_column, fontsize=9)
                    
                else:
                    # Simple Bar Plot
                    bar_values = grouped_bars.reindex(all_years, fill_value=0).values
                    rects = ax1.bar(all_years, bar_values, color='skyblue', label='Total Amount')
                    
                    # Add value labels
                    for rect in rects:
                        height = rect.get_height()
                        if height > 0:
                            ax1.text(rect.get_x() + rect.get_width() / 2, height + ax1.get_ylim()[1] * 0.02,
                                     format_currency(height), ha='center', va='bottom', fontsize=9, rotation=90)

                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
                ax1.set_ylim(bottom=0)
            
            # --- Secondary Y-axis (Line: Number of Deals) ---
            if show_line:
                ax2 = ax1.twinx()
                ax2.set_ylabel('Number of Deals', color='red', fontsize=12)
                line_values_full = pd.Series(line_values, index=line_labels).reindex(all_years, fill_value=0).values
                
                # Plot Line
                line_plot = ax2.plot(all_years, line_values_full, color='red', marker='o', linestyle='-', label='Number of Deals')
                
                # Add value labels
                for x, y in zip(all_years, line_values_full):
                    if y > 0:
                         ax2.text(x, y + ax2.get_ylim()[1] * 0.02, int(y), ha='center', va='bottom', color='red', fontsize=9)

                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(bottom=0)
                ax2.yaxis.get_major_locator().set_params(integer=True) # Ensure Y-axis ticks are integers
                
                # Combine legends if only the line is shown
                if not show_bars:
                    ax1.legend(line_plot, ['Number of Deals'], loc='upper left', fontsize=9)

            # --- Final Touches ---
            ax1.set_xticks(all_years)
            ax1.set_xticklabels(all_years, rotation=45, ha='right', fontsize=10)
            fig.tight_layout(pad=3.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Display Chart
            st.pyplot(fig)

            # Download Button
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button(
                label="Download Chart as PNG",
                data=buf.getvalue(),
                file_name=f"Grant_Summary_Chart_{start_year}-{end_year}.png",
                mime="image/png"
            )

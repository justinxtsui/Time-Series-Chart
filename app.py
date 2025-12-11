import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

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
    
    st.success(f"File uploaded successfully! Shape: {data.shape}")
    st.dataframe(data.head())
    
    # Column selection
    st.subheader("Configure Chart Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date column selection
        date_columns = data.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        date_column = st.selectbox("Select Date Column (X-axis)", date_columns)
    
    with col2:
        # Value column selection
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        value_column = st.selectbox("Select Value Column (Bar height)", numeric_columns)
    
    with col3:
        # Category column selection (optional)
        category_columns = ['None'] + data.columns.tolist()
        category_column = st.selectbox("Select Category Column (Bar colors - optional)", category_columns)
    
    # Display options
    st.subheader("Display Options")
    col4, col5 = st.columns(2)
    
    with col4:
        show_bars = st.checkbox("Show Bars", value=True)
    
    with col5:
        show_line = st.checkbox("Show Line (Row Count)", value=True)
    
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
        
        # Create the chart
        chart_fig, chart_ax1 = plt.subplots(figsize=(14, 8))
        
        bar_width = 0.6
        x_pos = np.arange(len(final_data))
        
        # Determine if we have categories or not
        if category_column != 'None':
            # Get category columns (all columns except time_period and row_count)
            category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]
            
            # Define colors for stacked bars
            colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
            
            if show_bars:
                # Create stacked bars
                bottom = np.zeros(len(final_data))
                for idx, cat in enumerate(category_cols):
                    color = colors[idx % len(colors)]
                    chart_ax1.bar(x_pos, final_data[cat], bar_width, bottom=bottom, 
                           label=cat, color=color, alpha=1.0)
                    bottom += final_data[cat].values
        else:
            # Single bar without categories
            if show_bars:
                chart_ax1.bar(x_pos, final_data[value_column], bar_width, 
                       label=value_column, color='#6F2A58', alpha=1.0)
        
        # Set up x-axis
        chart_ax1.set_xticks(x_pos)
        chart_ax1.set_xticklabels(final_data['time_period'], fontsize=12)
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
                    marker='o', linewidth=1.5, markersize=5, label='Row Count')
            chart_ax2.tick_params(axis='y', labelsize=10, right=False, labelright=False, 
                           left=False, labelleft=False, length=0)
            chart_ax2.set_ylim(0, final_data['row_count'].max() * 1.5)
            
            # Remove spines for second axis
            chart_ax2.spines['top'].set_visible(False)
            chart_ax2.spines['right'].set_visible(False)
            chart_ax2.spines['left'].set_visible(False)
            chart_ax2.spines['bottom'].set_visible(False)
        
        # Add legend
        if show_bars:
            chart_ax1.legend(loc='upper left', fontsize=12, frameon=False)
        
        plt.title('Data Visualization', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Display the chart in Streamlit
        st.pyplot(chart_fig)
        
        # Add download button for the chart
        buf = BytesIO()
        chart_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Chart as PNG",
            data=buf,
            file_name="chart.png",
            mime="image/png"
        )

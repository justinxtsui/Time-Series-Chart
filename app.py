import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D

# Set page config with custom theme
st.set_page_config(
    page_title="Chart Generator Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Section styling */
    .section-container {
        background: white;
        padding: 1.8rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        color: #1f2937;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .section-header::before {
        content: "";
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-right: 12px;
        border-radius: 2px;
    }
    
    /* Upload box styling */
    .upload-box {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f9fafb;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #667eea;
        background: #f3f4f6;
    }
    
    /* Info box styling */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        width: 100%;
        background: white;
        color: #667eea;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        border: 2px solid #667eea;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #667eea;
        color: white;
        transform: translateY(-1px);
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    /* Selectbox and slider styling */
    .stSelectbox label, .stSlider label {
        font-weight: 600;
        color: #374151;
        font-size: 1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">📊 Chart Generator Pro</h1>
    <p class="header-subtitle">Transform your data into beautiful, professional visualizations</p>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Upload Your Data</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>📝 Required columns:</strong><br>
    • <code>Date the participant received the grant</code><br>
    • <code>Amount received (converted to GBP)</code>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose your file",
    type=['xlsx', 'xls', 'csv'],
    help="Upload an Excel or CSV file with your data",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ Error Loading File</strong><br>
            Could not read the uploaded file. Please ensure it's a valid Excel or CSV file.<br>
            Error: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Fixed column names
    date_column = 'Date the participant received the grant'
    value_column = 'Amount received (converted to GBP)'
    
    # Check if required columns exist
    if date_column not in data.columns or value_column not in data.columns:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Missing Required Columns</strong><br>
            Your file must contain:<br>
            • <code>Date the participant received the grant</code><br>
            • <code>Amount received (converted to GBP)</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    # Ensure date column is datetime
    try:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data.dropna(subset=[date_column], inplace=True)
    except Exception:
        st.error(f"Could not convert '{date_column}' to datetime format.")
        st.stop()
        
    # Process dates
    min_year = data[date_column].dt.year.min()
    max_year = data[date_column].dt.year.max()
        
    # Success message with stats
    st.markdown("""
    <div class="success-box">
        <strong>✅ File uploaded successfully!</strong>
    </div>
    """, unsafe_allow_html=True)
        
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <p class="stats-number">{len(data):,}</p>
            <p class="stats-label">Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <p class="stats-number">{len(data.columns)}</p>
            <p class="stats-label">Columns</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <p class="stats-number">{int(min_year)}</p>
            <p class="stats-label">First Year</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <p class="stats-number">{int(max_year)}</p>
            <p class="stats-label">Last Year</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
        
    # Data preview
    with st.expander("👁️ Preview Data", expanded=False):
        st.dataframe(data.head(10), use_container_width=True)
        
    st.markdown("<hr>", unsafe_allow_html=True)
        
    # Configuration section
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Configure Your Chart</h2>', unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
        
    with col1:
        st.markdown("##### 🎨 Visualization Options")
            
        # Category column selection
        category_columns = ['None'] + data.columns.tolist()
        category_column = st.selectbox(
            "Category Column (for stacked bars)",
            category_columns,
            help="Select a column to create stacked bars with different colors"
        )
            
        # Display options
        show_bars = st.checkbox("Show Bars", value=True, help="Display bar chart")
        show_line = st.checkbox("Show Line (Number of deals)", value=True, help="Display line chart with deal count")
            
    with col2:
        st.markdown("##### 📅 Date Range")
            
        # Year Range Selection
        if pd.isna(min_year) or pd.isna(max_year):
            st.warning("No valid dates found for year selection.")
            year_range = (2000, 2024)
        else:
            year_range = st.slider(
                "Select Year Range",
                min_value=int(min_year),
                max_value=int(max_year),
                value=(int(min_year), int(max_year)),
                step=1,
                help="Filter data by year range"
            )
        
    st.markdown('</div>', unsafe_allow_html=True)
        
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
        
    # Function to check if a color is dark
    def is_dark_color(hex_color):
        """Check if a hex color is dark. Returns True if dark, False if light."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5

    # Generate chart button
    st.markdown("<br>", unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("🎨 Generate Chart", type="primary", use_container_width=True)
        
    if generate_button:
        with st.spinner('✨ Creating your beautiful chart...'):
            # Filter data based on selected year range
            start_year, end_year = year_range
            chart_data = data[data[date_column].dt.year.between(start_year, end_year, inclusive='both')].copy()
                
            if chart_data.empty:
                st.error(f"No data available for the selected year range: {start_year} - {end_year}")
                st.stop()
                    
            # Extract year
            chart_data['time_period'] = chart_data[date_column].dt.year
                
            # Group data
            if category_column != 'None':
                grouped = chart_data.groupby(['time_period', category_column]).agg({
                    value_column: 'sum'
                }).reset_index()
                    
                row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
                pivot_data = grouped.pivot(index='time_period', columns=category_column, values=value_column).fillna(0)
                final_data = pivot_data.reset_index()
                final_data = final_data.merge(row_counts, on='time_period')
            else:
                grouped = chart_data.groupby('time_period').agg({
                    value_column: 'sum'
                }).reset_index()
                row_counts = chart_data.groupby('time_period').size().reset_index(name='row_count')
                final_data = grouped.merge(row_counts, on='time_period')
                
            # Create the chart
            chart_fig, chart_ax1 = plt.subplots(figsize=(12, 7))
            chart_fig.patch.set_facecolor('white')
                
            bar_width = 0.8
            x_pos = np.arange(len(final_data))
                
            # Calculate dynamic font size
            fig_width = chart_fig.get_figwidth()
            ax_bbox = chart_ax1.get_position()
            ax_width_inches = fig_width * ax_bbox.width
            bar_width_inches = (ax_width_inches / len(final_data)) * bar_width
            dynamic_font_size = max(9, min(24, int(bar_width_inches * 16)))
                
            # Identify category columns
            category_cols = []
            if category_column != 'None':
                category_cols = [col for col in final_data.columns if col not in ['time_period', 'row_count']]

            # Calculate y_max
            if category_column == 'None':
                y_max = final_data[value_column].max()
            else:
                y_max = final_data[category_cols].sum(axis=1).max()

            # Define vertical offset for bar labels
            vertical_offset = y_max * 0.008
                
            # Create bars
            if category_column != 'None':
                colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
                    
                if show_bars:
                    bottom = np.zeros(len(final_data))
                    for idx, cat in enumerate(category_cols):
                        color = colors[idx % len(colors)]
                        chart_ax1.bar(x_pos, final_data[cat], bar_width, bottom=bottom, 
                                    label=cat, color=color, alpha=1.0)
                            
                        # Add labels
                        for i, x in enumerate(x_pos):
                            val = final_data[cat].iloc[i]
                            if val > 0:
                                label_text = format_currency(val)
                                current_color = colors[idx % len(colors)]
                                    
                                if is_dark_color(current_color):
                                    text_color = '#FFFFFF'
                                else:
                                    text_color = '#000000'
                                    
                                if idx == 0:
                                    y_pos = vertical_offset
                                    chart_ax1.text(x, y_pos, label_text, ha='center', va='bottom',
                                            fontsize=dynamic_font_size, fontfamily='Public Sans', 
                                            fontweight=600, color=text_color)
                                else:
                                    y_pos = bottom[i] + val / 2
                                    chart_ax1.text(x, y_pos, label_text, ha='center', va='center',
                                            fontsize=dynamic_font_size, fontfamily='Public Sans', 
                                            fontweight=600, color=text_color)
                            
                        bottom += final_data[cat].values
            else:
                if show_bars:
                    chart_ax1.bar(x_pos, final_data[value_column], bar_width, 
                                label='Amount raised', color='#EDD9E4', alpha=1.0)
                        
                    baseline_position = vertical_offset
                    for i, x in enumerate(x_pos):
                        val = final_data[value_column].iloc[i]
                        if val > 0:
                            label_text = format_currency(val)
                            if is_dark_color('#EDD9E4'):
                                text_color = '#FFFFFF'
                            else:
                                text_color = '#000000'
                            chart_ax1.text(x, baseline_position, label_text, ha='center', va='bottom',
                                    fontsize=dynamic_font_size, fontfamily='Public Sans', 
                                    fontweight=600, color=text_color)
                
            # Set up x-axis
            chart_ax1.set_xticks(x_pos)
            chart_ax1.set_xticklabels(final_data['time_period'])
            plt.setp(chart_ax1.get_xticklabels(), fontsize=dynamic_font_size, 
                     fontfamily='Public Sans', fontweight='normal')
                
            chart_ax1.tick_params(axis='y', labelsize=10, left=False, labelleft=False, 
                                right=False, labelright=False, length=0)
            chart_ax1.tick_params(axis='x', bottom=False, length=0, pad=6)
                
            # Remove spines
            for spine in chart_ax1.spines.values():
                spine.set_visible(False)
            chart_ax1.grid(False)
                
            # Create line
            if show_line:
                chart_ax2 = chart_ax1.twinx()
                chart_ax2.plot(x_pos, final_data['row_count'], color='black', 
                            marker='o', linewidth=1.0, markersize=5, label='Number of deals')
                chart_ax2.tick_params(axis='y', labelsize=10, right=False, labelright=False, 
                                    left=False, labelleft=False, length=0)
                chart_ax2.set_ylim(0, final_data['row_count'].max() * 1.5)
                    
                if category_column != 'None':
                    colors = ['#EDD9E4', '#6F2A58', '#A8D5BA', '#FF6B6B', '#4ECDC4', '#FFE66D']
                    
                # Add line labels
                for i, (x, y) in enumerate(zip(x_pos, final_data['row_count'])):
                    place_below = False
                    if i < len(final_data) - 1 and final_data['row_count'].iloc[i+1] > y:
                        place_below = True
                    if i > 0 and final_data['row_count'].iloc[i-1] > y:
                        place_below = True
                        
                    text_color = '#000000'
                    y_range = chart_ax2.get_ylim()[1] - chart_ax2.get_ylim()[0]
                    base_offset = y_range * 0.015
                        
                    if place_below and category_column != 'None':
                        bar_ax1_max = chart_ax1.get_ylim()[1]
                        line_ax2_max = chart_ax2.get_ylim()[1]
                        line_fraction = y / line_ax2_max
                        approx_bar_y = line_fraction * bar_ax1_max
                        label_position = y - base_offset
                        label_position_bar_scale = (label_position / line_ax2_max) * bar_ax1_max
                            
                        cumulative_height = 0
                        for seg_idx, cat in enumerate(category_cols):
                            segment_value = final_data[cat].iloc[i]
                            segment_top = cumulative_height + segment_value
                            segment_middle = cumulative_height + segment_value / 2
                                
                            if seg_idx == 0:
                                bar_label_position = vertical_offset
                            else:
                                bar_label_position = segment_middle
                                
                            bar_label_position_line_scale = (bar_label_position / bar_ax1_max) * line_ax2_max
                            danger_zone = y_range * 0.12
                                
                            if abs(label_position - bar_label_position_line_scale) < danger_zone:
                                base_offset = y_range * 0.06
                                break
                                
                            if label_position_bar_scale >= cumulative_height and label_position_bar_scale <= segment_top:
                                segment_color = colors[seg_idx % len(colors)]
                                if is_dark_color(segment_color):
                                    text_color = '#FFFFFF'
                                else:
                                    text_color = '#000000'
                                
                            cumulative_height += segment_value
                        
                    elif place_below and category_column == 'None':
                        if is_dark_color('#EDD9E4'):
                            text_color = '#FFFFFF'
                        else:
                            text_color = '#000000'
                        
                    if place_below:
                        chart_ax2.text(x, y - base_offset, str(int(y)), ha='center', va='top', 
                                fontsize=dynamic_font_size, fontfamily='Public Sans', 
                                color=text_color, fontweight=600)
                    else:
                        chart_ax2.text(x, y + base_offset, str(int(y)), ha='center', va='bottom', 
                                fontsize=dynamic_font_size, fontfamily='Public Sans', 
                                color=text_color, fontweight=600)
                    
                for spine in chart_ax2.spines.values():
                    spine.set_visible(False)
                
            # Create legend
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
                        prop={'family': 'Public Sans', 'weight': 'normal'}, labelspacing=1.2)
                
            plt.title('Data Visualization', fontsize=14, fontweight=600, pad=20, fontfamily='Public Sans')
            plt.tight_layout()
                
            # Display chart
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Your Chart</h2>', unsafe_allow_html=True)
            st.pyplot(chart_fig)
            st.markdown('</div>', unsafe_allow_html=True)
                
            # Download section
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Download Your Chart</h2>', unsafe_allow_html=True)
                
            col1, col2 = st.columns(2)
                
            with col1:
                st.markdown("##### 🖼️ PNG Format")
                st.markdown("High-resolution image perfect for presentations and documents")
                buf_png = BytesIO()
                chart_fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                buf_png.seek(0)
                st.download_button(
                    label="📥 Download PNG",
                    data=buf_png,
                    file_name="chart.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            with col2:
                st.markdown("##### 📐 SVG Format")
                st.markdown("Scalable vector graphic ideal for editing and print")
                buf_svg = BytesIO()
                chart_fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                buf_svg.seek(0)
                st.download_button(
                    label="📥 Download SVG",
                    data=buf_svg,
                    file_name="chart.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )
                
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Show helpful information when no file is uploaded
    st.markdown("""
    <div class="info-box">
    <strong>👋 Welcome!</strong><br>
    Get started by uploading your data file above. We support Excel (.xlsx, .xls) and CSV formats.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="section-container" style="text-align: center;">
            <h3 style="font-size: 2.5rem; margin: 0;">📊</h3>
            <h4 style="margin: 1rem 0 0.5rem 0;">Multiple Views</h4>
            <p style="color: #6b7280; margin: 0;">Combine bar charts and line graphs for comprehensive insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-container" style="text-align: center;">
            <h3 style="font-size: 2.5rem; margin: 0;">🎨</h3>
            <h4 style="margin: 1rem 0 0.5rem 0;">Beautiful Design</h4>
            <p style="color: #6b7280; margin: 0;">Professional styling with smart color contrast</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="section-container" style="text-align: center;">
            <h3 style="font-size: 2.5rem; margin: 0;">⚡</h3>
            <h4 style="margin: 1rem 0 0.5rem 0;">Export Ready</h4>
            <p style="color: #6b7280; margin: 0;">Download in PNG or SVG for any use case</p>
        </div>
        """, unsafe_allow_html=True)

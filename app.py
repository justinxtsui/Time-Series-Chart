import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title('Dynamic Fundraising Chart')

st.write('Upload a CSV file with columns for Deal date and Amount raised (converted to GBP).')

uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

if uploaded_file is not None:
    fundraising_df = pd.read_csv(uploaded_file)
    if 'Deal date' not in fundraising_df or 'Amount raised (converted to GBP)' not in fundraising_df:
        st.error("CSV must contain columns: 'Deal date' and 'Amount raised (converted to GBP)'.")
    else:
        fundraising_df['Deal date'] = pd.to_datetime(fundraising_df['Deal date'])
        fundraising_df['Year'] = fundraising_df['Deal date'].dt.year
        
        # Time period selection
        time_period = st.selectbox('Display data by:', ['Year', 'Half', 'Quarter', 'Month'])
        
        # Create period column based on selection
        if time_period == 'Year':
            fundraising_df['Period'] = fundraising_df['Year']
            fundraising_df['Period_Label'] = fundraising_df['Year'].astype(str)
        elif time_period == 'Half':
            fundraising_df['Period'] = fundraising_df['Year'].astype(str) + '-H' + ((fundraising_df['Deal date'].dt.month - 1) // 6 + 1).astype(str)
            fundraising_df['Period_Label'] = fundraising_df['Period']
        elif time_period == 'Quarter':
            fundraising_df['Period'] = fundraising_df['Year'].astype(str) + '-Q' + fundraising_df['Deal date'].dt.quarter.astype(str)
            fundraising_df['Period_Label'] = fundraising_df['Period']
        else:  # Month
            fundraising_df['Period'] = fundraising_df['Deal date'].dt.to_period('M')
            fundraising_df['Period_Label'] = fundraising_df['Deal date'].dt.strftime('%Y-%m')
        
        # Aggregate by period and include Year for filtering
        period_data = fundraising_df.groupby(['Period', 'Period_Label']).agg({
            'Amount raised (converted to GBP)': 'sum',
            'Company name': 'count' if 'Company name' in fundraising_df else 'size',
            'Year': 'first'
        }).reset_index()
        period_data.columns = ['Period', 'Period_Label', 'Total Amount', 'Number of Deals', 'Year']
        period_data['Amount (£m)'] = period_data['Total Amount'] / 1_000_000
        
        # Year range selection
        available_years = sorted(fundraising_df['Year'].unique())
        min_year, max_year = min(available_years), max(available_years)
        
        col_year1, col_year2 = st.columns(2)
        with col_year1:
            start_year = st.number_input('Start year:', min_value=min_year, max_value=max_year, value=max(min_year, max_year-9))
        with col_year2:
            end_year = st.number_input('End year:', min_value=min_year, max_value=max_year, value=max_year)
        
        # Filter data by year range using the Year column in period_data
        period_data_filtered = period_data[
            (period_data['Year'] >= start_year) & 
            (period_data['Year'] <= end_year)
        ].copy()
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            display_option = st.radio('Display:', ['Both', 'Bars Only', 'Line Only'])
        with col2:
            predicted_periods = st.multiselect('Mark periods as predicted:', list(period_data_filtered['Period_Label']))

        # Chart logic
        def format_currency(value):
            if value == 0:
                return '£0'
            if value >= 1_000_000_000:
                formatted = value / 1_000_000_000
                if formatted >= 100: return f'£{formatted:.0f}b'
                elif formatted >= 10: return f'£{formatted:.1f}b'
                else: return f'£{formatted:.2f}b'
            elif value >= 1_000_000:
                formatted = value / 1_000_000
                if formatted >= 100: return f'£{formatted:.0f}m'
                elif formatted >= 10: return f'£{formatted:.1f}m'
                else: return f'£{formatted:.2f}m'
            elif value >= 1_000:
                formatted = value / 1_000
                if formatted >= 100: return f'£{formatted:.0f}k'
                elif formatted >= 10: return f'£{formatted:.1f}k'
                else: return f'£{formatted:.2f}k'
            else:
                if value >= 100: return f'£{value:.0f}'
                elif value >= 10: return f'£{value:.1f}'
                else: return f'£{value:.2f}'

        num_bars = len(period_data_filtered)
        base_font_size = max(10, min(17, 120 / num_bars))
        bar_width = max(0.4, min(0.7, 6 / num_bars))

        color_bars = '#A4A2F2'
        plt.rcParams['font.family'] = 'Public Sans'
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Draw bars if needed
        if display_option in ['Both', 'Bars Only']:
            x_positions = range(len(period_data_filtered))
            bars = ax1.bar(x_positions, period_data_filtered['Amount (£m)'], 
                          color=color_bars, alpha=1.0, width=bar_width)
            
            # Shade predicted periods
            for i, period_label in enumerate(period_data_filtered['Period_Label']):
                if period_label in predicted_periods:
                    ax1.bar(i, period_data_filtered.iloc[i]['Amount (£m)'], 
                           color=color_bars, alpha=0.4, width=bar_width, hatch='//')
            
            fixed_label_height = period_data_filtered['Amount (£m)'].max() * 0.02
            for i, row in period_data_filtered.iterrows():
                amount_label = format_currency(row['Total Amount'])
                ax1.text(list(period_data_filtered.index).index(i), fixed_label_height, amount_label,
                        fontsize=base_font_size, ha='center', va='bottom', fontweight='bold')
        
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.tick_params(axis='x', which='both', length=0, pad=8)
        
        # Draw line if needed
        if display_option in ['Both', 'Line Only']:
            ax2 = ax1.twinx()
            color_line = 'black'
            max_bar_height = period_data_filtered['Amount (£m)'].max() if display_option == 'Both' else period_data_filtered['Number of Deals'].max()
            marker_size = base_font_size * 0.4
            
            # Split line into segments for predicted periods
            x_positions = range(len(period_data_filtered))
            deals = period_data_filtered['Number of Deals'].values
            period_labels = period_data_filtered['Period_Label'].values
            
            for i in range(len(x_positions)):
                if i < len(x_positions) - 1:
                    linestyle = '--' if period_labels[i] in predicted_periods or period_labels[i+1] in predicted_periods else '-'
                    ax2.plot([x_positions[i], x_positions[i+1]], [deals[i], deals[i+1]], color=color_line, 
                            linestyle=linestyle, linewidth=2, marker='o', markersize=marker_size)
                elif i == 0:
                    ax2.plot([x_positions[i]], [deals[i]], color=color_line, marker='o', markersize=marker_size)
            
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.yaxis.set_visible(False)
            
            for i, row in period_data_filtered.iterrows():
                ax2.text(list(period_data_filtered.index).index(i), row['Number of Deals'] * 1.03, f"{int(row['Number of Deals'])}",
                        fontsize=base_font_size, ha='center', va='bottom', fontweight='bold')
            
            if display_option == 'Both':
                ax1.set_ylim(0, max_bar_height * 1.1)
                ax2.set_ylim(period_data_filtered['Number of Deals'].min() * 0.7,
                            period_data_filtered['Number of Deals'].max() * 1.3)
        
        plt.title(f'Fundraising Activity by {time_period} ({start_year}-{end_year})', 
                 fontsize=base_font_size, fontweight='bold', pad=20)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_marker_size = base_font_size * 0.75
        legend_elements = []
        if display_option in ['Both', 'Bars Only']:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color_bars, markersize=legend_marker_size, 
                                         label='Amount Raised'))
        if display_option in ['Both', 'Line Only']:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='black', markersize=legend_marker_size, 
                                         label='Number of Deals'))
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=base_font_size, frameon=False)
        
        ax1.grid(False)
        ax1.set_xticks(range(len(period_data_filtered)))
        ax1.set_xticklabels(period_data_filtered['Period_Label'], fontsize=base_font_size, ha='center', rotation=45 if time_period in ['Quarter', 'Month'] else 0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download as SVG
        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
        svg_buffer.seek(0)
        st.download_button(
            label='Download Chart as SVG',
            data=svg_buffer,
            file_name=f'fundraising_chart_{start_year}-{end_year}_{time_period}.svg',
            mime='image/svg+xml'
        )
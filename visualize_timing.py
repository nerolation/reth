#!/usr/bin/env python3
"""
Download timing data from GCS and create Gantt chart visualization
showing transaction execution timeline with IO and EVM phases.

Requirements:
  pip install pandas google-cloud-storage pyarrow plotly
"""

import os
import pandas as pd
from google.cloud import storage
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set GCP credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google-creds.json"

# Configuration
BUCKET_NAME = 'ethereum-execution-times'

def list_available_blocks(bucket_name: str, limit: int = None) -> list:
    """List all available block numbers in GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blocks = []
    for blob in bucket.list_blobs():
        if blob.name.endswith('.parquet'):
            try:
                block_num = int(blob.name.replace('.parquet', ''))
                blocks.append(block_num)
            except ValueError:
                continue
    
    blocks.sort(reverse=True)  # Highest block first
    
    if limit:
        blocks = blocks[:limit]
    
    print(f"Found {len(blocks)} blocks in GCS")
    return blocks

def download_block_data(bucket_name: str, block_number: int) -> pd.DataFrame:
    """Download and return data for a specific block."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{block_number}.parquet")
    
    # Download to temporary file
    temp_file = f"/tmp/block_{block_number}.parquet"
    blob.download_to_filename(temp_file)
    
    # Read parquet
    df = pd.read_parquet(temp_file)
    
    # Clean up
    os.remove(temp_file)
    
    return df

def create_gantt_chart(df: pd.DataFrame, block_number: int, max_transactions: int = 50):
    """Create Gantt chart showing transaction execution timeline."""
    
    # Limit transactions for readability
    if len(df) > max_transactions:
        df = df.head(max_transactions)
        title_suffix = f" (first {max_transactions} of {len(df)} transactions)"
    else:
        title_suffix = f" ({len(df)} transactions)"
    
    # Calculate cumulative start times (assuming sequential execution)
    df['cumulative_start'] = 0.0
    for i in range(1, len(df)):
        # Each transaction starts after the previous one finishes
        df.loc[i, 'cumulative_start'] = df.loc[:i-1, ['IO_time', 'EVM_time']].sum().sum()
    
    # Create data for Gantt chart
    gantt_data = []
    
    for idx, row in df.iterrows():
        # IO phase
        if row['IO_time'] > 0:
            gantt_data.append({
                'Task': row['tx_hash'],
                'Start': row['cumulative_start'],
                'Finish': row['cumulative_start'] + row['IO_time'],
                'Phase': 'IO',
                'Duration': row['IO_time']
            })
        
        # EVM phase
        if row['EVM_time'] > 0:
            gantt_data.append({
                'Task': row['tx_hash'],
                'Start': row['cumulative_start'] + row['IO_time'],
                'Finish': row['cumulative_start'] + row['IO_time'] + row['EVM_time'],
                'Phase': 'EVM',
                'Duration': row['EVM_time']
            })
    
    gantt_df = pd.DataFrame(gantt_data)
    
    # Create the Gantt chart
    fig = go.Figure()
    
    # Define colors for phases
    colors = {'IO': '#FF6B6B', 'EVM': '#4ECDC4'}
    
    for phase in ['IO', 'EVM']:
        phase_data = gantt_df[gantt_df['Phase'] == phase]
        
        fig.add_trace(go.Bar(
            y=phase_data['Task'],
            x=phase_data['Duration'],
            base=phase_data['Start'],
            name=phase,
            orientation='h',
            marker_color=colors[phase],
            hovertemplate='<b>%{y}</b><br>' +
                          f'{phase} Time: %{{x:.2f}} ms<br>' +
                          'Start: %{base:.2f} ms<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Transaction Execution Timeline - Block {block_number}{title_suffix}',
        xaxis_title='Time (ms)',
        yaxis_title='Transaction',
        barmode='overlay',
        height=max(600, len(df) * 20),  # Dynamic height based on number of transactions
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Reverse y-axis to show first transaction at top
    fig.update_yaxes(autorange="reversed")
    
    return fig

def create_summary_charts(df: pd.DataFrame, block_number: int):
    """Create summary charts for the block's timing data."""
    
    # Calculate total time per transaction
    df['total_time'] = df['IO_time'] + df['EVM_time']
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Distribution', 'IO vs EVM Time', 
                       'Transaction Times (sorted)', 'Time Percentages'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # 1. Histogram of total times
    fig.add_trace(
        go.Histogram(x=df['total_time'], name='Total Time', nbinsx=30),
        row=1, col=1
    )
    
    # 2. Scatter plot of IO vs EVM time
    fig.add_trace(
        go.Scatter(
            x=df['IO_time'], 
            y=df['EVM_time'], 
            mode='markers',
            name='Transactions',
            text=df['tx_hash'],
            hovertemplate='<b>%{text}</b><br>IO: %{x:.2f} ms<br>EVM: %{y:.2f} ms<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Sorted bar chart of transaction times
    df_sorted = df.sort_values('total_time', ascending=False).head(20)
    
    fig.add_trace(
        go.Bar(x=df_sorted['tx_hash'], y=df_sorted['IO_time'], name='IO Time'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df_sorted['tx_hash'], y=df_sorted['EVM_time'], name='EVM Time'),
        row=2, col=1
    )
    
    # 4. Pie chart of total IO vs EVM time
    total_io = df['IO_time'].sum()
    total_evm = df['EVM_time'].sum()
    
    fig.add_trace(
        go.Pie(
            labels=['IO Time', 'EVM Time'],
            values=[total_io, total_evm],
            hole=0.3
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Block {block_number} Timing Analysis',
        height=800,
        showlegend=True,
        barmode='stack'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="IO Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="EVM Time (ms)", row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    return fig

def main():
    # Get list of available blocks
    blocks = list_available_blocks(BUCKET_NAME, limit=10)
    
    if not blocks:
        print("No blocks found in GCS")
        return
    
    # Use the highest (most recent) block
    latest_block = blocks[0]
    print(f"\nProcessing block {latest_block}...")
    
    # Download block data
    df = download_block_data(BUCKET_NAME, latest_block)
    print(f"Downloaded {len(df)} transactions")
    
    # Print summary statistics
    print(f"\nSummary for block {latest_block}:")
    print(f"Total transactions: {len(df)}")
    print(f"Average IO time: {df['IO_time'].mean():.2f} ms")
    print(f"Average EVM time: {df['EVM_time'].mean():.2f} ms")
    print(f"Total block time: {(df['IO_time'] + df['EVM_time']).sum():.2f} ms")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create Gantt chart
    gantt_fig = create_gantt_chart(df, latest_block)
    gantt_fig.write_html(f"block_{latest_block}_gantt.html")
    print(f"Saved Gantt chart to block_{latest_block}_gantt.html")
    
    # Create summary charts
    summary_fig = create_summary_charts(df, latest_block)
    summary_fig.write_html(f"block_{latest_block}_summary.html")
    print(f"Saved summary charts to block_{latest_block}_summary.html")
    
    # Optional: Process multiple blocks for comparison
    print("\nWould you like to analyze multiple blocks? (y/n): ", end='')
    if input().lower() == 'y':
        num_blocks = min(5, len(blocks))
        
        all_blocks_data = []
        for block_num in blocks[:num_blocks]:
            df = download_block_data(BUCKET_NAME, block_num)
            df['block_number'] = block_num
            all_blocks_data.append(df)
        
        combined_df = pd.concat(all_blocks_data, ignore_index=True)
        
        # Create comparison visualization
        fig = px.box(
            combined_df, 
            x='block_number', 
            y='IO_time',
            title='IO Time Distribution Across Blocks'
        )
        fig.write_html("blocks_comparison.html")
        print("Saved comparison chart to blocks_comparison.html")

if __name__ == "__main__":
    main()
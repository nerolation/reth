#!/usr/bin/env python3
"""
Advanced visualization with transaction hash mapping and interactive Gantt charts.

Requirements:
  pip install pandas google-cloud-storage pyarrow plotly kaleido
"""

import os
import pandas as pd
from google.cloud import storage
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Set GCP credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google-creds.json"

# Configuration
BUCKET_NAME = 'ethereum-execution-times'

class TimingVisualizer:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def find_latest_blocks(self, limit: int = 10) -> list:
        """Find the latest blocks available in GCS."""
        blocks = []
        for blob in self.bucket.list_blobs():
            if blob.name.endswith('.parquet'):
                try:
                    block_num = int(blob.name.replace('.parquet', ''))
                    blocks.append(block_num)
                except ValueError:
                    continue
        
        blocks.sort(reverse=True)
        return blocks[:limit] if limit else blocks
    
    def download_block(self, block_number: int) -> pd.DataFrame:
        """Download block data from GCS."""
        blob = self.bucket.blob(f"{block_number}.parquet")
        temp_file = f"/tmp/block_{block_number}.parquet"
        blob.download_to_filename(temp_file)
        df = pd.read_parquet(temp_file)
        os.remove(temp_file)
        return df
    
    def load_tx_mapping(self, mapping_file: str = None) -> dict:
        """Load transaction hash mapping from file.
        
        Format expected: {"tx_<block>_<index>": "0xActualTxHash..."}
        """
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}
    
    def join_tx_hashes(self, df: pd.DataFrame, tx_mapping: dict) -> pd.DataFrame:
        """Join actual transaction hashes to the dataframe."""
        if tx_mapping:
            df['actual_tx_hash'] = df['tx_hash'].map(tx_mapping)
            # Use actual hash if available, otherwise keep temporary identifier
            df['display_hash'] = df['actual_tx_hash'].fillna(df['tx_hash'])
            # Shorten for display
            df['short_hash'] = df['display_hash'].apply(
                lambda x: f"{x[:6]}...{x[-4:]}" if x.startswith('0x') else x
            )
        else:
            df['display_hash'] = df['tx_hash']
            df['short_hash'] = df['tx_hash']
        
        return df
    
    def create_interactive_gantt(self, df: pd.DataFrame, block_number: int, 
                                show_top_n: int = 100) -> go.Figure:
        """Create an interactive Gantt chart with transaction details."""
        
        # Sort by total time and limit
        df['total_time'] = df['IO_time'] + df['EVM_time']
        df_sorted = df.nlargest(show_top_n, 'total_time')
        
        # Calculate positions
        df_sorted = df_sorted.reset_index(drop=True)
        df_sorted['y_position'] = range(len(df_sorted))
        
        # Calculate cumulative timeline
        df_sorted['start_time'] = 0.0
        for i in range(1, len(df_sorted)):
            df_sorted.loc[i, 'start_time'] = df_sorted.loc[:i-1, 'total_time'].sum()
        
        fig = go.Figure()
        
        # Add IO phases
        io_data = df_sorted[df_sorted['IO_time'] > 0]
        fig.add_trace(go.Bar(
            y=io_data['y_position'],
            x=io_data['IO_time'],
            base=io_data['start_time'],
            name='IO Operations',
            orientation='h',
            marker_color='#E74C3C',
            text=io_data['short_hash'],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white'),
            hovertemplate=(
                '<b>Transaction:</b> %{text}<br>' +
                '<b>IO Time:</b> %{x:.2f} ms<br>' +
                '<b>Start:</b> %{base:.2f} ms<br>' +
                '<extra></extra>'
            )
        ))
        
        # Add EVM phases
        evm_data = df_sorted[df_sorted['EVM_time'] > 0].copy()
        evm_data['evm_start'] = evm_data['start_time'] + evm_data['IO_time']
        
        fig.add_trace(go.Bar(
            y=evm_data['y_position'],
            x=evm_data['EVM_time'],
            base=evm_data['evm_start'],
            name='EVM Execution',
            orientation='h',
            marker_color='#3498DB',
            text=evm_data['short_hash'],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white'),
            hovertemplate=(
                '<b>Transaction:</b> %{text}<br>' +
                '<b>EVM Time:</b> %{x:.2f} ms<br>' +
                '<b>Start:</b> %{base:.2f} ms<br>' +
                '<extra></extra>'
            )
        ))
        
        # Customize layout
        fig.update_layout(
            title=dict(
                text=f'Transaction Execution Timeline - Block {block_number}<br>' +
                     f'<sub>Showing top {show_top_n} transactions by total time</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Time (milliseconds)',
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title='Transactions',
                showticklabels=False,
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            barmode='overlay',
            height=max(600, show_top_n * 10),
            plot_bgcolor='rgba(240,240,240,0.3)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Add annotations for total block time
        total_time = df['total_time'].sum()
        fig.add_annotation(
            text=f"Total Block Execution Time: {total_time:.2f} ms",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        return fig
    
    def create_analysis_dashboard(self, df: pd.DataFrame, block_number: int) -> go.Figure:
        """Create a comprehensive analysis dashboard."""
        from plotly.subplots import make_subplots
        
        # Calculate metrics
        df['total_time'] = df['IO_time'] + df['EVM_time']
        df['io_percentage'] = (df['IO_time'] / df['total_time'] * 100).fillna(0)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Execution Time Distribution',
                'IO vs EVM Correlation',
                'Top 10 Slowest Transactions',
                'Time Breakdown by Phase',
                'IO Percentage Distribution',
                'Cumulative Time'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Histogram of total execution times
        fig.add_trace(
            go.Histogram(
                x=df['total_time'],
                nbinsx=50,
                name='Total Time',
                marker_color='#9B59B6',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Scatter plot: IO vs EVM time
        fig.add_trace(
            go.Scatter(
                x=df['IO_time'],
                y=df['EVM_time'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['total_time'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Total Time (ms)")
                ),
                text=df['short_hash'],
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'IO: %{x:.2f} ms<br>' +
                    'EVM: %{y:.2f} ms<br>' +
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Top 10 slowest transactions
        top_10 = df.nlargest(10, 'total_time')
        fig.add_trace(
            go.Bar(
                x=top_10['short_hash'],
                y=top_10['IO_time'],
                name='IO Time',
                marker_color='#E74C3C'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=top_10['short_hash'],
                y=top_10['EVM_time'],
                name='EVM Time',
                marker_color='#3498DB'
            ),
            row=2, col=1
        )
        
        # 4. Overall time breakdown
        total_io = df['IO_time'].sum()
        total_evm = df['EVM_time'].sum()
        fig.add_trace(
            go.Pie(
                labels=['IO Operations', 'EVM Execution'],
                values=[total_io, total_evm],
                hole=0.4,
                marker_colors=['#E74C3C', '#3498DB'],
                textinfo='label+percent',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. IO percentage distribution
        fig.add_trace(
            go.Histogram(
                x=df['io_percentage'],
                nbinsx=30,
                name='IO %',
                marker_color='#E67E22',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Cumulative time curve
        df_sorted = df.sort_values('total_time')
        df_sorted['cumulative_time'] = df_sorted['total_time'].cumsum()
        df_sorted['tx_percentile'] = (range(1, len(df_sorted) + 1) / len(df_sorted)) * 100
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted['tx_percentile'],
                y=df_sorted['cumulative_time'],
                mode='lines',
                line=dict(color='#27AE60', width=3),
                fill='tozeroy',
                fillcolor='rgba(39, 174, 96, 0.3)',
                name='Cumulative Time',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Total Time (ms)", row=1, col=1)
        fig.update_xaxes(title_text="IO Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="EVM Time (ms)", row=1, col=2)
        fig.update_xaxes(title_text="Transaction", tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_xaxes(title_text="IO Percentage (%)", row=3, col=1)
        fig.update_xaxes(title_text="Transaction Percentile (%)", row=3, col=2)
        fig.update_yaxes(title_text="Cumulative Time (ms)", row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Block {block_number} Execution Analysis<br>' +
                     f'<sub>{len(df)} transactions | Total time: {df["total_time"].sum():.2f} ms</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=1200,
            showlegend=True,
            barmode='stack',
            template='plotly_white'
        )
        
        return fig

def main():
    visualizer = TimingVisualizer(BUCKET_NAME)
    
    # Find latest blocks
    print("Finding latest blocks in GCS...")
    blocks = visualizer.find_latest_blocks(limit=5)
    
    if not blocks:
        print("No blocks found!")
        return
    
    print(f"Found {len(blocks)} recent blocks: {blocks}")
    
    # Process the latest block
    latest_block = blocks[0]
    print(f"\nProcessing block {latest_block}...")
    
    # Download data
    df = visualizer.download_block(latest_block)
    print(f"Downloaded {len(df)} transactions")
    
    # Optional: Load transaction hash mapping
    # tx_mapping = visualizer.load_tx_mapping('tx_mapping.json')
    # df = visualizer.join_tx_hashes(df, tx_mapping)
    # For now, use placeholder mapping
    df = visualizer.join_tx_hashes(df, {})
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Interactive Gantt chart
    gantt_fig = visualizer.create_interactive_gantt(df, latest_block, show_top_n=50)
    gantt_fig.write_html(f"block_{latest_block}_gantt_interactive.html")
    print(f"✓ Saved interactive Gantt chart")
    
    # Analysis dashboard
    dashboard_fig = visualizer.create_analysis_dashboard(df, latest_block)
    dashboard_fig.write_html(f"block_{latest_block}_dashboard.html")
    print(f"✓ Saved analysis dashboard")
    
    # Export data summary
    summary = {
        'block_number': latest_block,
        'total_transactions': len(df),
        'total_execution_time_ms': float(df['IO_time'].sum() + df['EVM_time'].sum()),
        'avg_io_time_ms': float(df['IO_time'].mean()),
        'avg_evm_time_ms': float(df['EVM_time'].mean()),
        'max_io_time_ms': float(df['IO_time'].max()),
        'max_evm_time_ms': float(df['EVM_time'].max()),
        'io_time_percentage': float(df['IO_time'].sum() / (df['IO_time'].sum() + df['EVM_time'].sum()) * 100)
    }
    
    with open(f'block_{latest_block}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics")
    
    print(f"\nAnalysis complete! Check the generated HTML files.")

if __name__ == "__main__":
    main()
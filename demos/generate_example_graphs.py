#!/usr/bin/env python3
"""
Generate Example Graphs for PCA Dashboard Demo
Creates sample visualizations showcasing dashboard capabilities
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

# Set plotly to generate static images
try:
    import kaleido
    if pio.kaleido.scope:
        pio.kaleido.scope.mathjax = None
except:
    pass  # Kaleido may not be available or configured

def setup_demo_data():
    """Generate sample stock data for demonstrations"""
    print("📊 Generating sample stock data...")
    
    # Tech stocks for demo
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate synthetic data as primary approach
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    stock_data = {}
    
    for i, symbol in enumerate(symbols):
        # Use different seeds for different stocks
        np.random.seed((hash(symbol) % 1000) + i * 10)
        
        # Generate more realistic stock price data
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with drift
        
        # Add some correlation between stocks (market factor)
        np.random.seed(42)  # Common market factor
        market_factor = np.random.normal(0, 0.01, n_days)
        returns += market_factor * 0.5  # 50% correlation with market
        
        # Generate prices from returns
        initial_price = 100 + i * 50  # Different starting prices
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        stock_data[symbol] = pd.Series(prices, index=dates)
    
    # Create DataFrame
    price_df = pd.DataFrame(stock_data)
    returns_df = price_df.pct_change().dropna()
    
    print(f"✅ Generated data for {len(symbols)} stocks over {len(price_df)} days")
    return price_df, returns_df, symbols

def generate_pca_analysis_graph(returns_df, symbols, save_path):
    """Generate PCA analysis visualization"""
    print("🎯 Generating PCA Analysis Graph...")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_df)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_returns)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'PCA Scatter Plot (PC1 vs PC2)',
            'Explained Variance Ratio', 
            'PCA Components Heatmap',
            'Cumulative Explained Variance'
        ],
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # PCA Scatter Plot
    colors = px.colors.qualitative.Set1[:len(symbols)]
    for i, symbol in enumerate(symbols):
        symbol_mask = np.arange(len(pca_result)) % len(symbols) == i
        fig.add_trace(
            go.Scatter(
                x=pca_result[symbol_mask, 0],
                y=pca_result[symbol_mask, 1],
                mode='markers',
                name=symbol,
                marker=dict(color=colors[i], size=8, opacity=0.7)
            ),
            row=1, col=1
        )
    
    # Explained Variance
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            name='Explained Variance',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # PCA Components Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pca.components_,
            x=symbols,
            y=[f'PC{i+1}' for i in range(pca.n_components_)],
            colorscale='RdBu',
            name='Components'
        ),
        row=2, col=1
    )
    
    # Cumulative Explained Variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    fig.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumsum))],
            y=cumsum,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='red', width=3)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Advanced PCA Analysis - Tech Stock Portfolio",
        title_x=0.5,
        showlegend=True,
        height=800,
        width=1200
    )
    
    # Save as PNG
    fig.write_image(save_path, width=1200, height=800, scale=2)
    print(f"✅ PCA Analysis saved to: {save_path}")

def generate_lstm_prediction_graph(price_df, symbols, save_path):
    """Generate LSTM-style prediction visualization (simulated)"""
    print("🧠 Generating LSTM Prediction Graph...")
    
    # Create simulated LSTM predictions
    fig = make_subplots(
        rows=len(symbols), cols=1,
        subplot_titles=[f'{symbol} - LSTM Price Prediction' for symbol in symbols],
        vertical_spacing=0.08
    )
    
    for i, symbol in enumerate(symbols):
        if symbol in price_df.columns:
            prices = price_df[symbol].dropna()
            
            # Split data for training/testing simulation
            split_idx = int(len(prices) * 0.8)
            train_data = prices[:split_idx]
            test_data = prices[split_idx:]
            
            # Simulate LSTM predictions (add some realistic noise)
            np.random.seed(42)
            predictions = test_data * (1 + np.random.normal(0, 0.05, len(test_data)))
            
            # Plot actual prices
            fig.add_trace(
                go.Scatter(
                    x=train_data.index,
                    y=train_data.values,
                    mode='lines',
                    name=f'{symbol} - Training Data',
                    line=dict(color='blue', width=2)
                ),
                row=i+1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=test_data.index,
                    y=test_data.values,
                    mode='lines',
                    name=f'{symbol} - Actual',
                    line=dict(color='green', width=2)
                ),
                row=i+1, col=1
            )
            
            # Plot predictions
            fig.add_trace(
                go.Scatter(
                    x=test_data.index,
                    y=predictions.values,
                    mode='lines',
                    name=f'{symbol} - LSTM Prediction',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=i+1, col=1
            )
            
            # Add prediction confidence interval
            upper_bound = predictions * 1.05
            lower_bound = predictions * 0.95
            
            fig.add_trace(
                go.Scatter(
                    x=test_data.index.tolist() + test_data.index[::-1].tolist(),
                    y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{symbol} Confidence'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title_text="LSTM Stock Price Predictions - Deep Learning Analysis",
        title_x=0.5,
        height=300 * len(symbols),
        width=1200,
        showlegend=True
    )
    
    # Save as PNG
    fig.write_image(save_path, width=1200, height=300 * len(symbols), scale=2)
    print(f"✅ LSTM Predictions saved to: {save_path}")

def generate_garch_volatility_graph(returns_df, symbols, save_path):
    """Generate GARCH volatility analysis visualization"""
    print("📈 Generating GARCH Volatility Graph...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Rolling Volatility (30-day)',
            'Volatility Distribution',
            'Returns vs Volatility Scatter',
            'Volatility Clustering'
        ],
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    colors = px.colors.qualitative.Set1[:len(symbols)]
    
    for i, symbol in enumerate(symbols):
        if symbol in returns_df.columns:
            returns = returns_df[symbol].dropna()
            
            # Calculate rolling volatility (30-day)
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            
            # Rolling volatility plot
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name=f'{symbol} Volatility',
                    line=dict(color=colors[i], width=2)
                ),
                row=1, col=1
            )
            
            # Volatility distribution
            fig.add_trace(
                go.Histogram(
                    x=rolling_vol.dropna().values,
                    name=f'{symbol} Vol Dist',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=colors[i]
                ),
                row=1, col=2
            )
            
            # Returns vs Volatility scatter
            valid_data = pd.concat([returns, rolling_vol], axis=1).dropna()
            if len(valid_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=valid_data.iloc[:, 1],  # volatility
                        y=valid_data.iloc[:, 0],  # returns
                        mode='markers',
                        name=f'{symbol} Returns vs Vol',
                        marker=dict(color=colors[i], size=6, opacity=0.6)
                    ),
                    row=2, col=1
                )
            
            # Volatility clustering (absolute returns)
            abs_returns = np.abs(returns)
            fig.add_trace(
                go.Scatter(
                    x=abs_returns.index,
                    y=abs_returns.values,
                    mode='lines',
                    name=f'{symbol} |Returns|',
                    line=dict(color=colors[i], width=1),
                    opacity=0.7
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title_text="GARCH Volatility Analysis - Advanced Risk Metrics",
        title_x=0.5,
        height=800,
        width=1200,
        showlegend=True
    )
    
    # Save as PNG
    fig.write_image(save_path, width=1200, height=800, scale=2)
    print(f"✅ GARCH Volatility saved to: {save_path}")

def generate_correlation_matrix_graph(returns_df, symbols, save_path):
    """Generate correlation matrix heatmap"""
    print("🔗 Generating Correlation Matrix Graph...")
    
    # Calculate correlation matrix
    corr_matrix = returns_df[symbols].corr()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Stock Returns Correlation Matrix', 'Hierarchical Clustering'],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True
        ),
        row=1, col=1
    )
    
    # Create distance matrix for clustering
    distance_matrix = 1 - np.abs(corr_matrix)
    
    fig.add_trace(
        go.Heatmap(
            z=distance_matrix.values,
            x=distance_matrix.columns,
            y=distance_matrix.columns,
            colorscale='Viridis',
            text=np.round(distance_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Stock Correlation Analysis - Market Relationships",
        title_x=0.5,
        height=600,
        width=1200
    )
    
    # Save as PNG
    fig.write_image(save_path, width=1200, height=600, scale=2)
    print(f"✅ Correlation Matrix saved to: {save_path}")

def generate_dashboard_overview_graph(price_df, returns_df, symbols, save_path):
    """Generate dashboard overview with multiple panels"""
    print("📊 Generating Dashboard Overview...")
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Stock Price Performance',
            'Daily Returns Distribution',
            'Cumulative Returns',
            'Risk-Return Scatter',
            'Trading Volume Simulation',
            'Portfolio Performance Summary'
        ],
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    colors = px.colors.qualitative.Set1[:len(symbols)]
    
    # Calculate metrics for summary
    summary_data = []
    
    for i, symbol in enumerate(symbols):
        if symbol in price_df.columns and symbol in returns_df.columns:
            prices = price_df[symbol].dropna()
            returns = returns_df[symbol].dropna()
            
            # Normalize prices to start at 100
            normalized_prices = (prices / prices.iloc[0]) * 100
            
            # Stock price performance
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices.values,
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(color=colors[i], width=2)
                ),
                row=1, col=1
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(
                    x=returns.values * 100,  # Convert to percentage
                    name=f'{symbol} Returns',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color=colors[i]
                ),
                row=1, col=2
            )
            
            # Cumulative returns
            cum_returns = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    mode='lines',
                    name=f'{symbol} Cum Returns',
                    line=dict(color=colors[i], width=2)
                ),
                row=2, col=1
            )
            
            # Risk-return scatter
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=[annual_vol],
                    y=[annual_return],
                    mode='markers+text',
                    text=[symbol],
                    textposition="top center",
                    name=f'{symbol} Risk-Return',
                    marker=dict(color=colors[i], size=15)
                ),
                row=2, col=2
            )
            
            # Simulated volume
            np.random.seed(42)
            volume = np.random.lognormal(15, 0.5, len(prices[-30:]))  # Last 30 days
            fig.add_trace(
                go.Bar(
                    x=prices.index[-30:],
                    y=volume,
                    name=f'{symbol} Volume',
                    marker_color=colors[i],
                    opacity=0.7
                ),
                row=3, col=1
            )
            
            # Summary statistics
            summary_data.append([
                symbol,
                f"{annual_return:.2%}",
                f"{annual_vol:.2%}",
                f"{annual_return/annual_vol:.2f}" if annual_vol != 0 else "N/A",
                f"{prices.iloc[-1]/prices.iloc[0]:.2f}x"
            ])
    
    # Add summary table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Stock', 'Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Total Return'],
                fill_color='lightblue',
                align='center'
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color='lavender',
                align='center'
            )
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        title_text="PCA Dashboard - Comprehensive Stock Analysis Overview",
        title_x=0.5,
        height=1200,
        width=1400,
        showlegend=True
    )
    
    # Save as PNG
    fig.write_image(save_path, width=1400, height=1200, scale=2)
    print(f"✅ Dashboard Overview saved to: {save_path}")

def main():
    """Generate all example graphs for demos"""
    print("🚀 PCA Dashboard - Example Graph Generator")
    print("=" * 50)
    
    # Create output directory
    demo_dir = Path(__file__).parent
    
    try:
        # Setup demo data
        price_df, returns_df, symbols = setup_demo_data()
        
        # Generate all graphs
        graphs_to_generate = [
            ("dashboard_overview.png", generate_dashboard_overview_graph),
            ("pca_analysis.png", generate_pca_analysis_graph),
            ("lstm_predictions.png", generate_lstm_prediction_graph),
            ("garch_volatility.png", generate_garch_volatility_graph),
            ("correlation_matrix.png", generate_correlation_matrix_graph),
        ]
        
        for filename, generator_func in graphs_to_generate:
            save_path = demo_dir / filename
            try:
                if generator_func == generate_dashboard_overview_graph:
                    generator_func(price_df, returns_df, symbols, str(save_path))
                elif generator_func in [generate_pca_analysis_graph, generate_garch_volatility_graph, generate_correlation_matrix_graph]:
                    generator_func(returns_df, symbols, str(save_path))
                elif generator_func == generate_lstm_prediction_graph:
                    generator_func(price_df, symbols, str(save_path))
                    
            except Exception as e:
                print(f"❌ Error generating {filename}: {e}")
                continue
        
        print("\n" + "=" * 50)
        print("✅ Example graph generation completed!")
        print("📁 Generated files:")
        for filename, _ in graphs_to_generate:
            filepath = demo_dir / filename
            if filepath.exists():
                print(f"   ✅ {filename}")
            else:
                print(f"   ❌ {filename} (failed)")
                
        print(f"\n📂 All graphs saved in: {demo_dir}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
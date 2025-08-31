#!/usr/bin/env python3
"""
Sample Data Generator for Dashboard Demo

Creates synthetic data that mimics real stock market data
for demonstration purposes when Yahoo Finance is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_stock_data(symbols, days=252, start_date=None):
    """
    Generate realistic sample stock data with correlations
    
    Args:
        symbols: List of stock symbols
        days: Number of trading days to generate
        start_date: Starting date for data generation
    
    Returns:
        DataFrame with stock prices and returns
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, periods=days)
    
    # Base parameters for realistic stock movement
    np.random.seed(42)  # For reproducible results
    
    # Stock characteristics (annual volatility, drift)
    stock_params = {
        'AAPL': {'vol': 0.25, 'drift': 0.12, 'start_price': 150},
        'MSFT': {'vol': 0.22, 'drift': 0.10, 'start_price': 300},
        'GOOGL': {'vol': 0.28, 'drift': 0.08, 'start_price': 2500},
        'AMZN': {'vol': 0.30, 'drift': 0.15, 'start_price': 3000},
        'META': {'vol': 0.35, 'drift': 0.05, 'start_price': 200},
        'TSLA': {'vol': 0.45, 'drift': 0.20, 'start_price': 800},
        'NVDA': {'vol': 0.40, 'drift': 0.25, 'start_price': 400},
        'JPM': {'vol': 0.20, 'drift': 0.08, 'start_price': 120},
        'JNJ': {'vol': 0.15, 'drift': 0.06, 'start_price': 165}
    }
    
    # Generate correlated returns
    n_stocks = len(symbols)
    correlation_matrix = generate_correlation_matrix(symbols)
    
    # Generate random returns with correlation structure
    returns = np.random.multivariate_normal(
        mean=[0] * n_stocks,
        cov=correlation_matrix,
        size=days
    )
    
    # Convert to DataFrame
    data = {}
    
    for i, symbol in enumerate(symbols):
        params = stock_params.get(symbol, {'vol': 0.25, 'drift': 0.10, 'start_price': 100})
        
        # Scale returns by volatility and add drift
        daily_returns = returns[:, i] * params['vol'] / np.sqrt(252) + params['drift'] / 252
        
        # Generate price series using cumulative returns
        prices = [params['start_price']]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        data[symbol] = prices[1:]  # Remove initial price
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df

def generate_correlation_matrix(symbols):
    """Generate realistic correlation matrix for stocks"""
    n = len(symbols)
    
    # Base correlation (all stocks somewhat correlated to market)
    base_corr = 0.3
    
    # Sector correlations
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    finance_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS']
    
    # Initialize correlation matrix
    corr_matrix = np.full((n, n), base_corr)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Increase correlation within sectors
    for i, stock1 in enumerate(symbols):
        for j, stock2 in enumerate(symbols):
            if i != j:
                # Tech sector correlation
                if stock1 in tech_stocks and stock2 in tech_stocks:
                    corr_matrix[i, j] = 0.6
                # Finance sector correlation
                elif stock1 in finance_stocks and stock2 in finance_stocks:
                    corr_matrix[i, j] = 0.7
    
    # Ensure positive definite matrix
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    return corr_matrix

def create_sample_pca_results():
    """Generate sample PCA analysis results"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Sample PCA results
    pca_results = {
        'explained_variance_ratio': [0.4, 0.25, 0.18, 0.12, 0.05],
        'cumulative_variance': [0.4, 0.65, 0.83, 0.95, 1.0],
        'components': np.random.randn(5, 5),
        'symbols': symbols
    }
    
    return pca_results

def create_sample_garch_results():
    """Generate sample GARCH volatility results"""
    dates = pd.bdate_range(start='2023-01-01', end='2023-12-31')
    
    # Simulate volatility clustering
    np.random.seed(42)
    volatility = []
    vol = 0.02
    
    for _ in range(len(dates)):
        vol = 0.000001 + 0.05 * vol + 0.92 * vol + 0.000001 * np.random.randn()**2
        volatility.append(np.sqrt(vol))
    
    garch_results = {
        'dates': dates,
        'volatility_forecast': volatility,
        'var_5': [-0.03] * len(dates),  # 5% VaR
        'var_1': [-0.05] * len(dates),  # 1% VaR
    }
    
    return garch_results

def save_sample_visualizations():
    """Create and save sample visualization plots"""
    print("📊 Generating sample visualizations...")
    
    # 1. Sample PCA scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PCA 2D scatter
    x = np.random.randn(5)
    y = np.random.randn(5)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    axes[0, 0].scatter(x, y, s=100, alpha=0.7)
    for i, symbol in enumerate(symbols):
        axes[0, 0].annotate(symbol, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    axes[0, 0].set_title('PCA Component Analysis')
    axes[0, 0].set_xlabel('PC1 (40% variance)')
    axes[0, 0].set_ylabel('PC2 (25% variance)')
    
    # Correlation heatmap
    corr_data = generate_correlation_matrix(symbols)
    im = axes[0, 1].imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Stock Correlation Matrix')
    axes[0, 1].set_xticks(range(len(symbols)))
    axes[0, 1].set_yticks(range(len(symbols)))
    axes[0, 1].set_xticklabels(symbols)
    axes[0, 1].set_yticklabels(symbols)
    
    # Volatility forecast
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    vol = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, 100)) + 0.005 * np.random.randn(100)
    axes[1, 0].plot(dates, vol, 'b-', alpha=0.7)
    axes[1, 0].set_title('GARCH Volatility Forecast')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Sample price prediction
    actual = 150 + 10 * np.cumsum(np.random.randn(50) * 0.1)
    predicted = actual + np.random.randn(50) * 2
    
    axes[1, 1].plot(range(50), actual, 'b-', label='Actual', alpha=0.8)
    axes[1, 1].plot(range(50), predicted, 'r--', label='LSTM Prediction', alpha=0.8)
    axes[1, 1].set_title('LSTM Price Prediction')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('/Users/elhoyabembe/Documents/GitHub/Investalogical_Final/PCA_Dashboard/demos/sample_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Sample visualization saved as 'sample_analysis_charts.png'")

def main():
    """Generate and display sample data"""
    print("🎯 Generating Sample Data for PCA Dashboard Demo")
    print("=" * 60)
    
    # Generate sample stock data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    stock_data = generate_sample_stock_data(symbols, days=252)
    
    print(f"📊 Generated {len(stock_data)} days of stock data")
    print(f"📈 Stocks: {', '.join(symbols)}")
    print("\nSample data preview:")
    print(stock_data.head())
    
    # Save sample data
    stock_data.to_csv('/Users/elhoyabembe/Documents/GitHub/Investalogical_Final/PCA_Dashboard/demos/sample_stock_data.csv')
    print("\n💾 Sample data saved as 'sample_stock_data.csv'")
    
    # Generate sample analysis results
    pca_results = create_sample_pca_results()
    garch_results = create_sample_garch_results()
    
    print(f"\n📊 PCA explains {pca_results['cumulative_variance'][2]:.1%} variance with 3 components")
    print(f"📈 GARCH volatility range: {min(garch_results['volatility_forecast']):.3f} - {max(garch_results['volatility_forecast']):.3f}")
    
    # Create sample visualizations
    save_sample_visualizations()
    
    print("\n" + "=" * 60)
    print("📋 Demo files created in demos/ folder:")
    print("   • sample_stock_data.csv - Historical price data")
    print("   • sample_analysis_charts.png - Analysis visualizations") 
    print("   • Use these files to understand dashboard capabilities")
    print("=" * 60)

if __name__ == "__main__":
    main()
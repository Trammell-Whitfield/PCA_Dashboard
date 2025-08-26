# Enhanced PCA Stock Analysis Dashboard with Advanced Features
# Integrating volatility analysis, economic indicators, and improved error handling

import datetime
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, callback_context, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from scipy.stats import normaltest, jarque_bera, zscore
from scipy import stats
import warnings
import traceback
import logging
from typing import Tuple, Dict, Optional, Any

# Configure logging first (needed for imports)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Enhanced VAR/GARCH imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'VAR'))
try:
    from GARCH_update import (
        IntegratedVARGARCHAnalysis, 
        VARGARCHVisualizer, 
        run_var_garch_analysis,
        create_interpretation_cards
    )
    ADVANCED_GARCH_AVAILABLE = True
    logger.info("Advanced GARCH module loaded successfully")
except ImportError as e:
    logger.warning(f"Advanced GARCH module not available: {e}")
    ADVANCED_GARCH_AVAILABLE = False
    # Fallback imports
    from statsmodels.tsa.api import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    from scipy.optimize import minimize

# Enhanced Time Series Analysis Import
try:
    from time_series_analysis import create_enhanced_timeseries_tab, EnhancedTimeSeriesAnalyzer
    ENHANCED_TIMESERIES_AVAILABLE = True
    logger.info("Enhanced Time Series Analysis module loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced Time Series Analysis module not available: {e}")
    ENHANCED_TIMESERIES_AVAILABLE = False

# LSTM imports - Updated for new enhanced API
try:
    from LSTM_Pred import EnhancedLSTMPredictor, run_enhanced_model_demo
    LSTM_AVAILABLE = True
    logger.info("Enhanced LSTM module loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced LSTM module not available: {e}")
    LSTM_AVAILABLE = False

# Initialize configurations
pio.templates.default = "plotly_white"

# Enhanced styling
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Advanced PCA Stock Analysis Dashboard"

# Expose server for WSGI deployment (required for Render)
server = app.server

# Enhanced stock presets with more options
STOCK_PRESETS = {
    "Tech Giants": "AAPL,MSFT,GOOGL,AMZN,META",
    "Banking": "JPM,BAC,WFC,C,GS",
    "Blue Chips": "AAPL,MSFT,JNJ,PG,KO",
    "Energy": "XOM,CVX,COP,SLB,EOG",
    "Healthcare": "JNJ,PFE,UNH,MRK,ABBV",
    "Consumer": "KO,PEP,PG,WMT,HD",
    "Semiconductors": "NVDA,AMD,INTC,TSM,QCOM",
    "Electric Vehicles": "TSLA,NIO,XPEV,LI,LCID",
    "REITs": "VNQ,O,PLD,CCI,AMT",
    "Demo Data": "DEMO",
    "Custom": ""
}

# Enhanced economic indicators for comprehensive market analysis
ECONOMIC_INDICATORS = {
    "📈 Treasury Yields": {
        "tickers": ["^TNX", "^FVX", "^IRX"], 
        "description": "10Y, 5Y, 3M Treasury rates - fundamental interest rate indicators",
        "impact": "Higher yields often correlate with defensive rotation and reduced risk appetite"
    },
    "💱 Currency Markets": {
        "tickers": ["DX-Y.NYB", "EURUSD=X", "GBPUSD=X"], 
        "description": "Dollar Index, EUR/USD, GBP/USD - global currency strength indicators",
        "impact": "Dollar strength affects multinational company earnings and emerging markets"
    },
    "🥇 Commodities": {
        "tickers": ["GC=F", "CL=F", "SI=F"], 
        "description": "Gold, Oil, Silver - inflation hedges and economic activity indicators",
        "impact": "Commodity prices reflect inflation expectations and economic growth"
    },
    "😰 Market Volatility": {
        "tickers": ["^VIX", "^VIX9D"], 
        "description": "VIX indices - market fear and uncertainty gauges",
        "impact": "High volatility indicates market stress and affects all asset classes"
    },
    "🏦 Credit Markets": {
        "tickers": ["HYG", "LQD", "TLT"], 
        "description": "High Yield, Investment Grade, Long Treasury - credit risk indicators",
        "impact": "Credit spreads reveal risk appetite and economic outlook expectations"
    },
    "🏘️ Real Estate": {
        "tickers": ["^RUT", "VNQ", "XLRE"], 
        "description": "Russell 2000, REITs, Real Estate sector - domestic economic health",
        "impact": "Real estate performance reflects interest rate sensitivity and economic growth"
    }
}

class EnhancedVolatilityAnalyzer:
    """Enhanced volatility analysis integrated into PCA dashboard"""
    
    def __init__(self, volatility_window: int = 20):
        self.volatility_window = volatility_window
    
    def calculate_advanced_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple volatility measures"""
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=self.volatility_window).std()
        data['EW_Volatility'] = data['Daily_Return'].ewm(span=self.volatility_window).std()
        
        # High-Low volatility measures (if OHLC data available)
        if all(col in data.columns for col in ['High', 'Low', 'Open']):
            data['GK_Volatility'] = np.sqrt(
                0.5 * (np.log(data['High'] / data['Low']))**2 - 
                (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']))**2
            )
        
        if all(col in data.columns for col in ['High', 'Low']):
            data['Parkinson_Volatility'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(data['High'] / data['Low']))**2
            )
        
        return data
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {}
        
        # Remove extreme outliers (beyond 3 standard deviations)
        z_scores = np.abs(stats.zscore(returns_clean))
        returns_clean = returns_clean[z_scores < 3]
        
        if len(returns_clean) == 0:
            return {}
        
        # Basic metrics
        volatility = returns_clean.std() * np.sqrt(252)  # Annualized
        
        # Enhanced Sharpe ratio calculation
        annual_return = returns_clean.mean() * 252
        
        # Fix risk-free rate calculation for better Sharpe ratios
        # Use appropriate risk-free rate based on data characteristics
        if abs(annual_return) < 0.1:  # For factor scores or normalized data
            # Use a much smaller risk-free rate to avoid overwhelming small returns
            risk_free_rate = 0.001  # 0.1% - very conservative for factor analysis
        elif abs(annual_return) < 0.5:  # Small but meaningful returns
            risk_free_rate = 0.01   # 1% - reasonable for low returns
        else:
            risk_free_rate = 0.02   # 2% - standard for normal stock returns
        
        # Improved Sharpe ratio with better handling
        if volatility > 1e-6 and not np.isnan(annual_return) and not np.isnan(volatility):
            excess_return = annual_return - risk_free_rate
            sharpe_ratio = excess_return / volatility
            
            # For very small returns, consider using the return/volatility ratio directly
            # This gives a more meaningful measure for factor analysis
            if abs(annual_return) < 0.05:  # Very small returns (typical for factor scores)
                sharpe_ratio = annual_return / volatility  # Information ratio style
            
            # Ensure reasonable bounds for Sharpe ratio
            sharpe_ratio = np.clip(sharpe_ratio, -5, 5)
        else:
            sharpe_ratio = 0
        
        # VaR calculations (95% and 99%)
        var_95 = np.percentile(returns_clean, 5)
        var_99 = np.percentile(returns_clean, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns_clean[returns_clean <= var_95].mean()
        es_99 = returns_clean[returns_clean <= var_99].mean()
        
        # Maximum Drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Skewness and Kurtosis
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        
        # Downside deviation (semi-volatility)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = (returns_clean.mean() * 252) / downside_deviation if downside_deviation != 0 else 0
        
        return {
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'VaR_95': var_95,
            'VaR_99': var_99,
            'ES_95': es_95,
            'ES_99': es_99,
            'Max_Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Downside_Deviation': downside_deviation
        }

class EnhancedDataHandler:
    """Enhanced data handling with better error handling and validation"""
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate stock ticker and get company name"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try to get basic info
            if not info or 'longName' not in info:
                # Fallback: try to get some historical data
                test_data = stock.history(period="5d")
                if test_data.empty:
                    return False, ""
                company_name = info.get('longName', info.get('shortName', ticker))
            else:
                company_name = info['longName']
            
            return True, company_name
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False, ""
    
    @staticmethod
    def get_enhanced_stock_data(tickers, start_date, end_date):
        """Enhanced stock data fetching with comprehensive error handling"""
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            
            if len(ticker_list) < 2:
                raise ValueError("Please provide at least 2 tickers for meaningful PCA analysis")
            
            logger.info(f"Attempting to download: {ticker_list}")
            
            # Convert dates to string format
            start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
            # Strategy 1: Batch download with comprehensive parameters
            try:
                data = yf.download(
                    ticker_list, 
                    start=start_str, 
                    end=end_str, 
                    progress=False,
                    auto_adjust=True,  # Use adjusted prices
                    prepost=False,
                    threads=True,
                    ignore_tz=True,
                    group_by='ticker'  # Better organization for multiple tickers
                )
                
                if not data.empty:
                    logger.info(f"Batch download successful. Data shape: {data.shape}")
                    return EnhancedDataHandler._process_batch_data(data, ticker_list)
                    
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
            
            # Strategy 2: Individual ticker downloads
            logger.info("Trying individual ticker downloads...")
            individual_data = {}
            
            for ticker in ticker_list:
                try:
                    ticker_data = yf.download(
                        ticker,
                        start=start_str,
                        end=end_str,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if not ticker_data.empty and 'Close' in ticker_data.columns:
                        individual_data[ticker] = ticker_data
                        logger.info(f"Successfully downloaded {ticker}")
                    else:
                        logger.warning(f"No valid data for {ticker}")
                        
                except Exception as e:
                    logger.error(f"Failed to download {ticker}: {e}")
                    continue
            
            if individual_data:
                # Combine individual data
                combined_data = {}
                common_dates = None
                
                for ticker, data in individual_data.items():
                    if common_dates is None:
                        common_dates = data.index
                    else:
                        common_dates = common_dates.intersection(data.index)
                
                for ticker, data in individual_data.items():
                    combined_data[ticker] = data.loc[common_dates, 'Close']
                
                final_data = pd.DataFrame(combined_data)
                logger.info(f"Individual download successful. Combined data shape: {final_data.shape}")
                return EnhancedDataHandler._clean_data(final_data)
            
            # Strategy 3: Generate synthetic data for demonstration
            logger.warning("All download strategies failed. Generating synthetic data...")
            return EnhancedDataHandler._generate_synthetic_data(ticker_list, start_str, end_str)
            
        except Exception as e:
            logger.error(f"Error in get_enhanced_stock_data: {str(e)}")
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    @staticmethod
    def _process_batch_data(data, ticker_list):
        """Process batch downloaded data"""
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns
            price_data = {}
            for ticker in ticker_list:
                if ticker in data.columns.get_level_values(0):
                    ticker_data = data[ticker]
                    if 'Close' in ticker_data.columns:
                        price_data[ticker] = ticker_data['Close']
                    elif 'Adj Close' in ticker_data.columns:
                        price_data[ticker] = ticker_data['Adj Close']
            
            if price_data:
                final_data = pd.DataFrame(price_data)
                return EnhancedDataHandler._clean_data(final_data)
        
        # Single level columns (single ticker)
        if len(ticker_list) == 1 and 'Close' in data.columns:
            final_data = pd.DataFrame({ticker_list[0]: data['Close']})
            return EnhancedDataHandler._clean_data(final_data)
        
        raise ValueError("Unable to process batch data structure")
    
    @staticmethod
    def _clean_data(data):
        """Clean and validate data"""
        # Remove columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Remove rows with any NaN values for PCA consistency
        data = data.dropna()
        
        if data.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        logger.info(f"Cleaned data shape: {data.shape}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
    
    @staticmethod
    def _generate_synthetic_data(ticker_list, start_date, end_date):
        """Generate synthetic stock data for demonstration"""
        logger.info("Generating synthetic data for demonstration...")
        
        np.random.seed(42)  # Reproducible results
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # Business days only
        
        synthetic_data = {}
        base_price = 100
        
        for i, ticker in enumerate(ticker_list):
            n_days = len(dates)
            returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
            
            # Add market correlation
            if i > 0:
                market_factor = np.random.normal(0, 0.01, n_days)
                returns += market_factor * 0.3
            
            # Generate price series
            prices = [base_price + i * 20]  # Different starting prices
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            synthetic_data[ticker] = pd.Series(prices, index=dates)
        
        return pd.DataFrame(synthetic_data)

def calculate_enhanced_metrics(returns, pca_result, n_components):
    """Calculate comprehensive PCA metrics with risk analysis"""
    explained_var_ratio = pca_result.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca_result.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=returns.columns
    )
    
    # Calculate communalities
    communalities = np.sum(loadings**2, axis=1)
    
    # Calculate factor scores
    factor_scores = pd.DataFrame(
        pca_result.transform(returns),
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=returns.index
    )
    
    # Enhanced statistical tests and risk metrics
    vol_analyzer = EnhancedVolatilityAnalyzer()
    stock_metrics = {}
    
    for col in returns.columns:
        _, p_normal = normaltest(returns[col].dropna())
        risk_metrics = vol_analyzer.calculate_risk_metrics(returns[col])
        stock_metrics[col] = {
            'normality_p': p_normal,
            **risk_metrics
        }
    
    return {
        'explained_variance': explained_var_ratio,
        'loadings': loadings,
        'communalities': communalities,
        'factor_scores': factor_scores,
        'stock_metrics': stock_metrics,
        'volatility_analyzer': vol_analyzer
    }

# Enhanced UI Components
def create_enhanced_header():
    """Create enhanced header with more professional styling"""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3 text-primary"),
                    "Advanced PCA Stock Analysis Dashboard"
                ], className="text-dark mb-3"),
                html.P([
                    "🔬 Advanced portfolio analysis using Principal Component Analysis, enhanced volatility modeling, and risk assessment. ",
                    "Discover hidden market relationships, measure factor exposures, and optimize your investment strategy."
                ], className="lead text-muted"),
                dbc.Badge([
                    html.I(className="fas fa-star me-1"),
                    "Enhanced with Volatility & Risk Analysis"
                ], color="info", className="me-2"),
                dbc.Badge([
                    html.I(className="fas fa-brain me-1"),
                    "ML-Powered Insights"
                ], color="success", className="me-2"),
                dbc.Badge([
                    html.I(className="fas fa-shield-alt me-1"),
                    "Risk Management Tools"
                ], color="warning")
            ], className="text-center p-4 bg-light rounded mb-4")
        ])
    ])

def create_enhanced_controls():
    """Create enhanced control panel with new features"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-cog me-2"), 
                "Analysis Configuration"
            ], className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            # Row 1: Presets and Components
            dbc.Row([
                dbc.Col([
                    html.Label("📊 Stock Preset:", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id="preset-dropdown",
                        options=[{"label": f"📈 {k}", "value": v} for k, v in STOCK_PRESETS.items()],
                        value="",
                        placeholder="Select a preset or enter custom tickers",
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("🔢 Number of Components:", className="form-label fw-bold"),
                    dcc.Slider(
                        id="component-slider",
                        min=2, max=10, step=1, value=3,
                        marks={i: str(i) for i in range(2, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-3"
                    )
                ], width=6)
            ]),
            
            # Row 2: Tickers and Analysis Options
            dbc.Row([
                dbc.Col([
                    html.Label("🎯 Stock Tickers:", className="form-label fw-bold"),
                    dcc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL)",
                        className="form-control mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("⚙️ Analysis Options:", className="form-label fw-bold"),
                    dbc.Checklist(
                        id="analysis-options",
                        options=[
                            {"label": " Standardize Returns", "value": "standardize"},
                            {"label": " Include Volatility Analysis", "value": "volatility"},
                            {"label": " Risk Metrics", "value": "risk"}
                        ],
                        value=["standardize", "volatility", "risk"],
                        className="mb-3"
                    )
                ], width=6)
            ]),
            
            # Row 3: Date Range and Economic Indicators
            dbc.Row([
                dbc.Col([
                    html.Label("📅 Date Range:", className="form-label fw-bold"),
                    dcc.DatePickerRange(
                        id="date-picker",
                        start_date=datetime.datetime.now() - datetime.timedelta(days=365 * 2),
                        end_date=datetime.datetime.now() - datetime.timedelta(days=1),
                        display_format="YYYY-MM-DD",
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("🌍 Economic Indicators (Optional):", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id="economic-indicators",
                        options=[
                            {
                                "label": html.Div([
                                    html.Div(k, style={"fontWeight": "bold"}),
                                    html.Div(v["description"], style={"fontSize": "0.8em", "color": "gray"})
                                ]),
                                "value": k
                            } for k, v in ECONOMIC_INDICATORS.items()
                        ],
                        value=[],
                        multi=True,
                        placeholder="Select economic indicators for enhanced analysis",
                        className="mb-3",
                        style={"minHeight": "40px"}
                    ),
                    html.Div([
                        html.Small("💡 Economic indicators help identify macro factors affecting your portfolio", 
                                 className="text-info")
                    ], className="mb-2")
                ], width=6)
            ]),
            
            # Row 4: Action Button
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-rocket me-2"),
                        "🚀 Run Enhanced Analysis"
                    ], id="submit-button", color="primary", size="lg", className="w-100 shadow")
                ], width=12, className="text-center")
            ])
        ])
    ], className="mb-4 shadow")

# Update the results section to include new tabs
def create_enhanced_results_section():
    """Create enhanced results section with new analysis tabs"""
    return html.Div([
        # Enhanced Summary Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="total-variance-card", className="text-primary mb-1"),
                        html.P("📊 Total Variance Explained", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-primary shadow-sm")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="components-card", className="text-success mb-1"),
                        html.P("🔢 Principal Components", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-success shadow-sm")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="stocks-card", className="text-info mb-1"),
                        html.P("📈 Stocks Analyzed", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-info shadow-sm")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="period-card", className="text-warning mb-1"),
                        html.P("📅 Analysis Period", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-warning shadow-sm")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="avg-volatility-card", className="text-danger mb-1"),
                        html.P("📊 Avg Volatility", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-danger shadow-sm")
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(id="sharpe-card", className="text-secondary mb-1"),
                        html.P("⚡ Avg Sharpe Ratio", className="text-muted mb-0 small")
                    ])
                ], className="text-center border-secondary shadow-sm")
            ], width=2)
        ], className="mb-4"),
        
        # Enhanced Tabs Section
        dbc.Tabs([
            dbc.Tab(label="📊 Variance Analysis", tab_id="variance-tab", 
                   tab_class_name="fw-bold"),
            dbc.Tab(label="🎯 Factor Loadings", tab_id="loadings-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="📈 Time Series", tab_id="timeseries-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="🔥 Volatility Analysis", tab_id="volatility-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="⚠️ Risk Assessment", tab_id="risk-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="🚀 VAR/GARCH Analysis", tab_id="vargarch-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="🧠 LSTM Predictions", tab_id="lstm-tab",
                   tab_class_name="fw-bold"),
            dbc.Tab(label="🔍 Detailed Analysis", tab_id="detailed-tab",
                   tab_class_name="fw-bold")
        ], id="analysis-tabs", active_tab="variance-tab", className="mb-3"),
        
        html.Div(id="tab-content", className="mt-4")
    ], id="results-section", style={"display": "none"})

# Continue with the rest of the application...
# [Rest of the code will continue in the next part due to length limits]

# Define missing UI components before layout
def create_error_alert():
    """Create enhanced error alert component"""
    return dbc.Alert(
        id="error-alert",
        color="danger",
        dismissable=True,
        is_open=False,
        className="mt-3 shadow-sm",
        style={"borderRadius": "12px"}
    )

def create_enhanced_loading_component():
    """Create sophisticated loading modal with progress tracking"""
    return dbc.Modal([
        dbc.ModalBody([
            html.Div([
                # Main spinner and title section
                html.Div([
                    dbc.Spinner(
                        size="lg", 
                        color="primary", 
                        spinner_style={
                            "width": "4rem", 
                            "height": "4rem",
                            "borderWidth": "4px"
                        }
                    ),
                    html.H4([
                        html.I(className="fas fa-microscope me-2"),
                        "Analyzing Your Portfolio"
                    ], className="mt-3 text-center text-primary fw-bold"),
                    html.P("Advanced financial analysis in progress...", 
                          className="text-center text-muted mb-0")
                ], className="text-center mb-4"),
                
                # Progress bar section
                html.Div([
                    dbc.Progress(
                        id="loading-progress", 
                        value=0,
                        striped=True, 
                        animated=True, 
                        color="info",
                        style={"height": "12px", "borderRadius": "10px"}
                    ),
                    html.Div(
                        id="loading-status", 
                        className="text-center mt-2 text-muted fw-bold"
                    ),
                ], className="mb-4"),
                
                # Processing steps visualization
                html.Div([
                    html.H6([
                        html.I(className="fas fa-cogs me-2"),
                        "Processing Steps:"
                    ], className="mb-3 text-primary"),
                    html.Div(id="loading-steps", children=[
                        html.Div([
                            html.I(className="fas fa-circle-notch fa-spin me-2 text-primary", id="step1-icon"),
                            html.Span("📊 Fetching market data...", id="step1-text", className="fw-bold")
                        ], className="mb-2 p-2 bg-light rounded", id="step1"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step2-icon"),
                            html.Span("🧮 Calculating enhanced metrics...", id="step2-text", className="text-muted")
                        ], className="mb-2 p-2", id="step2"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step3-icon"),
                            html.Span("🔬 Performing advanced analysis...", id="step3-text", className="text-muted")
                        ], className="mb-2 p-2", id="step3"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step4-icon"),
                            html.Span("📈 Creating visualizations...", id="step4-text", className="text-muted")
                        ], className="mb-2 p-2", id="step4"),
                    ])
                ], className="border rounded p-3 bg-light"),
                
                # Tips and estimated time section
                html.Div([
                    html.Div([
                        html.I(className="fas fa-clock me-1 text-info"),
                        html.Span(id="estimated-time", children="⏱️ Estimated time: 15-45 seconds")
                    ], className="text-muted mb-2"),
                    html.Div([
                        html.I(className="fas fa-lightbulb me-1 text-warning"),
                        html.Span("💡 Tip: More stocks and features = longer processing time")
                    ], className="text-info mb-2"),
                    html.Div([
                        html.I(className="fas fa-coffee me-1 text-success"),
                        html.Span("☕ Perfect time for a quick coffee break!")
                    ], className="text-success")
                ], className="mt-3 text-center small")
            ], className="p-4")
        ])
    ], 
    id="loading-modal", 
    is_open=False, 
    backdrop="static", 
    keyboard=False, 
    size="md",
    style={"borderRadius": "20px"}
    )

def add_interval_component():
    """Add interval component for real-time progress tracking"""
    return dcc.Interval(
        id='loading-interval',
        interval=500,  # Update every 500ms for smooth progress
        n_intervals=0,
        disabled=True,
        max_intervals=25  # Prevent infinite running
    )

# App Layout
app.layout = dbc.Container([
    create_enhanced_header(),
    create_enhanced_controls(),
    create_error_alert(),
    create_enhanced_loading_component(),
    add_interval_component(),
    create_enhanced_results_section()
], fluid=True, className="py-4")

# Callbacks and Enhanced Analysis Functions

# Global variable to store analysis results
pca_results = None

@app.callback(
    Output("ticker-input", "value"),
    Input("preset-dropdown", "value")
)
def update_ticker_input(preset_value):
    """Update ticker input when preset is selected"""
    if preset_value and preset_value != "Custom":
        return preset_value
    return ""

# Enhanced loading and analysis callback
@app.callback(
    [Output("loading-modal", "is_open"),
     Output("loading-progress", "value"),
     Output("loading-status", "children"),
     Output("step1-icon", "className"),
     Output("step1-text", "className"),
     Output("step2-icon", "className"),
     Output("step2-text", "className"),
     Output("step3-icon", "className"),
     Output("step3-text", "className"),
     Output("step4-icon", "className"),
     Output("step4-text", "className"),
     Output("loading-interval", "disabled"),
     Output("error-alert", "is_open"),
     Output("error-alert", "children"),
     Output("results-section", "style"),
     Output("total-variance-card", "children"),
     Output("components-card", "children"),
     Output("stocks-card", "children"),
     Output("period-card", "children"),
     Output("avg-volatility-card", "children"),
     Output("sharpe-card", "children")],
    [Input("submit-button", "n_clicks"),
     Input("loading-interval", "n_intervals")],
    [State("ticker-input", "value"),
     State("component-slider", "value"),
     State("date-picker", "start_date"),
     State("date-picker", "end_date"),
     State("analysis-options", "value"),
     State("economic-indicators", "value")],
    prevent_initial_call=True
)
def handle_enhanced_analysis(n_clicks, n_intervals, tickers, n_components, start_date, end_date, analysis_options, economic_indicators):
    """Enhanced analysis with new features"""
    
    # Default icon classes
    default_icon = "fas fa-circle me-2 text-muted"
    active_icon = "fas fa-circle-notch fa-spin me-2 text-primary"
    complete_icon = "fas fa-check-circle me-2 text-success"
    default_text = "text-muted"
    active_text = "text-primary fw-bold"
    complete_text = "text-success"
    
    # Default return values
    default_return = (False, 0, "", default_icon, default_text, default_icon, default_text, 
                     default_icon, default_text, default_icon, default_text, True,
                     False, "", {"display": "none"}, "", "", "", "", "", "")
    
    # Initial state
    if not n_clicks or not tickers:
        return default_return
    
    # Check which triggered the callback
    ctx_triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    # If submit button was clicked, start the loading process
    if ctx_triggered == 'submit-button.n_clicks':
        return (True, 25, "🔍 Fetching market data...", 
                active_icon, active_text, default_icon, default_text, 
                default_icon, default_text, default_icon, default_text, False,
                False, "", {"display": "none"}, "", "", "", "", "", "")
    
    # Handle interval updates for progress tracking
    if ctx_triggered == 'loading-interval.n_intervals':
        if n_intervals < 5:  # Step 1: Fetching data
            return (True, 25, "📊 Fetching stock and economic data...", 
                    active_icon, active_text, default_icon, default_text, 
                    default_icon, default_text, default_icon, default_text, False,
                    False, "", {"display": "none"}, "", "", "", "", "", "")
        elif n_intervals < 10:  # Step 2: Processing returns
            return (True, 50, "🧮 Processing returns and volatility...", 
                    complete_icon, complete_text, active_icon, active_text, 
                    default_icon, default_text, default_icon, default_text, False,
                    False, "", {"display": "none"}, "", "", "", "", "", "")
        elif n_intervals < 15:  # Step 3: PCA analysis
            return (True, 75, "🔬 Performing advanced analysis...", 
                    complete_icon, complete_text, complete_icon, complete_text, 
                    active_icon, active_text, default_icon, default_text, False,
                    False, "", {"display": "none"}, "", "", "", "", "", "")
        elif n_intervals < 20:  # Step 4: Generating charts
            return (True, 90, "📈 Creating enhanced visualizations...", 
                    complete_icon, complete_text, complete_icon, complete_text, 
                    complete_icon, complete_text, active_icon, active_text, False,
                    False, "", {"display": "none"}, "", "", "", "", "", "")
        else:  # Complete - perform actual analysis
            try:
                logger.info("Starting enhanced analysis...")
                
                # Parse analysis options
                standardize = "standardize" in (analysis_options or [])
                include_volatility = "volatility" in (analysis_options or [])
                include_risk = "risk" in (analysis_options or [])
                
                # Handle demo data case
                if tickers.upper().strip() == "DEMO":
                    logger.info("Using demo synthetic data...")
                    data = EnhancedDataHandler._generate_synthetic_data(
                        ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E'],
                        start_date, end_date
                    )
                else:
                    # Fetch and process real data
                    logger.info(f"Fetching data for: {tickers}")
                    data = EnhancedDataHandler.get_enhanced_stock_data(tickers, start_date, end_date)
                
                # Fetch economic indicators if selected
                economic_data = None
                if economic_indicators:
                    try:
                        economic_data = fetch_economic_data(economic_indicators, start_date, end_date)
                        logger.info(f"Fetched economic indicators: {economic_indicators}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch economic indicators: {e}")
                
                # Calculate returns
                returns = data.pct_change().dropna()
                
                # Enhanced volatility analysis if requested
                volatility_data = None
                if include_volatility:
                    vol_analyzer = EnhancedVolatilityAnalyzer()
                    volatility_data = {}
                    for ticker in data.columns:
                        ticker_data = pd.DataFrame({
                            'Close': data[ticker],
                            'High': data[ticker] * (1 + np.random.normal(0, 0.01, len(data))),  # Simulated for demo
                            'Low': data[ticker] * (1 - np.random.normal(0, 0.01, len(data))),
                            'Open': data[ticker].shift(1)
                        }).dropna()
                        
                        enhanced_vol = vol_analyzer.calculate_advanced_volatility(ticker_data)
                        volatility_data[ticker] = enhanced_vol
                
                # Standardize returns if requested
                if standardize:
                    scaler = StandardScaler()
                    returns_scaled = pd.DataFrame(
                        scaler.fit_transform(returns),
                        columns=returns.columns,
                        index=returns.index
                    )
                else:
                    returns_scaled = returns
                
                # Apply PCA
                pca = PCA(n_components=min(n_components, len(returns.columns)))
                pca.fit(returns_scaled)
                
                # Calculate enhanced metrics
                metrics = calculate_enhanced_metrics(returns_scaled, pca, pca.n_components_)
                
                # Calculate summary statistics
                total_variance = sum(pca.explained_variance_ratio_) * 100
                n_stocks = len(returns.columns)
                period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                
                # Calculate average volatility and Sharpe ratio
                avg_volatility = np.mean([metrics['stock_metrics'][stock].get('Volatility', 0) 
                                        for stock in returns.columns if stock in metrics['stock_metrics']]) * 100
                avg_sharpe = np.mean([metrics['stock_metrics'][stock].get('Sharpe_Ratio', 0) 
                                    for stock in returns.columns if stock in metrics['stock_metrics']])
                
                # Run advanced VAR/GARCH analysis if available and requested
                vargarch_results = None
                if ADVANCED_GARCH_AVAILABLE and ('volatility' in (analysis_options or []) or 'risk' in (analysis_options or [])):
                    try:
                        logger.info("Running advanced VAR/GARCH analysis...")
                        vargarch_analyzer = IntegratedVARGARCHAnalysis(max_lags=5)
                        vargarch_results = vargarch_analyzer.analyze_portfolio(
                            returns,
                            economic_data=economic_data,
                            analyze_var=True,
                            analyze_garch=True,
                            max_garch_stocks=min(5, len(returns.columns))
                        )
                        logger.info("Advanced VAR/GARCH analysis completed!")
                    except Exception as e:
                        logger.warning(f"Advanced VAR/GARCH analysis failed: {e}")
                        vargarch_results = None

                # Store results globally for other callbacks
                global pca_results, lstm_results_cache, timeseries_results_cache, timeseries_cache_key
                # Reset caches for new analysis
                lstm_results_cache = None
                timeseries_results_cache = None
                timeseries_cache_key = None
                
                pca_results = {
                    'pca': pca,
                    'returns': returns,
                    'returns_scaled': returns_scaled,
                    'data': data,
                    'n_components': pca.n_components_,
                    'metrics': metrics,
                    'volatility_data': volatility_data,
                    'economic_data': economic_data,
                    'analysis_options': analysis_options or [],
                    'economic_indicators': economic_indicators or [],
                    'vargarch_results': vargarch_results,
                    'advanced_garch_available': ADVANCED_GARCH_AVAILABLE
                }
                
                logger.info("Enhanced analysis completed successfully!")
                return (False, 100, "✅ Analysis complete!", 
                        complete_icon, complete_text, complete_icon, complete_text, 
                        complete_icon, complete_text, complete_icon, complete_text, True,
                        False, "", {"display": "block"}, 
                        f"{total_variance:.1f}%", str(pca.n_components_), 
                        str(n_stocks), f"{period_days} days",
                        f"{avg_volatility:.1f}%", f"{avg_sharpe:.2f}")
                
            except Exception as e:
                logger.error(f"Error in enhanced analysis: {str(e)}")
                logger.error(traceback.format_exc())
                error_msg = [
                    html.H5("🚨 Analysis Error", className="alert-heading"),
                    html.P(str(e)),
                    html.Hr(),
                    html.P([
                        "💡 Try using 'Demo Data' from the preset dropdown for a working example. ",
                        "Make sure your tickers are valid and you have sufficient date range."
                    ], className="mb-0 text-muted")
                ]
                return (False, 100, "❌ Error occurred", 
                        complete_icon, complete_text, complete_icon, complete_text, 
                        complete_icon, complete_text, complete_icon, complete_text, True,
                        True, error_msg, {"display": "none"}, "", "", "", "", "", "")
    
    return default_return

def fetch_economic_data(indicator_categories, start_date, end_date):
    """Fetch enhanced economic indicator data with comprehensive metadata"""
    economic_data = {}
    economic_metadata = {}
    
    for category in indicator_categories:
        if category in ECONOMIC_INDICATORS:
            indicator_info = ECONOMIC_INDICATORS[category]
            tickers = indicator_info["tickers"]
            
            # Store metadata for this category
            economic_metadata[category] = {
                "description": indicator_info["description"],
                "impact": indicator_info["impact"],
                "tickers": tickers
            }
            
            try:
                # Download with explicit parameters to avoid warnings
                category_data = yf.download(
                    tickers, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    auto_adjust=True,  # Explicitly set to avoid warning
                    prepost=False,
                    threads=False,  # Disable threading for economic data
                    group_by='ticker' if len(tickers) > 1 else None
                )
                
                if not category_data.empty:
                    # Process the data to get Close prices
                    if isinstance(category_data.columns, pd.MultiIndex):
                        for ticker in tickers:
                            try:
                                if ticker in category_data.columns.get_level_values(0):
                                    if 'Close' in category_data[ticker].columns:
                                        economic_data[f"{category}_{ticker}"] = category_data[ticker]['Close']
                                    elif 'Adj Close' in category_data[ticker].columns:
                                        economic_data[f"{category}_{ticker}"] = category_data[ticker]['Adj Close']
                            except Exception:
                                continue
                    else:
                        # Single ticker case
                        if 'Close' in category_data.columns:
                            economic_data[f"{category}_{tickers[0]}"] = category_data['Close']
                        elif 'Adj Close' in category_data.columns:
                            economic_data[f"{category}_{tickers[0]}"] = category_data['Adj Close']
                        
            except Exception as e:
                logger.warning(f"Failed to fetch {category}: {str(e)}")
                continue
    
    # Return both data and metadata for enhanced analysis
    result = pd.DataFrame(economic_data).dropna() if economic_data else None
    if result is not None:
        result.metadata = economic_metadata  # Attach metadata to the DataFrame
    return result

@app.callback(
    Output("tab-content", "children"),
    [Input("analysis-tabs", "active_tab"),
     Input("results-section", "style")],
    prevent_initial_call=True
)
def update_enhanced_tab_content(active_tab, results_style):
    """Update tab content with enhanced features"""
    # Check if results are visible and we have data
    if results_style.get("display") == "none" or pca_results is None:
        return create_no_results_message()
    
    results = pca_results
    
    try:
        if active_tab == "variance-tab":
            return create_variance_tab(results['pca'], results['metrics'])
        elif active_tab == "loadings-tab":
            return create_loadings_tab(results['metrics'])
        elif active_tab == "timeseries-tab":
            # Add specific diagnostics for time series tab
            logger.info(f"Loading time series tab. ENHANCED_TIMESERIES_AVAILABLE: {ENHANCED_TIMESERIES_AVAILABLE}")
            
            # Check if we have required data
            metrics = results.get('metrics')
            returns = results.get('returns')
            if not metrics or returns is None or (hasattr(returns, '__len__') and len(returns) == 0):
                return create_time_series_error_message("Missing required data for time series analysis")
            
            factor_scores = results.get('metrics', {}).get('factor_scores')
            if factor_scores is None or len(factor_scores) == 0:
                return create_time_series_error_message("No factor scores available for time series analysis")
            
            logger.info(f"Time series data check - Factor scores shape: {factor_scores.shape}, Returns shape: {results.get('returns', pd.DataFrame()).shape}")
            
            try:
                if ENHANCED_TIMESERIES_AVAILABLE:
                    logger.info("🔄 Starting enhanced time series analysis - this may take 30-120 seconds...")
                    # Show initial loading message
                    loading_start_time = datetime.datetime.now()
                    logger.info(f"⏱️  Analysis started at: {loading_start_time.strftime('%H:%M:%S')}")
                    
                    # Return loading screen followed by actual analysis
                    return create_enhanced_timeseries_with_loading(results)
                else:
                    logger.info("Creating fallback time series tab...")
                    return create_timeseries_tab(results)
            except Exception as ts_error:
                logger.error(f"Specific time series error: {str(ts_error)}")
                import traceback
                logger.error(f"Time series traceback: {traceback.format_exc()}")
                return create_time_series_error_message(f"Time series creation failed: {str(ts_error)}")
        elif active_tab == "volatility-tab":
            return create_volatility_tab(results)
        elif active_tab == "risk-tab":
            return create_risk_tab(results)
        elif active_tab == "vargarch-tab":
            return create_vargarch_tab(results)
        elif active_tab == "lstm-tab":
            return create_lstm_tab(results)
        elif active_tab == "detailed-tab":
            return create_detailed_tab(results['metrics'], results['returns'])
        else:
            return html.Div("Select a tab to view results.")
    except Exception as e:
        logger.error(f"Error creating tab content: {str(e)}")
        return create_error_message(str(e))

def create_no_results_message():
    """Create message when no results are available"""
    return html.Div([
        dbc.Alert([
            html.H5("🔍 No Analysis Results Available", className="alert-heading"),
            html.P("Please run the analysis first by clicking the '🚀 Run Enhanced Analysis' button."),
            html.Hr(),
            html.P([
                "💡 Try using 'Demo Data' from the preset dropdown for a quick test! ",
                "You can also explore different analysis options like volatility analysis and risk metrics."
            ], className="mb-0")
        ], color="info", className="mt-4")
    ])

def create_error_message(error_str):
    """Create error message"""
    return dbc.Alert([
        html.H5("🚨 Error Loading Charts"),
        html.P(f"Error: {error_str}"),
        html.P("Please try running the analysis again with different parameters.")
    ], color="danger")

def create_time_series_error_message(error_str):
    """Create specific error message for time series tab"""
    return dbc.Alert([
        html.H5("📈 Time Series Analysis Issue", className="alert-heading"),
        html.P(f"Issue: {error_str}"),
        html.Hr(),
        html.H6("Troubleshooting Steps:", className="mb-2"),
        html.Ol([
            html.Li("Ensure you have run the PCA analysis first by clicking '🚀 Run Enhanced Analysis'"),
            html.Li("Try using 'Demo Data' from the preset dropdown if you haven't loaded any data"),
            html.Li("Make sure your data has at least 30 observations for time series analysis"),
            html.Li("Check that your stock data contains valid time series information")
        ]),
        html.P([
            "💡 ",
            html.Strong("Tip: "),
            "The time series analysis requires PCA factor scores and stock returns data to be available."
        ], className="text-info mt-2")
    ], color="warning", className="mt-4")


def create_timeseries_immediate_loading():
    """Create immediate loading screen for time series analysis"""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        # Large spinner
                        dbc.Spinner(
                            size="lg",
                            color="primary",
                            spinner_style={"width": "4rem", "height": "4rem"},
                            spinner_class_name="mb-4"
                        ),
                        
                        # Title
                        html.H3([
                            html.I(className="fas fa-chart-line me-3"),
                            "Enhanced Time Series Analysis"
                        ], className="text-primary mb-3"),
                        
                        # Progress message
                        html.H5("🔄 Processing Advanced Analytics...", className="text-muted mb-4"),
                        
                        # Info alert
                        dbc.Alert([
                            html.H6("⏱️ Processing Time", className="alert-heading"),
                            html.P([
                                "Running comprehensive analysis including regime detection, ",
                                "rolling correlations, and advanced visualizations. "
                            ]),
                            html.P([
                                html.Strong("Estimated time: "),
                                "30-120 seconds depending on data size"
                            ], className="mb-0")
                        ], color="info"),
                        
                        # Loading steps
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Spinner(size="sm", color="secondary", spinner_class_name="me-2"),
                                "Normalizing factor scores and detecting regimes..."
                            ]),
                            dbc.ListGroupItem([
                                dbc.Spinner(size="sm", color="secondary", spinner_class_name="me-2"),
                                "Computing rolling statistics and momentum indicators..."
                            ]),
                            dbc.ListGroupItem([
                                dbc.Spinner(size="sm", color="secondary", spinner_class_name="me-2"),
                                "Analyzing stock-factor correlations..."
                            ]),
                            dbc.ListGroupItem([
                                dbc.Spinner(size="sm", color="secondary", spinner_class_name="me-2"),
                                "Generating 8-panel interactive visualization..."
                            ])
                        ], flush=True)
                        
                    ], className="text-center")
                ], width=6)
            ], justify="center", className="min-vh-75 align-items-center")
        ], fluid=True)
    ], style={"minHeight": "80vh"})


def create_enhanced_timeseries_with_loading(results):
    """Enhanced time series analysis with user-friendly loading experience and caching"""
    global timeseries_results_cache, timeseries_cache_key
    
    # Create a cache key based on PCA results to detect changes
    try:
        factor_scores = results.get('metrics', {}).get('factor_scores', pd.DataFrame())
        if hasattr(factor_scores, 'values') and len(factor_scores) > 0:
            # Use shape and a sample of values for cache key
            cache_data = f"{factor_scores.shape}_{str(factor_scores.iloc[0:5, 0:3].values.flatten())}"
        else:
            cache_data = "empty_factor_scores"
        current_cache_key = str(hash(cache_data))
    except Exception as e:
        logger.warning(f"Could not create cache key: {e}")
        current_cache_key = str(hash(str(datetime.datetime.now())))
    
    # Check if we have cached results for the current PCA analysis
    if (timeseries_results_cache is not None and 
        timeseries_cache_key == current_cache_key):
        logger.info("🚀 Loading cached time series analysis results")
        
        # Create completion banner for cached results
        completion_banner = dbc.Alert([
            html.P([
                html.I(className="fas fa-check-circle text-success me-2"),
                html.Strong("Cached Analysis Loaded! "),
                "Previously computed time series analysis results"
            ], className="mb-0")
        ], color="success", dismissable=True, className="mb-3")
        
        return html.Div([
            completion_banner,
            timeseries_results_cache
        ])
    
    start_time = datetime.datetime.now()
    logger.info(f"🚀 Enhanced Time Series Analysis started at {start_time.strftime('%H:%M:%S')}")
    
    # Create header with loading info that shows immediately
    loading_header = html.Div([
        dbc.Alert([
            html.Div([
                dbc.Spinner(color="primary", size="sm", spinner_class_name="me-2"),
                html.H5([
                    html.I(className="fas fa-chart-line me-2"),
                    "Enhanced Time Series Analysis - Processing..."
                ], className="mb-2"),
            ], className="d-flex align-items-center"),
            
            html.P([
                "⏳ Running comprehensive analysis with regime detection, rolling correlations, and advanced visualizations. ",
                f"Started at {start_time.strftime('%H:%M:%S')} - Estimated completion: 30-120 seconds."
            ], className="mb-2"),
            
            dbc.Progress(value=100, animated=True, striped=True, style={"height": "10px"}),
            
            html.P([
                "💡 This analysis includes rolling PCA, factor loadings drift analysis, and multi-dimensional correlation calculations. ",
                "The results will appear below once processing is complete."
            ], className="small text-muted mt-2 mb-0")
        ], color="info")
    ])
    
    try:
        # Run the actual enhanced analysis
        analysis_results = create_enhanced_timeseries_tab(results)
        
        # Cache the results
        timeseries_results_cache = analysis_results
        timeseries_cache_key = current_cache_key
        logger.info("✅ Time series analysis results cached successfully")
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"✅ Enhanced Time Series Analysis completed in {duration:.1f} seconds")
        
        # Add completion info to results
        completion_banner = dbc.Alert([
            html.P([
                html.I(className="fas fa-check-circle text-success me-2"),
                html.Strong("Analysis Complete! "),
                f"Finished at {end_time.strftime('%H:%M:%S')} ({duration:.1f}s)"
            ], className="mb-0")
        ], color="success", dismissable=True, className="mb-3")
        
        # Combine loading header, completion banner, and results
        return html.Div([
            completion_banner,
            analysis_results
        ])
        
    except Exception as e:
        logger.error(f"❌ Enhanced Time Series Analysis failed: {str(e)}")
        return html.Div([
            loading_header,
            create_time_series_error_message(f"Analysis failed after {(datetime.datetime.now() - start_time).total_seconds():.1f}s: {str(e)}")
        ])


def create_volatility_tab(results):
    """Create enhanced volatility analysis tab"""
    # Check if advanced GARCH results are available
    vargarch_results = results.get('vargarch_results')
    
    if vargarch_results and vargarch_results.get('garch_results') and ADVANCED_GARCH_AVAILABLE:
        # Use advanced GARCH analysis if available
        try:
            visualizer = VARGARCHVisualizer()
            
            # Create advanced volatility visualization
            vol_fig = visualizer.create_volatility_comparison(
                vargarch_results['garch_results'],
                results['returns']
            )
            
            return html.Div([
                dbc.Alert([
                    html.H5("🚀 Advanced GARCH Volatility Analysis", className="alert-heading"),
                    html.P(["This analysis uses professional GARCH(1,1) models to estimate conditional volatility. ",
                           "GARCH parameters show volatility persistence and shock impact."]),
                ], color="info", className="mb-4"),
                
                dcc.Graph(figure=vol_fig, className="mb-4"),
                
                # Add GARCH parameters table
                create_garch_parameters_table(vargarch_results['garch_results'])
            ])
            
        except Exception as e:
            logger.error(f"Error creating advanced volatility analysis: {e}")
            # Fall back to basic analysis
    
    # Basic volatility analysis (original code)
    if not results.get('volatility_data'):
        return dbc.Alert([
            html.H5("📊 Volatility Analysis Not Available"),
            html.P("Enable 'Include Volatility Analysis' in the analysis options and run the analysis again.")
        ], color="warning")
    
    volatility_data = results['volatility_data']
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Multiple Volatility Measures Comparison",
            "Volatility Distribution Analysis", 
            "Rolling Volatility Correlation",
            "Volatility Clustering Analysis"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Multiple volatility measures for first stock
    if volatility_data:
        first_stock = list(volatility_data.keys())[0]
        vol_df = volatility_data[first_stock].tail(252)  # Last year
        
        fig.add_trace(
            go.Scatter(x=vol_df.index, y=vol_df['Volatility'], 
                      name="Rolling Volatility", line=dict(color="#1f77b4")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=vol_df.index, y=vol_df['EW_Volatility'], 
                      name="EW Volatility", line=dict(color="#ff7f0e")),
            row=1, col=1
        )
        
        if 'GK_Volatility' in vol_df.columns:
            fig.add_trace(
                go.Scatter(x=vol_df.index, y=vol_df['GK_Volatility'], 
                          name="Garman-Klass", line=dict(color="#2ca02c")),
                row=1, col=1
            )
    
    # Plot 2: Volatility distribution
    if volatility_data:
        vol_values = []
        stock_names = []
        for stock, data in volatility_data.items():
            vol_values.extend(data['Volatility'].dropna().values)
            stock_names.extend([stock] * len(data['Volatility'].dropna()))
        
        fig.add_trace(
            go.Histogram(x=vol_values, nbinsx=30, name="Vol Distribution",
                        marker_color="#9467bd", opacity=0.7),
            row=1, col=2
        )
    
    # Plot 3: Rolling correlation between stocks' volatilities
    if len(volatility_data) >= 2:
        stocks = list(volatility_data.keys())[:2]
        vol1 = volatility_data[stocks[0]]['Volatility']
        vol2 = volatility_data[stocks[1]]['Volatility']
        
        rolling_corr = vol1.rolling(60).corr(vol2)
        
        fig.add_trace(
            go.Scatter(x=rolling_corr.index, y=rolling_corr.values,
                      name=f"Corr: {stocks[0]} vs {stocks[1]}", 
                      line=dict(color="#d62728")),
            row=2, col=1
        )
    
    # Plot 4: Volatility clustering (autocorrelation)
    if volatility_data:
        first_stock_vol = volatility_data[first_stock]['Volatility'].dropna()
        if len(first_stock_vol) > 30:
            autocorr = [first_stock_vol.autocorr(lag=i) for i in range(1, 21)]
            
            fig.add_trace(
                go.Bar(x=list(range(1, 21)), y=autocorr, 
                      name="Volatility Autocorr", marker_color="#8c564b"),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800, 
        title_text="📊 Advanced Volatility Analysis",
        showlegend=True
    )
    
    return dcc.Graph(figure=fig)

def create_risk_tab(results):
    """Create comprehensive risk analysis tab"""
    stock_metrics = results['metrics']['stock_metrics']
    
    if not stock_metrics:
        return dbc.Alert("No risk metrics available", color="warning")
    
    # Create risk metrics comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Risk-Return Scatter (Sharpe vs Volatility)",
            "Value at Risk (VaR) Analysis",
            "Maximum Drawdown Comparison", 
            "Risk Metrics Heatmap"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Extract metrics for plotting
    stocks = list(stock_metrics.keys())
    volatilities = [stock_metrics[s].get('Volatility', 0) * 100 for s in stocks]
    sharpe_ratios = [stock_metrics[s].get('Sharpe_Ratio', 0) for s in stocks]
    var_95 = [stock_metrics[s].get('VaR_95', 0) * 100 for s in stocks]
    max_drawdowns = [stock_metrics[s].get('Max_Drawdown', 0) * 100 for s in stocks]
    
    # Plot 1: Risk-Return scatter
    fig.add_trace(
        go.Scatter(x=volatilities, y=sharpe_ratios, mode='markers+text',
                  text=stocks, textposition="top center",
                  marker=dict(size=10, color=sharpe_ratios, colorscale='RdYlGn',
                            showscale=True, colorbar=dict(title="Sharpe Ratio")),
                  name="Risk-Return"),
        row=1, col=1
    )
    
    # Plot 2: VaR comparison
    fig.add_trace(
        go.Bar(x=stocks, y=var_95, name="VaR 95%", 
              marker_color="#d62728", opacity=0.7),
        row=1, col=2
    )
    
    # Plot 3: Maximum Drawdown
    fig.add_trace(
        go.Bar(x=stocks, y=max_drawdowns, name="Max Drawdown (%)", 
              marker_color="#ff7f0e", opacity=0.7),
        row=2, col=1
    )
    
    # Plot 4: Risk metrics heatmap
    risk_metrics_matrix = []
    metric_names = ['Volatility', 'Sharpe_Ratio', 'VaR_95', 'Max_Drawdown', 'Sortino_Ratio']
    
    for metric in metric_names:
        row = []
        for stock in stocks:
            value = stock_metrics[stock].get(metric, 0)
            if metric in ['Volatility', 'VaR_95', 'Max_Drawdown']:
                value *= 100  # Convert to percentage
            row.append(value)
        risk_metrics_matrix.append(row)
    
    fig.add_trace(
        go.Heatmap(z=risk_metrics_matrix, x=stocks, y=metric_names,
                  colorscale='RdBu_r', text=risk_metrics_matrix, 
                  texttemplate="%{text:.2f}", textfont={"size":10}),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="⚠️ Comprehensive Risk Analysis",
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="VaR 95% (%)", row=1, col=2)
    fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Hr(),
        create_risk_summary_table(stock_metrics)
    ])

def create_risk_summary_table(stock_metrics):
    """Create risk summary table"""
    summary_data = []
    for stock, metrics in stock_metrics.items():
        summary_data.append({
            'Stock': stock,
            'Volatility (%)': f"{metrics.get('Volatility', 0) * 100:.2f}",
            'Sharpe Ratio': f"{metrics.get('Sharpe_Ratio', 0):.3f}",
            'Sortino Ratio': f"{metrics.get('Sortino_Ratio', 0):.3f}",
            'VaR 95% (%)': f"{metrics.get('VaR_95', 0) * 100:.2f}",
            'Max Drawdown (%)': f"{metrics.get('Max_Drawdown', 0) * 100:.2f}",
            'Skewness': f"{metrics.get('Skewness', 0):.3f}",
            'Kurtosis': f"{metrics.get('Kurtosis', 0):.3f}"
        })
    
    return html.Div([
        html.H5("📋 Risk Metrics Summary"),
        dash_table.DataTable(
            data=summary_data,
            columns=[{'name': col, 'id': col} for col in summary_data[0].keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontSize': 12},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'Sharpe Ratio', 'filter_query': '{Sharpe Ratio} > 1'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Max Drawdown (%)', 'filter_query': '{Max Drawdown (%)} < -20'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ],
            sort_action='native',
            export_format='csv'
        )
    ])

# Additional utility functions for existing tabs (keeping the ones from original code)
def create_variance_tab(pca, metrics):
    """Enhanced variance analysis tab - FIXED pie chart issue"""
    explained_var = metrics['explained_variance']
    
    # Create subplots - fix pie chart issue by using domain type for pie
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Explained Variance by Component",
            "Cumulative Explained Variance",
            "Scree Plot (Eigenvalues)",
            "Variance Contribution Breakdown"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "domain"}]]  # Domain type for pie chart
    )
    
    # Explained Variance Bar Chart
    fig.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))],
               y=explained_var * 100, name="Individual Variance",
               marker_color="#1f77b4", opacity=0.8),
        row=1, col=1
    )
    
    # Cumulative Variance Line Chart
    cumulative_var = np.cumsum(explained_var) * 100
    fig.add_trace(
        go.Scatter(x=[f'PC{i+1}' for i in range(len(explained_var))],
                  y=cumulative_var, mode='lines+markers',
                  name="Cumulative Variance", line=dict(color="#ff7f0e", width=3)),
        row=1, col=2
    )
    
    # Add 80% threshold line
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80% Threshold", row=1, col=2)
    
    # Scree Plot
    eigenvals = pca.explained_variance_
    fig.add_trace(
        go.Scatter(x=[f'PC{i+1}' for i in range(len(eigenvals))],
                  y=eigenvals, mode='lines+markers',
                  name="Eigenvalues", line=dict(color="#2ca02c", width=3),
                  marker=dict(size=8)),
        row=2, col=1
    )
    
    # Kaiser criterion line
    fig.add_hline(y=1, line_dash="dash", line_color="red",
                  annotation_text="Kaiser Criterion (λ=1)", row=2, col=1)
    
    # Variance contribution pie chart - FIXED
    fig.add_trace(
        go.Pie(labels=[f'PC{i+1}' for i in range(len(explained_var))],
               values=explained_var, 
               name="Variance Breakdown",
               hole=0.3,  # Donut style
               marker=dict(colors=px.colors.qualitative.Set3)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800, 
        title_text="📊 Enhanced Variance Analysis", 
        showlegend=True,
        title_x=0.5
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_yaxes(title_text="Explained Variance (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Principal Component", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Variance (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Principal Component", row=2, col=1)
    fig.update_yaxes(title_text="Eigenvalue", row=2, col=1)
    
    return dcc.Graph(figure=fig)

def create_loadings_tab(metrics):
    """Enhanced factor loadings analysis"""
    loadings = metrics['loadings']
    communalities = metrics['communalities']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Factor Loadings Heatmap",
            "Biplot: First Two Components", 
            "Communalities Analysis",
            "Loading Magnitudes by Component"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Factor Loadings Heatmap
    fig.add_trace(
        go.Heatmap(z=loadings.T.values, x=loadings.index, y=loadings.columns,
                  colorscale='RdBu', zmid=0, name="Loadings",
                  text=loadings.T.values, texttemplate="%{text:.2f}"),
        row=1, col=1
    )
    
    # Biplot (first two components)
    if loadings.shape[1] >= 2:
        fig.add_trace(
            go.Scatter(x=loadings.iloc[:, 0], y=loadings.iloc[:, 1],
                      mode='markers+text', text=loadings.index,
                      textposition="top center", name="Stocks",
                      marker=dict(size=8, color="#9467bd")),
            row=1, col=2
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Communalities Chart
    fig.add_trace(
        go.Bar(x=communalities.index, y=communalities.values,
               name="Communalities", marker_color="#ff7f0e", opacity=0.8),
        row=2, col=1
    )
    
    # Loading magnitudes by component
    for i, col in enumerate(loadings.columns):
        fig.add_trace(
            go.Bar(x=loadings.index, y=np.abs(loadings[col]),
                   name=f"PC{i+1} Magnitude", opacity=0.7),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="🎯 Enhanced Factor Loadings Analysis")
    
    return dcc.Graph(figure=fig)

# DEPRECATED: This function is kept as fallback when enhanced time series module is not available
def create_timeseries_tab(results):
    """Legacy time series analysis - kept as fallback for enhanced time series visualization"""
    factor_scores = results['metrics']['factor_scores']
    returns = results['returns']
    
    # Create main time series visualization
    main_fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "📊 Principal Component Scores (Normalized)",
            "🔄 PC Scores Cumulative Performance",
            "📈 Rolling Correlation with First PC", 
            "📉 Individual Stock Price Movements",
            "⚡ Factor Score Volatility (60-day rolling)",
            "🌍 Economic Indicators (if available)"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08
    )
    
    # 1. Enhanced Normalized Principal Component Scores
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Calculate additional statistics for each PC
    for i, col in enumerate(factor_scores.columns):
        # Normalize scores to make them more comparable
        normalized_scores = (factor_scores[col] - factor_scores[col].mean()) / factor_scores[col].std()
        
        # Calculate rolling statistics
        rolling_mean = normalized_scores.rolling(window=30, min_periods=15).mean()
        rolling_std = normalized_scores.rolling(window=30, min_periods=15).std()
        upper_band = rolling_mean + 1.5 * rolling_std
        lower_band = rolling_mean - 1.5 * rolling_std
        
        # Add volatility bands (fill area)
        main_fig.add_trace(
            go.Scatter(
                x=factor_scores.index,
                y=upper_band,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name=f"{col} Upper Band"
            ),
            row=1, col=1
        )
        
        main_fig.add_trace(
            go.Scatter(
                x=factor_scores.index,
                y=lower_band,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.1)',
                showlegend=False,
                name=f"{col} Volatility Band"
            ),
            row=1, col=1
        )
        
        # Add rolling mean line
        main_fig.add_trace(
            go.Scatter(
                x=factor_scores.index,
                y=rolling_mean,
                name=f"{col} 30d MA",
                line=dict(width=1.5, color=colors[i % len(colors)], dash='dot'),
                opacity=0.6,
                hovertemplate=f"{col} Moving Average<br>Date: %{{x}}<br>30d MA: %{{y:.3f}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Main normalized scores line (enhanced)
        main_fig.add_trace(
            go.Scatter(
                x=factor_scores.index, 
                y=normalized_scores,
                name=f"{col} (Z-score)",
                line=dict(width=2.8, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{col}</b><br>" +
                            f"Date: %{{x}}<br>" +
                            f"Z-Score: %{{y:.3f}}<br>" +
                            f"Original: {factor_scores[col].loc[factor_scores.index[0]]:.4f}<br>" +
                            f"<i>Explained Variance: {results['metrics']['explained_variance'][i]*100:.1f}%</i><extra></extra>",
                mode='lines'
            ),
            row=1, col=1
        )
    
    # Add enhanced reference lines
    main_fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.7, row=1, col=1)
    main_fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=1, opacity=0.4, row=1, col=1)
    main_fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=1, opacity=0.4, row=1, col=1)
    main_fig.add_hline(y=1, line_dash="dot", line_color="orange", line_width=0.8, opacity=0.3, row=1, col=1)
    main_fig.add_hline(y=-1, line_dash="dot", line_color="orange", line_width=0.8, opacity=0.3, row=1, col=1)
    
    # Add annotations for extreme values
    for i, col in enumerate(factor_scores.columns):
        normalized_scores = (factor_scores[col] - factor_scores[col].mean()) / factor_scores[col].std()
        extreme_high = normalized_scores[normalized_scores > 2.5]
        extreme_low = normalized_scores[normalized_scores < -2.5]
        
        # Annotate extreme highs
        for date, value in extreme_high.head(3).items():  # Limit to top 3
            main_fig.add_annotation(
                x=date, y=value,
                text=f"📈{value:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors[i % len(colors)],
                row=1, col=1,
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=colors[i % len(colors)]
            )
        
        # Annotate extreme lows  
        for date, value in extreme_low.head(3).items():  # Limit to top 3
            main_fig.add_annotation(
                x=date, y=value,
                text=f"📉{value:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors[i % len(colors)],
                row=1, col=1,
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=colors[i % len(colors)]
            )
    
    # 2. Cumulative performance of PC scores
    for i, col in enumerate(factor_scores.columns):
        cumulative_scores = (1 + factor_scores[col]).cumprod()
        main_fig.add_trace(
            go.Scatter(
                x=factor_scores.index,
                y=cumulative_scores,
                name=f"{col} Cumulative",
                line=dict(width=2, color=colors[i % len(colors)], dash='dot'),
                hovertemplate=f"{col}<br>Date: %{{x}}<br>Cumulative: %{{y:.3f}}<extra></extra>"
            ),
            row=1, col=2
        )
    
    # 3. Rolling Correlations with first PC (improved)
    first_pc = factor_scores.iloc[:, 0]
    stock_colors = px.colors.qualitative.Pastel
    for i, stock in enumerate(returns.columns[:6]):  # Limit to 6 stocks for clarity
        rolling_corr = returns[stock].rolling(window=60, min_periods=30).corr(first_pc)
        main_fig.add_trace(
            go.Scatter(
                x=rolling_corr.index, 
                y=rolling_corr.values,
                name=f"{stock}",
                line=dict(width=1.5, color=stock_colors[i % len(stock_colors)]),
                opacity=0.8,
                hovertemplate=f"{stock} vs PC1<br>Date: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>"
            ),
            row=2, col=1
        )
    
    # Add correlation reference lines
    main_fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)
    main_fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    main_fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 4. Individual stock normalized price movements  
    data = results.get('data', returns)
    if data is not None and len(data) > 0:
        for i, stock in enumerate(returns.columns[:5]):  # Show top 5 stocks
            if stock in data.columns:
                # Normalize to start at 1
                normalized_prices = data[stock] / data[stock].iloc[0]
                main_fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized_prices,
                        name=f"{stock} Price",
                        line=dict(width=1.5),
                        hovertemplate=f"{stock}<br>Date: %{{x}}<br>Normalized Price: %{{y:.3f}}<extra></extra>"
                    ),
                    row=2, col=2
                )
    
    # 5. Enhanced Factor Score Volatility
    for i, col in enumerate(factor_scores.columns):
        rolling_vol = factor_scores[col].rolling(60).std()
        rolling_mean = factor_scores[col].rolling(60).mean()
        
        # Add volatility
        main_fig.add_trace(
            go.Scatter(
                x=rolling_vol.index, 
                y=rolling_vol.values,
                name=f"{col} Volatility",
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"{col}<br>Date: %{{x}}<br>60d Volatility: %{{y:.4f}}<extra></extra>"
            ),
            row=3, col=1
        )
    
    # 6. Economic Indicators (enhanced)
    if results.get('economic_data') is not None and len(results['economic_data']) > 0:
        econ_data = results['economic_data']
        econ_colors = px.colors.qualitative.Dark2
        
        for i, col in enumerate(econ_data.columns[:4]):  # Show first 4 indicators
            # Better normalization
            col_data = econ_data[col].dropna()
            if len(col_data) > 0:
                normalized_data = (col_data - col_data.mean()) / col_data.std()
                main_fig.add_trace(
                    go.Scatter(
                        x=col_data.index,
                        y=normalized_data.values,
                        name=col.replace('_', ' ').title(),
                        line=dict(width=2, color=econ_colors[i % len(econ_colors)]),
                        hovertemplate=f"{col}<br>Date: %{{x}}<br>Normalized Value: %{{y:.3f}}<extra></extra>"
                    ),
                    row=3, col=2
                )
    else:
        # Enhanced placeholder message
        main_fig.add_annotation(
            x=0.5, y=0.5, 
            xref="x6 domain", yref="y6 domain",
            text="📊 No Economic Indicators Selected<br><br>💡 Enable economic indicators in<br>Analysis Options for enhanced insights",
            showarrow=False, 
            font=dict(size=12, color="gray"),
            bgcolor="rgba(240,240,240,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    # Update layout with better formatting and interactivity
    main_fig.update_layout(
        height=1000,
        title_text="📈 Enhanced Time Series Analysis - Principal Components Evolution",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )
    
    # Add range selector buttons specifically for the PC scores plot
    main_fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(count=90, label="3M", step="day", stepmode="backward"),
                dict(count=180, label="6M", step="day", stepmode="backward"),
                dict(count=365, label="1Y", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="rgba(150, 150, 150, 0.1)",
            bordercolor="rgba(150, 150, 150, 0.2)",
            borderwidth=1,
            font=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Update axis labels with enhanced formatting
    main_fig.update_xaxes(title_text="Date", row=3, col=1)
    main_fig.update_xaxes(title_text="Date", row=3, col=2)
    
    # Enhanced Y-axis for PC scores
    main_fig.update_yaxes(
        title_text="Z-Score (Standard Deviations)",
        row=1, col=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(128, 128, 128, 0.8)",
        zerolinewidth=2,
        tickformat=".2f",
        title_font=dict(size=12, color='#34495e'),
        showspikes=True,
        spikethickness=1,
        spikecolor='rgba(128, 128, 128, 0.5)'
    )
    
    main_fig.update_yaxes(title_text="Cumulative Performance", row=1, col=2)
    main_fig.update_yaxes(title_text="Rolling Correlation", row=2, col=1)
    main_fig.update_yaxes(title_text="Normalized Price", row=2, col=2)
    main_fig.update_yaxes(title_text="Rolling Volatility", row=3, col=1)
    main_fig.update_yaxes(title_text="Economic Indicator", row=3, col=2)
    
    # Enhanced summary statistics with normalized scores analysis
    pc1_stats = factor_scores.iloc[:, 0]
    pc1_normalized = (pc1_stats - pc1_stats.mean()) / pc1_stats.std()
    
    # Calculate regime statistics  
    extreme_high_count = len(pc1_normalized[pc1_normalized > 2])
    extreme_low_count = len(pc1_normalized[pc1_normalized < -2])
    recent_trend = "Bullish" if pc1_stats.tail(30).mean() > pc1_stats.tail(60).mean() else "Bearish"
    volatility_regime = "High" if pc1_stats.rolling(30).std().iloc[-1] > pc1_stats.rolling(90).std().iloc[-1] else "Low"
    
    summary_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Enhanced PC1 Statistics", className="card-title", style={'color': '#2c3e50'}),
                    html.P(f"Z-Score Mean: {pc1_normalized.mean():.4f}"),
                    html.P(f"Z-Score Std: {pc1_normalized.std():.4f}"),
                    html.P(f"Current Z-Score: {pc1_normalized.iloc[-1]:.3f}", 
                          style={'color': 'red' if abs(pc1_normalized.iloc[-1]) > 2 else 'green'}),
                    html.P(f"Annualized Sharpe: {(pc1_stats.mean() * 252) / (pc1_stats.std() * np.sqrt(252)):.3f}"),
                ])
            ], style={'border': '1px solid #bdc3c7'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 Regime Analysis", className="card-title", style={'color': '#2c3e50'}),
                    html.P(f"Extreme Highs: {extreme_high_count} events"),
                    html.P(f"Extreme Lows: {extreme_low_count} events"),
                    html.P(f"Recent Trend: {recent_trend}",
                          style={'color': 'green' if recent_trend == 'Bullish' else 'red'}),
                    html.P(f"Volatility: {volatility_regime}",
                          style={'color': 'red' if volatility_regime == 'High' else 'green'}),
                ])
            ], style={'border': '1px solid #bdc3c7'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Range & Distribution", className="card-title", style={'color': '#2c3e50'}),
                    html.P(f"Z-Score Max: {pc1_normalized.max():.3f}"),
                    html.P(f"Z-Score Min: {pc1_normalized.min():.3f}"),
                    html.P(f"Z-Score Range: {pc1_normalized.max() - pc1_normalized.min():.3f}"),
                    html.P(f"Kurtosis: {pc1_normalized.kurtosis():.3f}",
                          style={'color': 'red' if pc1_normalized.kurtosis() > 3 else 'green'}),
                ])
            ], style={'border': '1px solid #bdc3c7'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("ℹ️ Enhanced Guide", className="card-title", style={'color': '#2c3e50'}),
                    html.P("🎨 Shaded bands show ±1.5σ range"),
                    html.P("📍 Red lines at ±2σ (extreme zones)"),
                    html.P("📈📉 Arrows mark extreme events"),
                    html.P("📊 30-day MA shows trend direction"),
                ])
            ], style={'border': '1px solid #bdc3c7'})
        ], width=3)
    ], className="mb-4")
    
    return html.Div([
        summary_cards,
        dcc.Graph(figure=main_fig),
        dbc.Alert([
            html.H6("💡 Time Series Interpretation Guide:", className="alert-heading"),
            html.Ul([
                html.Li("📊 Normalized PC scores show relative factor performance over time"),
                html.Li("🔄 Cumulative performance tracks factor evolution since start"),
                html.Li("📈 Rolling correlations reveal changing factor exposures"),
                html.Li("⚡ Volatility patterns show periods of market stress"),
                html.Li("🌍 Economic indicators provide macro context for factor movements")
            ])
        ], color="info", className="mt-4")
    ])

def create_detailed_tab(metrics, returns):
    """Enhanced detailed analysis with more comprehensive metrics"""
    loadings = metrics['loadings']
    communalities = metrics['communalities']
    stock_metrics = metrics['stock_metrics']
    
    # Enhanced Summary Statistics Table
    summary_data = []
    for stock in returns.columns:
        stock_data = stock_metrics.get(stock, {})
        row = {
            'Stock': stock,
            'Mean Return (%)': f"{returns[stock].mean() * 100:.3f}",
            'Volatility (%)': f"{stock_data.get('Volatility', returns[stock].std()) * 100:.2f}",
            'Sharpe Ratio': f"{stock_data.get('Sharpe_Ratio', 0):.3f}",
            'Sortino Ratio': f"{stock_data.get('Sortino_Ratio', 0):.3f}",
            'Max Drawdown (%)': f"{stock_data.get('Max_Drawdown', 0) * 100:.2f}",
            'VaR 95% (%)': f"{stock_data.get('VaR_95', 0) * 100:.2f}",
            'Skewness': f"{stock_data.get('Skewness', 0):.3f}",
            'Kurtosis': f"{stock_data.get('Kurtosis', 0):.3f}",
            'Communality': f"{communalities[stock]:.3f}",
            'Normality p-value': f"{stock_data.get('normality_p', 0):.3f}"
        }
        
        # Add loadings for each component
        for i, col in enumerate(loadings.columns):
            row[f'PC{i+1} Loading'] = f"{loadings.iloc[loadings.index.get_loc(stock), i]:.3f}"
        
        summary_data.append(row)
    
    summary_table = dash_table.DataTable(
        data=summary_data,
        columns=[{'name': col, 'id': col} for col in summary_data[0].keys()],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'fontSize': 11, 'padding': '8px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            # Highlight good Sharpe ratios
            {
                'if': {'column_id': 'Sharpe Ratio', 'filter_query': '{Sharpe Ratio} > 1'},
                'backgroundColor': '#d4edda',
                'color': 'black',
            },
            # Highlight high volatility
            {
                'if': {'column_id': 'Volatility (%)', 'filter_query': '{Volatility (%)} > 30'},
                'backgroundColor': '#f8d7da',
                'color': 'black',
            },
            # Highlight good communalities
            {
                'if': {'column_id': 'Communality', 'filter_query': '{Communality} > 0.7'},
                'backgroundColor': '#d1ecf1',
                'color': 'black',
            }
        ],
        sort_action='native',
        export_format='csv',
        export_headers='display'
    )
    
    # Enhanced Interpretation Guide
    interpretation = dbc.Card([
        dbc.CardHeader(html.H5("📋 Enhanced Interpretation Guide")),
        dbc.CardBody([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P("Factor loadings indicate how strongly each stock correlates with each principal component:"),
                    html.Ul([
                        html.Li("Values close to ±1: Strong correlation with the component"),
                        html.Li("Values near 0: Little relationship with the component"),
                        html.Li("Similar loadings: Stocks that move together"),
                        html.Li("Opposite signs: Stocks that move in opposite directions")
                    ])
                ], title="Factor Loadings"),
                
                dbc.AccordionItem([
                    html.P("Communalities show how well the principal components explain each stock:"),
                    html.Ul([
                        html.Li("High values (>0.7): Stock is well explained by the components"),
                        html.Li("Moderate values (0.5-0.7): Reasonable explanation"),
                        html.Li("Low values (<0.5): Unique stock-specific factors dominate")
                    ])
                ], title="Communalities"),
                
                dbc.AccordionItem([
                    html.P("Risk metrics help assess investment quality:"),
                    html.Ul([
                        html.Li("Sharpe Ratio >1: Good risk-adjusted returns"),
                        html.Li("Sortino Ratio: Focus on downside risk"),
                        html.Li("VaR 95%: Maximum expected loss 95% of the time"),
                        html.Li("Max Drawdown: Largest peak-to-trough decline")
                    ])
                ], title="Risk Metrics"),
                
                dbc.AccordionItem([
                    html.P("Statistical properties of returns:"),
                    html.Ul([
                        html.Li("Skewness >0: Right-skewed (more extreme positive returns)"),
                        html.Li("Skewness <0: Left-skewed (more extreme negative returns)"),
                        html.Li("Kurtosis >3: Fat tails (more extreme events)"),
                        html.Li("Normality p-value <0.05: Returns are not normally distributed")
                    ])
                ], title="Distribution Analysis")
            ], start_collapsed=True)
        ])
    ])
    
    # Portfolio insights
    portfolio_insights = create_portfolio_insights(loadings, stock_metrics, returns)
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("📊 Comprehensive Analysis Results"),
                summary_table
            ], width=8),
            dbc.Col([interpretation], width=4)
        ]),
        html.Hr(),
        portfolio_insights
    ])

def create_portfolio_insights(loadings, stock_metrics, returns):
    """Create portfolio insights and recommendations"""
    
    # Calculate portfolio-level metrics
    n_stocks = len(returns.columns)
    total_return = returns.mean().sum() * 252  # Annualized
    avg_volatility = np.mean([stock_metrics[stock].get('Volatility', 0) 
                             for stock in returns.columns if stock in stock_metrics])
    avg_sharpe = np.mean([stock_metrics[stock].get('Sharpe_Ratio', 0) 
                         for stock in returns.columns if stock in stock_metrics])
    
    # Identify factor clusters
    first_pc_loadings = loadings.iloc[:, 0].abs().sort_values(ascending=False)
    high_factor_stocks = first_pc_loadings.head(3).index.tolist()
    low_factor_stocks = first_pc_loadings.tail(3).index.tolist()
    
    insights_card = dbc.Card([
        dbc.CardHeader(html.H5("🎯 Portfolio Insights & Recommendations")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📈 Portfolio Overview", className="text-primary"),
                    html.P([
                        f"• Total stocks analyzed: {n_stocks}",
                        html.Br(),
                        f"• Average annualized volatility: {avg_volatility*100:.1f}%",
                        html.Br(),
                        f"• Average Sharpe ratio: {avg_sharpe:.2f}",
                        html.Br(),
                        f"• Expected portfolio return: {total_return*100:.1f}%"
                    ])
                ], width=6),
                dbc.Col([
                    html.H6("🔍 Factor Analysis", className="text-success"),
                    html.P([
                        f"• Stocks highly correlated with PC1: {', '.join(high_factor_stocks)}",
                        html.Br(),
                        f"• Stocks with unique factors: {', '.join(low_factor_stocks)}",
                        html.Br(),
                        f"• Diversification potential: {'High' if len(low_factor_stocks) > 2 else 'Moderate'}"
                    ])
                ], width=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("💡 Investment Recommendations", className="text-warning"),
                    create_investment_recommendations(stock_metrics, loadings, returns)
                ], width=12)
            ])
        ])
    ])
    
    return insights_card

def create_investment_recommendations(stock_metrics, loadings, returns):
    """Generate investment recommendations based on analysis"""
    recommendations = []
    
    # Find best risk-adjusted returns
    best_sharpe_stocks = sorted(stock_metrics.items(), 
                               key=lambda x: x[1].get('Sharpe_Ratio', 0), 
                               reverse=True)[:3]
    
    # Find most diversifying stocks (low correlation with PC1)
    diversifiers = loadings.iloc[:, 0].abs().sort_values().head(3)
    
    # Find conservative options (low volatility)
    conservative_stocks = sorted(stock_metrics.items(),
                                key=lambda x: x[1].get('Volatility', float('inf')))[:3]
    
    recommendations.extend([
        html.Li([
            html.Strong("🏆 Best Risk-Adjusted Returns: "),
            f"{', '.join([stock for stock, _ in best_sharpe_stocks])}"
        ]),
        html.Li([
            html.Strong("🎯 Best Diversifiers: "),
            f"{', '.join(diversifiers.index.tolist())}"
        ]),
        html.Li([
            html.Strong("🛡️ Conservative Options: "),
            f"{', '.join([stock for stock, _ in conservative_stocks])}"
        ])
    ])
    
    # Add specific warnings
    high_risk_stocks = [stock for stock, metrics in stock_metrics.items() 
                       if metrics.get('Volatility', 0) > 0.4]
    
    if high_risk_stocks:
        recommendations.append(
            html.Li([
                html.Strong("⚠️ High Risk Stocks: "),
                f"{', '.join(high_risk_stocks)} - Consider position sizing"
            ])
        )
    
    return html.Ul(recommendations)

def create_vargarch_tab(results):
    """Create advanced VAR/GARCH analysis tab"""
    if not ADVANCED_GARCH_AVAILABLE:
        return dbc.Alert([
            html.H5("🚀 Advanced VAR/GARCH Analysis Not Available"),
            html.P([
                "The advanced VAR/GARCH analysis module could not be loaded. ",
                "Please ensure that the GARCH_update.py file is available in the VAR/ directory."
            ]),
            html.Hr(),
            html.P("This feature provides:", className="mb-2"),
            html.Ul([
                html.Li("🔬 Advanced GARCH volatility modeling"),
                html.Li("📊 Vector Autoregression (VAR) analysis"), 
                html.Li("🔗 Granger causality testing"),
                html.Li("📈 Impulse response functions"),
                html.Li("📉 Forecast error variance decomposition")
            ])
        ], color="warning", className="m-4")
    
    vargarch_results = results.get('vargarch_results')
    
    if not vargarch_results:
        return dbc.Alert([
            html.H5("🚀 Advanced VAR/GARCH Analysis"),
            html.P([
                "Advanced VAR/GARCH analysis was not run. To enable this feature, please:",
                html.Br(),
                "• Enable 'Include Volatility Analysis' or 'Risk Metrics' options",
                html.Br(), 
                "• Re-run the analysis"
            ]),
            html.Hr(),
            html.P("This analysis provides:", className="mb-2"),
            html.Ul([
                html.Li("🔬 GARCH volatility modeling with persistence analysis"),
                html.Li("📊 VAR analysis for multi-asset dynamics"), 
                html.Li("🔗 Granger causality relationships"),
                html.Li("📈 Professional risk management insights"),
                html.Li("📉 Advanced portfolio recommendations")
            ])
        ], color="info", className="m-4")
    
    try:
        # Create visualizations using the advanced module
        visualizer = VARGARCHVisualizer()
        
        # Main dashboard
        if vargarch_results['var_results'] and vargarch_results['garch_results']:
            main_fig = visualizer.create_comprehensive_dashboard(
                vargarch_results['var_results'],
                vargarch_results['garch_results'], 
                results['returns']
            )
        else:
            main_fig = None
        
        # Volatility comparison if GARCH results available
        if vargarch_results['garch_results']:
            vol_fig = visualizer.create_volatility_comparison(
                vargarch_results['garch_results'],
                results['returns']
            )
        else:
            vol_fig = None
            
        # Granger causality heatmap if VAR results available
        if vargarch_results['var_results']:
            granger_fig = visualizer.create_granger_heatmap(vargarch_results['var_results'])
        else:
            granger_fig = None
        
        # Create interpretation cards
        interpretation_cards = create_interpretation_cards(
            vargarch_results['var_results'],
            vargarch_results['garch_results'],
            vargarch_results['summary_statistics']
        )
        
        # Build the tab content
        tab_content = [
            # Summary metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 VAR/GARCH Summary")),
                        dbc.CardBody([
                            html.P(f"🔢 Assets Analyzed: {vargarch_results['summary_statistics'].get('n_assets', 'N/A')}"),
                            html.P(f"📅 Analysis Period: {vargarch_results['summary_statistics'].get('date_range', 'N/A')}"),
                            html.P(f"🔗 VAR Optimal Lags: {vargarch_results['summary_statistics'].get('var_optimal_lags', 'N/A')}"),
                            html.P(f"📈 Significant Causalities: {vargarch_results['summary_statistics'].get('n_significant_causalities', 0)}"),
                            html.P(f"⚡ Avg GARCH Persistence: {vargarch_results['summary_statistics'].get('avg_garch_persistence', 0):.3f}")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("💡 Key Insights")),
                        dbc.CardBody(
                            create_insights_display(interpretation_cards['var_insights'] + interpretation_cards['garch_insights'])
                        )
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🎯 Recommendations")),
                        dbc.CardBody(
                            create_recommendations_display(
                                interpretation_cards['recommendations'] + 
                                vargarch_results.get('recommendations', [])
                            )
                        )
                    ])
                ], width=4)
            ], className="mb-4")
        ]
        
        # Add main dashboard if available
        if main_fig:
            tab_content.append(
                dbc.Card([
                    dbc.CardHeader(html.H5("🚀 Comprehensive VAR/GARCH Dashboard")),
                    dbc.CardBody([
                        dcc.Graph(figure=main_fig, className="mb-3")
                    ])
                ], className="mb-4")
            )
        
        # Add volatility comparison if available
        if vol_fig:
            tab_content.append(
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Advanced Volatility Analysis")),
                    dbc.CardBody([
                        dcc.Graph(figure=vol_fig, className="mb-3")
                    ])
                ], className="mb-4")
            )
        
        # Add Granger causality heatmap if available
        if granger_fig:
            tab_content.append(
                dbc.Card([
                    dbc.CardHeader(html.H5("🔗 Granger Causality Analysis")),
                    dbc.CardBody([
                        dcc.Graph(figure=granger_fig, className="mb-3"),
                        html.P([
                            "💡 This heatmap shows the p-values for Granger causality tests. ",
                            "Values below 0.05 (green) indicate significant causal relationships."
                        ], className="text-muted small")
                    ])
                ], className="mb-4")
            )
        
        # Add risk warnings if any
        if interpretation_cards['risk_warnings']:
            tab_content.append(
                dbc.Alert([
                    html.H5("⚠️ Risk Warnings", className="alert-heading"),
                    create_warnings_display(interpretation_cards['risk_warnings'])
                ], color="warning", className="mb-4")
            )
        
        return html.Div(tab_content)
        
    except Exception as e:
        logger.error(f"Error creating VAR/GARCH tab: {str(e)}")
        return dbc.Alert([
            html.H5("❌ Error Loading Advanced Analysis"),
            html.P(f"Error: {str(e)}"),
            html.P("The advanced VAR/GARCH analysis encountered an issue. Please try running the analysis again.")
        ], color="danger", className="m-4")

def create_insights_display(insights_list):
    """Create display for insights"""
    if not insights_list:
        return html.P("No specific insights available.", className="text-muted")
    
    return html.Div([
        html.Div([
            html.Strong(insight.get('title', 'Insight')),
            html.Br(),
            html.Span(insight.get('content', ''), className="small")
        ], className="mb-2") for insight in insights_list[:3]  # Show top 3 insights
    ])

def create_recommendations_display(recommendations_list):
    """Create display for recommendations"""
    if not recommendations_list:
        return html.P("No specific recommendations available.", className="text-muted")
    
    # Handle both string and dict recommendations
    formatted_recs = []
    for rec in recommendations_list[:3]:  # Show top 3 recommendations
        if isinstance(rec, str):
            formatted_recs.append(html.Li(rec, className="small"))
        elif isinstance(rec, dict):
            formatted_recs.append(html.Li([
                html.Strong(rec.get('title', 'Recommendation')),
                ": ",
                rec.get('content', '')
            ], className="small"))
    
    return html.Ul(formatted_recs) if formatted_recs else html.P("No recommendations available.", className="text-muted")

def create_warnings_display(warnings_list):
    """Create display for warnings"""
    if not warnings_list:
        return html.P("No warnings.", className="text-success")
    
    return html.Div([
        html.Div([
            html.Strong(warning.get('title', 'Warning')),
            html.Br(),
            html.Span(warning.get('content', ''), className="small")
        ], className="mb-2") for warning in warnings_list
    ])

def create_garch_parameters_table(garch_results):
    """Create a table displaying GARCH model parameters"""
    if not garch_results:
        return html.Div()
    
    # Prepare data for the table
    table_data = []
    for stock, result in garch_results.items():
        if 'parameters' in result and result['parameters'].get('converged'):
            params = result['parameters']
            table_data.append({
                'Stock': stock,
                'ω (Omega)': f"{params.get('omega', 0):.6f}",
                'α (Alpha)': f"{params.get('alpha', 0):.4f}",
                'β (Beta)': f"{params.get('beta', 0):.4f}",
                'Persistence (α+β)': f"{params.get('persistence', 0):.4f}",
                'Log Likelihood': f"{params.get('log_likelihood', 0):.2f}",
                'Converged': "✅" if params.get('converged') else "❌"
            })
        else:
            table_data.append({
                'Stock': stock,
                'ω (Omega)': "Failed",
                'α (Alpha)': "Failed",
                'β (Beta)': "Failed",
                'Persistence (α+β)': "Failed",
                'Log Likelihood': "Failed",
                'Converged': "❌"
            })
    
    if not table_data:
        return html.Div()
    
    return html.Div([
        html.H5("📊 GARCH Model Parameters", className="mb-3"),
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontSize': 12, 'padding': '8px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'Persistence (α+β)', 'filter_query': '{Persistence (α+β)} > 0.95'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Converged', 'filter_query': '{Converged} = ✅'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                }
            ],
            sort_action='native'
        ),
        html.P([
            "💡 GARCH Parameters Interpretation:",
            html.Br(),
            "• ω (omega): Baseline volatility level",
            html.Br(), 
            "• α (alpha): Impact of recent shocks on volatility",
            html.Br(),
            "• β (beta): Volatility persistence from previous periods",
            html.Br(),
            "• Persistence (α+β): Total volatility persistence (should be < 1 for stationarity)",
            html.Br(),
            "• High persistence (>0.95) indicates volatility clustering"
        ], className="text-muted small mt-3")
    ])

def create_lstm_tab(results):
    """Create LSTM prediction analysis tab with loading state"""
    if not LSTM_AVAILABLE:
        return dbc.Alert([
            html.H5("🧠 LSTM Predictions Not Available", className="alert-heading"),
            html.P([
                "LSTM functionality is not available. This may be due to missing dependencies. ",
                "Ensure TensorFlow/Keras is installed: ",
                html.Code("pip install tensorflow"),
            ]),
            html.Hr(),
            html.P("Once installed, restart the application to enable LSTM predictions.")
        ], color="warning", className="mt-4")
    
    # Return LSTM container with loading state
    return html.Div([
        # Initial loading state
        html.Div([
            dbc.Alert([
                html.H5("🧠 Preparing LSTM Neural Network Analysis", className="alert-heading mb-3"),
                html.P("Please wait while we train the LSTM model. This process may take 1-3 minutes depending on your system.", 
                       className="mb-3"),
                
                # Enhanced loading animation
                html.Div([
                    dbc.Spinner(
                        size="lg", 
                        color="primary",
                        spinner_style={"width": "3rem", "height": "3rem"}
                    ),
                    html.Div([
                        html.H6("🔄 Training Progress:", className="mt-3 mb-2"),
                        dbc.Progress(
                            id="lstm-progress",
                            value=0,
                            striped=True,
                            animated=True,
                            color="primary",
                            className="mb-2"
                        ),
                        html.P(id="lstm-status", children="Initializing LSTM model...", 
                               className="text-muted small")
                    ], className="mt-3")
                ], className="text-center"),
                
                # Information about LSTM processing
                html.Hr(),
                html.Div([
                    html.H6("📊 What's happening:", className="mb-2"),
                    html.Ul([
                        html.Li("🔧 Preparing multivariate time series data"),
                        html.Li("🧠 Building and compiling LSTM neural network"),
                        html.Li("📈 Training model on historical patterns"),
                        html.Li("🔮 Generating future predictions with confidence intervals"),
                        html.Li("📊 Creating comprehensive visualizations")
                    ], className="small")
                ], className="mt-3")
            ], color="info", className="mb-4"),
            
            # Progress tracking interval
            dcc.Interval(
                id="lstm-progress-interval",
                interval=500,  # Update every 500ms
                n_intervals=0,
                max_intervals=240  # Max 2 minutes
            )
        ], id="lstm-loading-container"),
        
        # Container for results (initially hidden)
        html.Div(id="lstm-results-container", style={"display": "none"})
    ])

def train_enhanced_lstm_for_dashboard(data: pd.DataFrame, 
                                    target_col: str,
                                    config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Optimized training function for the new enhanced LSTM model
    """
    try:
        # Default optimized configuration for dashboard use
        default_config = {
            'seq_length': 20,  # Reduced for faster training
            'lstm_units': [64, 32],  # Smaller units for speed
            'dense_units': [32, 16],  # Smaller dense layers
            'dropout_rate': 0.15,
            'learning_rate': 0.002,
            'use_attention': True,  # Use attention for better performance
            'use_bidirectional': False,  # Disable for speed
            'feature_selection': True,
            'n_features': 15  # Limit features for speed
        }
        
        # Update with user config
        if config:
            default_config.update(config)
        
        logger.info("Creating enhanced LSTM predictor for dashboard...")
        predictor = EnhancedLSTMPredictor(**default_config)
        
        # Train with optimized parameters for dashboard
        training_results = predictor.train(
            data=data,
            target_col=target_col,
            validation_split=0.15,
            epochs=25,  # Reduced epochs for speed
            batch_size=32,
            patience=8  # Reduced patience
        )
        
        # Calculate metrics
        metrics = {}
        if training_results.get('val_actual') is not None and training_results.get('val_predicted') is not None:
            try:
                metrics = predictor.calculate_metrics(
                    training_results['val_actual'], 
                    training_results['val_predicted']
                )
            except Exception as e:
                logger.warning(f"Metrics calculation failed: {e}")
                metrics = {}
        
        # Generate future predictions
        try:
            future_pred, conf_lower, conf_upper = predictor.predict_future(
                data=data,
                target_col=target_col,
                days_ahead=15  # Reduced horizon for speed
            )
        except Exception as e:
            logger.warning(f"Future predictions failed: {e}")
            # Return dummy predictions
            future_pred = np.zeros(15)
            conf_lower = np.zeros(15)
            conf_upper = np.zeros(15)
        
        return {
            'predictor': predictor,
            'training_results': training_results,
            'metrics': metrics,
            'future_predictions': future_pred,
            'confidence_lower': conf_lower,
            'confidence_upper': conf_upper
        }
        
    except Exception as e:
        logger.error(f"Enhanced LSTM training failed: {e}")
        # Return minimal fallback structure
        return {
            'predictor': None,
            'training_results': {},
            'metrics': {},
            'future_predictions': np.zeros(15),
            'confidence_lower': np.zeros(15),
            'confidence_upper': np.zeros(15)
        }

def create_lstm_results(results):
    """Create the actual LSTM results content (separated for loading)"""
    
    try:
        # Get the stock data from results
        stock_data = results.get('data')
        returns_data = results.get('returns')
        
        if stock_data is None or returns_data is None:
            return dbc.Alert([
                html.H5("📊 No Data Available for LSTM"),
                html.P("Stock data is required for LSTM predictions.")
            ], color="warning")
        
        # Prepare data for LSTM (use the first stock as target for demo)
        target_stock = stock_data.columns[0] if len(stock_data.columns) > 0 else None
        
        if target_stock is None:
            return dbc.Alert([
                html.H5("📊 No Target Stock Available"),
                html.P("At least one stock is required for LSTM predictions.")
            ], color="warning")
        
        # Create a multivariate DataFrame for LSTM
        lstm_data = stock_data.copy()
        lstm_data.index = pd.to_datetime(lstm_data.index)
        
        # Configure enhanced LSTM parameters for dashboard (updated for new API)
        lstm_config = {
            'seq_length': 15,  # Reduced for faster training
            'lstm_units': [32, 16],  # Two smaller layers
            'dense_units': [16],  # Single dense layer
            'dropout_rate': 0.15,
            'learning_rate': 0.003,
            'use_attention': False,  # Disable attention for speed
            'use_bidirectional': False,  # Disabled for speed
            'feature_selection': True,
            'n_features': 12  # Limit features for speed
        }
        
        # Enhanced LSTM training and prediction
        logger.info(f"Training enhanced LSTM model for {target_stock}")
        lstm_results = train_enhanced_lstm_for_dashboard(
            data=lstm_data,
            target_col=target_stock,
            config=lstm_config
        )
        
        predictor = lstm_results['predictor']
        training_results = lstm_results['training_results']
        metrics = lstm_results['metrics']
        future_pred = lstm_results['future_predictions']
        conf_lower = lstm_results['confidence_lower']
        conf_upper = lstm_results['confidence_upper']
        
        # Create visualizations
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"📈 {target_stock} - Actual vs Predicted Prices",
                "🎯 Training History (Loss)",
                "🔮 15-Day Future Predictions with Confidence Intervals"
            ],
            vertical_spacing=0.1
        )
        
        # Plot 1: Actual vs Predicted (updated for new API)
        if training_results.get('val_actual') is not None and training_results.get('val_predicted') is not None:
            val_actual = training_results['val_actual']
            val_predicted = training_results['val_predicted']
            test_dates = lstm_data.index[-len(val_actual):] if len(val_actual) <= len(lstm_data) else lstm_data.index[-len(val_actual):]
            
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=val_actual,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=val_predicted,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=1
            )
        
        # Plot 2: Training history
        if training_results.get('history'):
            history = training_results['history']
            epochs = list(range(1, len(history.history['loss']) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.history['loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            if 'val_loss' in history.history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
        
        # Plot 3: Future predictions (updated for 15 days)
        future_dates = pd.date_range(start=lstm_data.index[-1] + pd.Timedelta(days=1), periods=15, freq='D')
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_pred,
                mode='lines',
                name='Future Prediction',
                line=dict(color='purple', width=3)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=conf_upper,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='lightgray', width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=conf_lower,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='lightgray', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)',
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            title=f"🧠 LSTM Analysis for {target_stock}",
            showlegend=True
        )
        
        # Create metrics cards
        metrics_cards = []
        if metrics:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics_cards.append(
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6(metric_name.replace('_', ' ').title(), className="text-muted mb-1"),
                                    html.H4(f"{metric_value:.4f}", className="text-primary mb-0")
                                ])
                            ], className="text-center shadow-sm")
                        ], width=2)
                    )
        
        # Model summary (updated for new API)
        model_summary = "No model summary available"
        if predictor and hasattr(predictor, 'model') and predictor.model:
            try:
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    predictor.model.summary()
                model_summary = f.getvalue()
            except Exception as e:
                logger.warning(f"Could not get model summary: {e}")
                model_summary = f"Enhanced LSTM Model\nParameters: {predictor.model.count_params():,}" if predictor.model else "Model not available"
        
        return html.Div([
            dbc.Alert([
                html.H5("🧠 LSTM Neural Network Predictions", className="alert-heading"),
                html.P([
                    f"Advanced LSTM model trained on {target_stock} with multivariate features. ",
                    "The model uses sequence learning to predict future price movements with confidence intervals."
                ])
            ], color="info", className="mb-4"),
            
            # Metrics row
            dbc.Row(metrics_cards, className="mb-4") if metrics_cards else html.Div(),
            
            # Main chart
            dcc.Graph(figure=fig, className="mb-4"),
            
            # Model details
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🔧 Model Configuration", className="mb-0")),
                        dbc.CardBody([
                            html.P(f"Sequence Length: {lstm_config['seq_length']} days"),
                            html.P(f"LSTM Units: {lstm_config['lstm_units']}"),
                            html.P(f"Dense Units: {lstm_config['dense_units']}"),
                            html.P(f"Dropout Rate: {lstm_config['dropout_rate']}"),
                            html.P(f"Learning Rate: {lstm_config['learning_rate']}"),
                            html.P(f"Attention: {'Yes' if lstm_config.get('use_attention', False) else 'No'}"),
                            html.P(f"Features: {lstm_config.get('n_features', 'Auto')}")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 Model Architecture", className="mb-0")),
                        dbc.CardBody([
                            html.Pre(model_summary, style={'fontSize': '10px', 'height': '200px', 'overflow': 'auto'})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Interpretation
            dbc.Alert([
                html.H6("💡 Enhanced LSTM Model Interpretation:", className="alert-heading"),
                html.Ul([
                    html.Li("Enhanced LSTM with advanced feature engineering and attention mechanisms"),
                    html.Li("Technical indicators and price patterns automatically extracted"),
                    html.Li("Feature selection optimizes model performance and reduces overfitting"),
                    html.Li("Confidence intervals provide uncertainty estimates for risk management"),
                    html.Li("Model includes regularization and early stopping for better generalization"),
                    html.Li("Use predictions as directional guidance combined with fundamental analysis")
                ])
            ], color="light", className="mt-4")
        ])
        
    except Exception as e:
        logger.error(f"Error in LSTM analysis: {str(e)}")
        return dbc.Alert([
            html.H5("⚠️ LSTM Analysis Error", className="alert-heading"),
            html.P(f"Error occurred during LSTM analysis: {str(e)}"),
            html.P("This may be due to insufficient data or model configuration issues.")
        ], color="danger", className="mt-4")

# Global variables to track processing
lstm_results_cache = None
timeseries_results_cache = None
timeseries_cache_key = None

@app.callback(
    [Output("lstm-loading-container", "style"),
     Output("lstm-results-container", "children"),
     Output("lstm-results-container", "style"),
     Output("lstm-progress", "value"),
     Output("lstm-status", "children")],
    [Input("lstm-progress-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_lstm_progress(n_intervals):
    """Handle LSTM progress and results loading"""
    global lstm_results_cache, pca_results
    
    # Check if we have PCA results and should process LSTM
    if pca_results is None:
        return {"display": "block"}, "", {"display": "none"}, 0, "Waiting for analysis results..."
    
    # Progress simulation (since LSTM training can't easily report real progress)
    progress_value = min(n_intervals * 5, 100)  # Increment by 5% every 500ms
    
    # Status messages based on progress
    if progress_value < 20:
        status = "🔧 Preparing multivariate time series data..."
    elif progress_value < 40:
        status = "🧠 Building and compiling LSTM neural network..."
    elif progress_value < 70:
        status = "📈 Training model on historical patterns..."
    elif progress_value < 90:
        status = "🔮 Generating future predictions..."
    else:
        status = "📊 Creating visualizations..."
    
    # If we haven't processed LSTM yet and progress is complete, do it now
    if lstm_results_cache is None and progress_value >= 100:
        try:
            lstm_results_cache = create_lstm_results(pca_results)
            return {"display": "none"}, lstm_results_cache, {"display": "block"}, 100, "✅ LSTM analysis complete!"
        except Exception as e:
            error_content = dbc.Alert([
                html.H5("⚠️ LSTM Analysis Error", className="alert-heading"),
                html.P(f"Error occurred during LSTM analysis: {str(e)}"),
                html.P("This may be due to insufficient data or model configuration issues.")
            ], color="danger", className="mt-4")
            return {"display": "none"}, error_content, {"display": "block"}, 100, "❌ Error in LSTM analysis"
    
    # If results are cached, show them
    if lstm_results_cache is not None:
        return {"display": "none"}, lstm_results_cache, {"display": "block"}, 100, "✅ LSTM analysis complete!"
    
    # Continue showing progress
    return {"display": "block"}, "", {"display": "none"}, progress_value, status

def create_error_alert():
    """Create enhanced error alert component"""
    return dbc.Alert(
        id="error-alert",
        color="danger",
        dismissable=True,
        is_open=False,
        className="mt-3 shadow-sm",
        style={"borderRadius": "12px"}
    )

def create_enhanced_loading_component():
    """Create sophisticated loading modal with progress tracking"""
    return dbc.Modal([
        dbc.ModalBody([
            html.Div([
                # Main spinner and title section
                html.Div([
                    dbc.Spinner(
                        size="lg", 
                        color="primary", 
                        spinner_style={
                            "width": "4rem", 
                            "height": "4rem",
                            "borderWidth": "4px"
                        }
                    ),
                    html.H4([
                        html.I(className="fas fa-microscope me-2"),
                        "Analyzing Your Portfolio"
                    ], className="mt-3 text-center text-primary fw-bold"),
                    html.P("Advanced financial analysis in progress...", 
                          className="text-center text-muted mb-0")
                ], className="text-center mb-4"),
                
                # Progress bar section
                html.Div([
                    dbc.Progress(
                        id="loading-progress", 
                        value=0,
                        striped=True, 
                        animated=True, 
                        color="info",
                        style={"height": "12px", "borderRadius": "10px"}
                    ),
                    html.Div(
                        id="loading-status", 
                        className="text-center mt-2 text-muted fw-bold"
                    ),
                ], className="mb-4"),
                
                # Processing steps visualization
                html.Div([
                    html.H6([
                        html.I(className="fas fa-cogs me-2"),
                        "Processing Steps:"
                    ], className="mb-3 text-primary"),
                    html.Div(id="loading-steps", children=[
                        html.Div([
                            html.I(className="fas fa-circle-notch fa-spin me-2 text-primary", id="step1-icon"),
                            html.Span("📊 Fetching market data...", id="step1-text", className="fw-bold")
                        ], className="mb-2 p-2 bg-light rounded", id="step1"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step2-icon"),
                            html.Span("🧮 Calculating enhanced metrics...", id="step2-text", className="text-muted")
                        ], className="mb-2 p-2", id="step2"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step3-icon"),
                            html.Span("🔬 Performing advanced analysis...", id="step3-text", className="text-muted")
                        ], className="mb-2 p-2", id="step3"),
                        html.Div([
                            html.I(className="fas fa-circle me-2 text-muted", id="step4-icon"),
                            html.Span("📈 Creating visualizations...", id="step4-text", className="text-muted")
                        ], className="mb-2 p-2", id="step4"),
                    ])
                ], className="border rounded p-3 bg-light"),
                
                # Tips and estimated time section
                html.Div([
                    html.Div([
                        html.I(className="fas fa-clock me-1 text-info"),
                        html.Span(id="estimated-time", children="⏱️ Estimated time: 15-45 seconds")
                    ], className="text-muted mb-2"),
                    html.Div([
                        html.I(className="fas fa-lightbulb me-1 text-warning"),
                        html.Span("💡 Tip: More stocks and features = longer processing time")
                    ], className="text-info mb-2"),
                    html.Div([
                        html.I(className="fas fa-coffee me-1 text-success"),
                        html.Span("☕ Perfect time for a quick coffee break!")
                    ], className="text-success")
                ], className="mt-3 text-center small")
            ], className="p-4")
        ])
    ], 
    id="loading-modal", 
    is_open=False, 
    backdrop="static", 
    keyboard=False, 
    size="md",
    style={"borderRadius": "20px"}
    )

def add_interval_component():
    """Add interval component for real-time progress tracking"""
    return dcc.Interval(
        id='loading-interval',
        interval=500,  # Update every 500ms for smooth progress
        n_intervals=0,
        disabled=True,
        max_intervals=25  # Prevent infinite running
    )

# Additional utility functions for enhanced functionality
def create_feature_badges():
    """Create feature highlight badges for the header"""
    return html.Div([
        dbc.Badge([
            html.I(className="fas fa-star me-1"),
            "Enhanced Volatility Analysis"
        ], color="info", className="me-2 mb-2"),
        dbc.Badge([
            html.I(className="fas fa-brain me-1"),
            "ML-Powered Insights"
        ], color="success", className="me-2 mb-2"),
        dbc.Badge([
            html.I(className="fas fa-shield-alt me-1"),
            "Advanced Risk Metrics"
        ], color="warning", className="me-2 mb-2"),
        dbc.Badge([
            html.I(className="fas fa-chart-line me-1"),
            "Economic Indicators"
        ], color="primary", className="me-2 mb-2"),
        dbc.Badge([
            html.I(className="fas fa-magic me-1"),
            "PCA Factor Analysis"
        ], color="secondary", className="me-2 mb-2")
    ], className="text-center mb-3")

def create_quick_start_guide():
    """Create a quick start guide modal"""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-rocket me-2"),
            "🚀 Quick Start Guide"
        ])),
        dbc.ModalBody([
            html.H6("1️⃣ Select Your Stocks", className="text-primary"),
            html.P("Choose from presets or enter custom tickers (e.g., AAPL,MSFT,GOOGL)"),
            
            html.H6("2️⃣ Configure Analysis", className="text-success"),
            html.P("Set components, date range, and enable advanced features"),
            
            html.H6("3️⃣ Run Analysis", className="text-warning"),
            html.P("Click 'Run Enhanced Analysis' and wait for results"),
            
            html.H6("4️⃣ Explore Results", className="text-info"),
            html.P("Navigate through tabs to discover insights and recommendations"),
            
            html.Hr(),
            dbc.Alert([
                html.I(className="fas fa-lightbulb me-2"),
                "💡 Tip: Start with 'Demo Data' to see all features in action!"
            ], color="info")
        ]),
        dbc.ModalFooter([
            dbc.Button("Got it!", id="close-guide", className="ms-auto", color="primary")
        ])
    ], id="quick-start-modal", size="lg")

# Add this to your layout function calls

# Enhanced CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Enhanced Dashboard Styling */
            .card { 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                transition: all 0.3s ease;
                border: none;
                border-radius: 12px;
                margin-bottom: 1rem;
            }
            .card:hover { 
                box-shadow: 0 8px 15px rgba(0,0,0,0.15); 
                transform: translateY(-2px);
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #007bff, #0056b3);
                border: none;
                transition: all 0.3s ease;
                border-radius: 8px;
                font-weight: 600;
            }
            .btn-primary:hover {
                background: linear-gradient(135deg, #0056b3, #004085);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,123,255,0.4);
            }
            
            /* Tab enhancements */
            .nav-tabs .nav-link {
                border-radius: 12px 12px 0 0;
                transition: all 0.3s ease;
                margin-right: 5px;
                font-weight: 600;
                color: #6c757d;
            }
            .nav-tabs .nav-link:hover {
                background-color: rgba(0,123,255,0.1);
                transform: translateY(-2px);
                color: #007bff;
            }
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white !important;
                border-color: #007bff;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,123,255,0.3);
            }
            
            /* Loading animations */
            @keyframes pulse-glow {
                0% { box-shadow: 0 0 20px rgba(0,123,255,0.2); }
                50% { box-shadow: 0 0 40px rgba(0,123,255,0.4); }
                100% { box-shadow: 0 0 20px rgba(0,123,255,0.2); }
            }
            
            .modal-content {
                border-radius: 20px;
                border: none;
                animation: pulse-glow 3s infinite;
            }
            
            /* Chart enhancements */
            .js-plotly-plot {
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            .js-plotly-plot:hover {
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            
            /* Form styling */
            .form-control, .form-select {
                border-radius: 8px;
                border: 2px solid #e9ecef;
                transition: all 0.3s ease;
            }
            .form-control:focus, .form-select:focus {
                border-color: #007bff;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
                transform: translateY(-1px);
            }
            
            /* Progress bar */
            .progress {
                height: 12px;
                border-radius: 10px;
                background-color: rgba(0,123,255,0.1);
                overflow: hidden;
            }
            .progress-bar {
                border-radius: 10px;
                background: linear-gradient(45deg, #007bff, #0056b3);
                background-size: 40px 40px;
                animation: progress-bar-stripes 1s linear infinite;
            }
            
            @keyframes progress-bar-stripes {
                0% { background-position: 40px 0; }
                100% { background-position: 0 0; }
            }
            
            /* Data table enhancements */
            .dash-table-container {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            /* Badge styling */
            .badge {
                border-radius: 20px;
                font-weight: 500;
                padding: 0.5rem 1rem;
            }
            
            /* Accordion styling */
            .accordion-button {
                border-radius: 8px !important;
                font-weight: 600;
            }
            
            /* Summary cards */
            .card-body h3 {
                font-size: 2rem;
                font-weight: 700;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .card-body h3 { font-size: 1.5rem; }
                .btn-lg { font-size: 1rem; padding: 0.75rem 1.5rem; }
                h1 { font-size: 1.75rem; }
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #007bff, #0056b3);
                border-radius: 10px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #0056b3, #004085);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == "__main__":
    # Try different ports if 8050 is busy
    import socket
    
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    # Find available port
    ports_to_try = [8050, 8051, 8052, 8053, 8054]
    port = None
    
    for p in ports_to_try:
        if is_port_available(p):
            port = p
            break
    
    if port:
        print(f"🚀 Starting Enhanced PCA Dashboard on http://localhost:{port}")
        print("=" * 60)
        print("📊 Dashboard Features:")
        print("  • Advanced PCA Analysis")
        print("  • Enhanced Volatility Modeling")
        print("  • Comprehensive Risk Assessment")
        print("  • Professional Visualizations")
        print("=" * 60)
        app.run(debug=True, port=port)
    else:
        print("❌ No available ports found. Please close other Dash apps or use:")
        print("   app.run(debug=True, port=8055)  # or any other port")



        
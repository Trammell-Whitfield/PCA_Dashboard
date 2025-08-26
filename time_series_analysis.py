# Enhanced Time Series Analysis Module
# Extracted and improved from main PCA dashboard

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

# Optional dependency for structural break detection
try:
    from ruptures import Binseg #ignore 
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ruptures package not available for structural break detection")

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EnhancedTimeSeriesAnalyzer:
    """
    Advanced time series analysis class for PCA factor scores and stock data
    """
    
    def __init__(self, window_short: int = 30, window_long: int = 60, volatility_window: int = 20):
        """
        Initialize the time series analyzer
        
        Args:
            window_short: Short-term rolling window for calculations
            window_long: Long-term rolling window for calculations  
            volatility_window: Window for volatility calculations
        """
        self.window_short = window_short
        self.window_long = window_long
        self.volatility_window = volatility_window
        
    def analyze_factor_scores(self, factor_scores: pd.DataFrame, 
                            explained_variance: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive analysis of PCA factor scores over time
        
        Args:
            factor_scores: DataFrame with factor scores (PC1, PC2, etc.)
            explained_variance: Array of explained variance ratios for each component
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # 1. Normalize factor scores for better visualization
        normalized_scores = self._normalize_scores(factor_scores)
        results['normalized_scores'] = normalized_scores
        
        # 2. Calculate rolling statistics
        rolling_stats = self._calculate_rolling_statistics(factor_scores)
        results['rolling_stats'] = rolling_stats
        
        # 3. Detect regime changes and extreme events
        regime_analysis = self._analyze_regimes(normalized_scores)
        results['regime_analysis'] = regime_analysis
        
        # 4. Calculate factor momentum and mean reversion
        momentum_analysis = self._analyze_momentum(factor_scores)
        results['momentum_analysis'] = momentum_analysis
        
        # 5. Analyze factor stability and persistence
        stability_analysis = self._analyze_stability(factor_scores)
        results['stability_analysis'] = stability_analysis
        
        # 6. Cross-factor relationships
        cross_factor_analysis = self._analyze_cross_factors(factor_scores)
        results['cross_factor_analysis'] = cross_factor_analysis
        
        # 7. Generate summary statistics
        summary_stats = self._generate_summary_statistics(
            factor_scores, normalized_scores, explained_variance
        )
        results['summary_stats'] = summary_stats
        
        return results
    
    def analyze_stock_correlations(self, returns: pd.DataFrame, 
                                 factor_scores: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how individual stock correlations with factors evolve over time
        
        Args:
            returns: Stock returns DataFrame
            factor_scores: Factor scores DataFrame
            
        Returns:
            Dictionary with correlation analysis results
        """
        results = {}
        
        # Rolling correlations with each factor
        rolling_correlations = {}
        for factor in factor_scores.columns:
            factor_correls = {}
            for stock in returns.columns:
                rolling_corr = returns[stock].rolling(
                    window=self.window_long, min_periods=self.window_short
                ).corr(factor_scores[factor])
                factor_correls[stock] = rolling_corr
            rolling_correlations[factor] = pd.DataFrame(factor_correls)
        
        results['rolling_correlations'] = rolling_correlations
        
        # Correlation stability analysis
        corr_stability = self._analyze_correlation_stability(rolling_correlations)
        results['correlation_stability'] = corr_stability
        
        # Factor loadings drift analysis
        loadings_drift = self._analyze_loadings_drift(returns, factor_scores)
        results['loadings_drift'] = loadings_drift
        
        return results
    
    def analyze_economic_relationships(self, factor_scores: pd.DataFrame,
                                     economic_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze relationships between factors and economic indicators
        
        Args:
            factor_scores: Factor scores DataFrame
            economic_data: Economic indicators DataFrame (optional)
            
        Returns:
            Dictionary with economic relationship analysis
        """
        results = {}
        
        if economic_data is None or len(economic_data) == 0:
            results['available'] = False
            results['message'] = "No economic data available for analysis"
            return results
        
        results['available'] = True
        
        # Align data by dates
        common_dates = factor_scores.index.intersection(economic_data.index)
        if len(common_dates) < 30:
            results['available'] = False
            results['message'] = "Insufficient overlapping data points"
            return results
        
        aligned_factors = factor_scores.loc[common_dates]
        aligned_econ = economic_data.loc[common_dates]
        
        # Rolling correlations with economic indicators
        econ_correlations = {}
        for factor in aligned_factors.columns:
            factor_econ_corr = {}
            for econ_indicator in aligned_econ.columns:
                rolling_corr = aligned_factors[factor].rolling(
                    window=self.window_long, min_periods=self.window_short
                ).corr(aligned_econ[econ_indicator])
                factor_econ_corr[econ_indicator] = rolling_corr
            econ_correlations[factor] = pd.DataFrame(factor_econ_corr)
        
        results['economic_correlations'] = econ_correlations
        
        # Lead-lag relationships
        lead_lag_analysis = self._analyze_lead_lag_relationships(
            aligned_factors, aligned_econ
        )
        results['lead_lag_analysis'] = lead_lag_analysis
        
        # Economic regime classification
        regime_classification = self._classify_economic_regimes(
            aligned_factors, aligned_econ
        )
        results['regime_classification'] = regime_classification
        
        return results
    
    def create_enhanced_visualizations(self, analysis_results: Dict[str, Any],
                                     returns: pd.DataFrame,
                                     economic_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create comprehensive time series visualizations
        
        Args:
            analysis_results: Results from factor score analysis
            returns: Stock returns DataFrame
            economic_data: Economic indicators DataFrame (optional)
            
        Returns:
            Plotly figure with multiple subplots
        """
        # Create subplot structure
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                "📊 Enhanced Factor Scores with Regimes",
                "🔄 Factor Momentum & Mean Reversion", 
                "📈 Rolling Volatility & Stability",
                "🌊 Cross-Factor Relationships",
                "📉 Stock-Factor Correlations Evolution",
                "🌍 Economic Indicators Impact",
                "⚡ Extreme Events & Recovery",
                "🎯 Factor Performance Attribution"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Extract analysis components
        normalized_scores = analysis_results['normalized_scores']
        rolling_stats = analysis_results['rolling_stats']
        regime_analysis = analysis_results['regime_analysis']
        momentum_analysis = analysis_results['momentum_analysis']
        stability_analysis = analysis_results['stability_analysis']
        cross_factor_analysis = analysis_results['cross_factor_analysis']
        
        # Color schemes
        colors = px.colors.qualitative.Set2
        regime_colors = {'bull': 'green', 'bear': 'red', 'neutral': 'gray'}
        
        # Plot 1: Enhanced Factor Scores with Regimes
        self._plot_factor_scores_with_regimes(
            fig, normalized_scores, regime_analysis, colors, 1, 1
        )
        
        # Plot 2: Factor Momentum & Mean Reversion
        self._plot_momentum_analysis(
            fig, momentum_analysis, colors, 1, 2
        )
        
        # Plot 3: Rolling Volatility & Stability
        self._plot_volatility_stability(
            fig, rolling_stats, stability_analysis, colors, 2, 1
        )
        
        # Plot 4: Cross-Factor Relationships
        self._plot_cross_factor_relationships(
            fig, cross_factor_analysis, colors, 2, 2
        )
        
        # Plot 5: Stock-Factor Correlations Evolution
        if len(returns.columns) > 0:
            self._plot_correlation_evolution(
                fig, returns, normalized_scores, colors, 3, 1
            )
        
        # Plot 6: Economic Indicators Impact
        self._plot_economic_impact(
            fig, normalized_scores, economic_data, colors, 3, 2
        )
        
        # Plot 7: Extreme Events & Recovery
        self._plot_extreme_events(
            fig, normalized_scores, regime_analysis, colors, 4, 1
        )
        
        # Plot 8: Factor Performance Attribution
        self._plot_performance_attribution(
            fig, analysis_results, colors, 4, 2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title={
                'text': "🎯 Advanced Time Series Analysis - Factor Evolution & Market Dynamics",
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=10),
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=9)
            )
        )
        
        # Add range selector for time navigation
        fig.update_xaxes(
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
                font=dict(size=9)
            ),
            row=1, col=1
        )
        
        return fig
    
    def generate_insights_report(self, analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate actionable insights from the time series analysis
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            List of insight dictionaries with titles and descriptions
        """
        insights = []
        
        try:
            summary_stats = analysis_results.get('summary_stats', {})
            regime_analysis = analysis_results.get('regime_analysis', {})
            momentum_analysis = analysis_results.get('momentum_analysis', {})
            stability_analysis = analysis_results.get('stability_analysis', {})
            
            # Factor Performance Insights
            if summary_stats:
                best_factor = summary_stats.get('best_performing_factor', 'PC1')
                worst_factor = summary_stats.get('worst_performing_factor', 'PC2')
                
                insights.append({
                    'type': 'performance',
                    'title': f'🏆 Best Performing Factor: {best_factor}',
                    'description': f'{best_factor} shows the strongest risk-adjusted performance with highest Sharpe ratio.',
                    'severity': 'success'
                })
                
                insights.append({
                    'type': 'performance', 
                    'title': f'⚠️ Underperforming Factor: {worst_factor}',
                    'description': f'{worst_factor} shows weaker performance and may need attention in portfolio allocation.',
                    'severity': 'warning'
                })
            
            # Regime Insights
            if regime_analysis:
                current_regime = regime_analysis.get('current_regime', 'neutral')
                regime_stability = regime_analysis.get('regime_stability', 0.5)
                
                regime_message = {
                    'bull': 'Current market regime is bullish with positive factor momentum.',
                    'bear': 'Current market regime is bearish with negative factor trends.',
                    'neutral': 'Current market regime is neutral with mixed signals.'
                }.get(current_regime, 'Market regime is unclear.')
                
                insights.append({
                    'type': 'regime',
                    'title': f'📊 Market Regime: {current_regime.title()}',
                    'description': f'{regime_message} Stability score: {regime_stability:.2f}',
                    'severity': 'info' if current_regime == 'neutral' else ('success' if current_regime == 'bull' else 'danger')
                })
            
            # Momentum Insights
            if momentum_analysis:
                strong_momentum_factors = momentum_analysis.get('strong_momentum_factors', [])
                mean_reverting_factors = momentum_analysis.get('mean_reverting_factors', [])
                
                if strong_momentum_factors:
                    insights.append({
                        'type': 'momentum',
                        'title': '🚀 Strong Momentum Detected',
                        'description': f'Factors {", ".join(strong_momentum_factors)} show persistent directional movement.',
                        'severity': 'info'
                    })
                
                if mean_reverting_factors:
                    insights.append({
                        'type': 'momentum',
                        'title': '🔄 Mean Reversion Opportunity',
                        'description': f'Factors {", ".join(mean_reverting_factors)} may be due for reversal.',
                        'severity': 'warning'
                    })
            
            # Stability Insights
            if stability_analysis:
                unstable_factors = stability_analysis.get('unstable_factors', [])
                stable_factors = stability_analysis.get('stable_factors', [])
                
                if stable_factors:
                    insights.append({
                        'type': 'stability',
                        'title': '🛡️ Stable Factors Identified',
                        'description': f'Factors {", ".join(stable_factors)} show consistent behavior for reliable allocation.',
                        'severity': 'success'
                    })
                
                if unstable_factors:
                    insights.append({
                        'type': 'stability', 
                        'title': '⚠️ Volatile Factors Warning',
                        'description': f'Factors {", ".join(unstable_factors)} show high volatility - use with caution.',
                        'severity': 'danger'
                    })
            
            # Portfolio Recommendations
            insights.append({
                'type': 'recommendation',
                'title': '💡 Portfolio Optimization Suggestion',
                'description': 'Consider rebalancing based on factor momentum and stability analysis.',
                'severity': 'info'
            })
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append({
                'type': 'error',
                'title': '⚠️ Analysis Incomplete',
                'description': 'Some insights could not be generated due to data limitations.',
                'severity': 'warning'
            })
        
        return insights
    
    # Private helper methods
    def _normalize_scores(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """Normalize factor scores using z-score normalization"""
        return factor_scores.apply(lambda x: (x - x.mean()) / x.std())
    
    def _calculate_rolling_statistics(self, factor_scores: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate various rolling statistics for factor scores"""
        results = {}
        
        # Rolling means
        results['rolling_mean_short'] = factor_scores.rolling(self.window_short).mean()
        results['rolling_mean_long'] = factor_scores.rolling(self.window_long).mean()
        
        # Rolling volatilities
        results['rolling_vol_short'] = factor_scores.rolling(self.window_short).std()
        results['rolling_vol_long'] = factor_scores.rolling(self.window_long).std()
        
        # Rolling correlations between factors
        if len(factor_scores.columns) > 1:
            rolling_corr = {}
            for i, col1 in enumerate(factor_scores.columns):
                for j, col2 in enumerate(factor_scores.columns[i+1:], i+1):
                    corr_key = f"{col1}_{col2}"
                    rolling_corr[corr_key] = factor_scores[col1].rolling(
                        self.window_long
                    ).corr(factor_scores[col2])
            results['rolling_correlations'] = pd.DataFrame(rolling_corr)
        
        # Rolling Sharpe ratios (improved calculation)
        rolling_sharpe = {}
        for col in factor_scores.columns:
            rolling_mean = factor_scores[col].rolling(self.window_long).mean()
            rolling_std = factor_scores[col].rolling(self.window_long).std()
            
            # Use information ratio style for factor scores (no risk-free rate)
            # This is more appropriate for factor analysis
            rolling_sharpe[col] = np.where(
                rolling_std > 1e-6,
                rolling_mean / rolling_std * np.sqrt(252),  # Annualized information ratio
                0
            )
        results['rolling_sharpe'] = pd.DataFrame(rolling_sharpe)
        
        return results
    
    def _analyze_regimes(self, normalized_scores: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regimes based on factor behavior"""
        results = {}
        
        if len(normalized_scores) == 0:
            return results
        
        # Use first principal component for regime classification
        pc1 = normalized_scores.iloc[:, 0] if len(normalized_scores.columns) > 0 else None
        
        if pc1 is None:
            return results
        
        # Simple regime classification based on z-scores and trends
        regimes = []
        for i in range(len(pc1)):
            if i < self.window_short:
                regimes.append('neutral')
                continue
            
            current_value = pc1.iloc[i]
            recent_trend = pc1.iloc[i-self.window_short:i].mean()
            
            if current_value > 1 and recent_trend > 0:
                regimes.append('bull')
            elif current_value < -1 and recent_trend < 0:
                regimes.append('bear') 
            else:
                regimes.append('neutral')
        
        regime_series = pd.Series(regimes, index=normalized_scores.index)
        results['regime_classification'] = regime_series
        results['current_regime'] = regimes[-1] if regimes else 'neutral'
        
        # Calculate regime transition probabilities
        transition_matrix = self._calculate_transition_matrix(regimes)
        results['transition_matrix'] = transition_matrix
        
        # Regime stability (how long regimes last on average)
        regime_durations = self._calculate_regime_durations(regimes)
        results['regime_stability'] = np.mean(regime_durations) if regime_durations else 0
        
        return results
    
    def _analyze_momentum(self, factor_scores: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum and mean reversion in factor scores"""
        results = {}
        
        # Calculate momentum indicators
        momentum_signals = {}
        mean_reversion_signals = {}
        
        for col in factor_scores.columns:
            series = factor_scores[col]
            
            # Momentum: compare short vs long moving averages
            short_ma = series.rolling(self.window_short).mean()
            long_ma = series.rolling(self.window_long).mean()
            momentum_signals[col] = short_ma - long_ma
            
            # Mean reversion: z-score relative to long-term mean
            long_mean = series.rolling(self.window_long).mean()
            long_std = series.rolling(self.window_long).std()
            mean_reversion_signals[col] = (series - long_mean) / long_std
        
        results['momentum_signals'] = pd.DataFrame(momentum_signals)
        results['mean_reversion_signals'] = pd.DataFrame(mean_reversion_signals)
        
        # Identify strong momentum and mean reversion candidates
        latest_momentum = results['momentum_signals'].iloc[-1]
        latest_mean_reversion = results['mean_reversion_signals'].iloc[-1]
        
        results['strong_momentum_factors'] = latest_momentum[
            np.abs(latest_momentum) > 0.5
        ].index.tolist()
        
        results['mean_reverting_factors'] = latest_mean_reversion[
            np.abs(latest_mean_reversion) > 2
        ].index.tolist()
        
        return results
    
    def _analyze_stability(self, factor_scores: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stability and persistence of factors"""
        results = {}
        
        stability_metrics = {}
        for col in factor_scores.columns:
            series = factor_scores[col]
            
            # Volatility of volatility (vol of rolling std)
            rolling_vol = series.rolling(self.window_short).std()
            vol_of_vol = rolling_vol.std()
            
            # Autocorrelation (persistence)
            autocorr = series.autocorr(lag=1)
            
            # Coefficient of variation
            cv = series.std() / np.abs(series.mean()) if series.mean() != 0 else np.inf
            
            stability_metrics[col] = {
                'vol_of_vol': vol_of_vol,
                'autocorrelation': autocorr,
                'coefficient_of_variation': cv
            }
        
        results['stability_metrics'] = stability_metrics
        
        # Classify factors as stable or unstable
        stable_factors = []
        unstable_factors = []
        
        for factor, metrics in stability_metrics.items():
            # A factor is considered stable if it has low vol-of-vol and reasonable persistence
            if (metrics['vol_of_vol'] < 0.02 and 
                0.1 < metrics['autocorrelation'] < 0.9 and
                metrics['coefficient_of_variation'] < 2):
                stable_factors.append(factor)
            else:
                unstable_factors.append(factor)
        
        results['stable_factors'] = stable_factors
        results['unstable_factors'] = unstable_factors
        
        return results
    
    def _analyze_cross_factors(self, factor_scores: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between different factors"""
        results = {}
        
        if len(factor_scores.columns) < 2:
            return results
        
        # Rolling correlations between all factor pairs
        rolling_correlations = {}
        for i, col1 in enumerate(factor_scores.columns):
            for j, col2 in enumerate(factor_scores.columns[i+1:], i+1):
                pair_key = f"{col1}_{col2}"
                rolling_corr = factor_scores[col1].rolling(
                    self.window_long
                ).corr(factor_scores[col2])
                rolling_correlations[pair_key] = rolling_corr
        
        results['rolling_correlations'] = pd.DataFrame(rolling_correlations)
        
        # Factor diversification score (lower correlation = better diversification)
        correlation_matrix = factor_scores.corr()
        # Get upper triangle (excluding diagonal)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        avg_correlation = upper_triangle.stack().mean()
        results['diversification_score'] = 1 - abs(avg_correlation)
        
        return results
    
    def _generate_summary_statistics(self, factor_scores: pd.DataFrame,
                                   normalized_scores: pd.DataFrame,
                                   explained_variance: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        results = {}
        
        # Basic performance metrics for each factor
        performance_metrics = {}
        for i, col in enumerate(factor_scores.columns):
            series = factor_scores[col]
            norm_series = normalized_scores[col]
            
            # Calculate improved Sharpe ratio for factor analysis
            annual_return = series.mean() * 252
            annual_volatility = series.std() * np.sqrt(252)
            
            # Use appropriate risk-free rate for factor scores
            if abs(annual_return) < 0.1:  # Factor scores typically have small returns
                risk_free_rate = 0.001  # 0.1% - very conservative
            else:
                risk_free_rate = 0.01   # 1% - for larger returns
            
            # Calculate Sharpe ratio with proper risk-free rate handling
            if annual_volatility > 1e-6:
                if abs(annual_return) < 0.05:  # Very small returns - use information ratio style
                    factor_sharpe_ratio = annual_return / annual_volatility
                else:  # Normal Sharpe calculation
                    factor_sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                factor_sharpe_ratio = np.clip(factor_sharpe_ratio, -5, 5)
            else:
                factor_sharpe_ratio = 0
            
            metrics = {
                'total_return': series.sum(),
                'annualized_return': annual_return,
                'annualized_volatility': annual_volatility,
                'sharpe_ratio': factor_sharpe_ratio,
                'max_drawdown': (norm_series.cumsum() - norm_series.cumsum().expanding().max()).min(),
                'explained_variance': explained_variance[i] if i < len(explained_variance) else 0,
                'current_z_score': norm_series.iloc[-1] if len(norm_series) > 0 else 0,
                'extreme_events': len(norm_series[np.abs(norm_series) > 2])
            }
            
            performance_metrics[col] = metrics
        
        results['performance_metrics'] = performance_metrics
        
        # Identify best and worst performing factors
        if performance_metrics:
            sharpe_ratios = {k: v['sharpe_ratio'] for k, v in performance_metrics.items()}
            results['best_performing_factor'] = max(sharpe_ratios, key=sharpe_ratios.get)
            results['worst_performing_factor'] = min(sharpe_ratios, key=sharpe_ratios.get)
        
        return results
    
    def _analyze_correlation_stability(self, rolling_correlations: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze how stable correlations are over time"""
        results = {}
        
        for factor, correlations in rolling_correlations.items():
            factor_stability = {}
            for stock in correlations.columns:
                corr_series = correlations[stock].dropna()
                if len(corr_series) > 0:
                    # Stability measured as inverse of standard deviation
                    stability = 1 / (corr_series.std() + 0.01)  # Add small constant to avoid division by zero
                    factor_stability[stock] = stability
                else:
                    factor_stability[stock] = 0
            
            results[factor] = factor_stability
        
        return results
    
    def _analyze_loadings_drift(self, returns: pd.DataFrame, factor_scores: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how factor loadings change over time"""
        results = {}
        
        # Calculate rolling factor loadings using rolling PCA
        window = self.window_long
        rolling_loadings = {}
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            # Standardize returns
            scaler = StandardScaler()
            standardized_returns = pd.DataFrame(
                scaler.fit_transform(window_returns),
                columns=window_returns.columns,
                index=window_returns.index
            )
            
            # Fit PCA
            pca = PCA(n_components=min(len(factor_scores.columns), len(returns.columns)))
            pca.fit(standardized_returns)
            
            # Store loadings
            date = returns.index[i]
            for j, component in enumerate(factor_scores.columns):
                if j < pca.n_components_:
                    loadings_key = f"{component}_loadings"
                    if loadings_key not in rolling_loadings:
                        rolling_loadings[loadings_key] = {}
                    
                    for k, stock in enumerate(returns.columns):
                        if k < len(pca.components_[j]):
                            stock_loading_key = f"{stock}"
                            if stock_loading_key not in rolling_loadings[loadings_key]:
                                rolling_loadings[loadings_key][stock_loading_key] = []
                            rolling_loadings[loadings_key][stock_loading_key].append({
                                'date': date,
                                'loading': pca.components_[j][k]
                            })
        
        results['rolling_loadings'] = rolling_loadings
        return results
    
    def _analyze_lead_lag_relationships(self, factors: pd.DataFrame, 
                                      economic_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead-lag relationships between factors and economic indicators"""
        results = {}
        
        max_lag = 10  # Maximum lag to test
        
        for factor in factors.columns:
            factor_results = {}
            for econ_indicator in economic_data.columns:
                lag_correlations = {}
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        corr = factors[factor].corr(economic_data[econ_indicator])
                    elif lag > 0:
                        # Factor leads economic indicator
                        corr = factors[factor].shift(lag).corr(economic_data[econ_indicator])
                    else:
                        # Economic indicator leads factor
                        corr = factors[factor].corr(economic_data[econ_indicator].shift(-lag))
                    
                    lag_correlations[lag] = corr
                
                factor_results[econ_indicator] = lag_correlations
            
            results[factor] = factor_results
        
        return results
    
    def _classify_economic_regimes(self, factors: pd.DataFrame,
                                 economic_data: pd.DataFrame) -> Dict[str, Any]:
        """Classify economic regimes based on indicators"""
        results = {}
        
        # Simple economic regime classification
        # This is a placeholder - in practice, you'd use more sophisticated methods
        
        if 'VIX' in str(economic_data.columns) or 'volatility' in str(economic_data.columns).lower():
            # Use volatility-based regime classification
            vix_col = [col for col in economic_data.columns if 'vix' in col.lower()]
            if vix_col:
                vix_data = economic_data[vix_col[0]]
                
                regimes = []
                for vix_value in vix_data:
                    if pd.isna(vix_value):
                        regimes.append('neutral')
                    elif vix_value > 30:
                        regimes.append('high_volatility')
                    elif vix_value < 15:
                        regimes.append('low_volatility')
                    else:
                        regimes.append('normal_volatility')
                
                results['volatility_regimes'] = pd.Series(regimes, index=vix_data.index)
        
        return results
    
    def _calculate_transition_matrix(self, regimes: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probability matrix"""
        unique_regimes = list(set(regimes))
        n_regimes = len(unique_regimes)
        
        if n_regimes == 0:
            return {}
        
        # Initialize transition matrix
        transition_counts = {regime1: {regime2: 0 for regime2 in unique_regimes} 
                           for regime1 in unique_regimes}
        
        # Count transitions
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_counts[current_regime][next_regime] += 1
        
        # Convert counts to probabilities
        transition_probs = {}
        for regime1 in unique_regimes:
            total_transitions = sum(transition_counts[regime1].values())
            if total_transitions > 0:
                transition_probs[regime1] = {
                    regime2: count / total_transitions 
                    for regime2, count in transition_counts[regime1].items()
                }
            else:
                transition_probs[regime1] = {regime2: 0 for regime2 in unique_regimes}
        
        return transition_probs
    
    def _calculate_regime_durations(self, regimes: List[str]) -> List[int]:
        """Calculate how long each regime lasts"""
        if not regimes:
            return []
        
        durations = []
        current_regime = regimes[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regimes[i]
                current_duration = 1
        
        # Add the last regime duration
        durations.append(current_duration)
        
        return durations
    
    # Plotting helper methods
    def _plot_factor_scores_with_regimes(self, fig, normalized_scores, regime_analysis, 
                                       colors, row, col):
        """Plot enhanced factor scores with regime highlighting"""
        # Add factor score lines
        for i, factor_col in enumerate(normalized_scores.columns):
            fig.add_trace(
                go.Scatter(
                    x=normalized_scores.index,
                    y=normalized_scores[factor_col],
                    mode='lines',
                    name=f"{factor_col} (Z-score)",
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    hovertemplate=f"<b>{factor_col}</b><br>" +
                                f"Date: %{{x}}<br>" +
                                f"Z-Score: %{{y:.3f}}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # Add regime background colors if available
        if 'regime_classification' in regime_analysis:
            regime_series = regime_analysis['regime_classification']
            self._add_regime_backgrounds(fig, regime_series, row, col)
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     line_width=1, opacity=0.7, row=row, col=col)
        fig.add_hline(y=2, line_dash="dash", line_color="red", 
                     line_width=1, opacity=0.4, row=row, col=col)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", 
                     line_width=1, opacity=0.4, row=row, col=col)
    
    def _plot_momentum_analysis(self, fig, momentum_analysis, colors, row, col):
        """Plot momentum and mean reversion signals"""
        if 'momentum_signals' in momentum_analysis:
            momentum_signals = momentum_analysis['momentum_signals']
            for i, factor in enumerate(momentum_signals.columns):
                fig.add_trace(
                    go.Scatter(
                        x=momentum_signals.index,
                        y=momentum_signals[factor],
                        mode='lines',
                        name=f"{factor} Momentum",
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f"{factor} Momentum<br>Date: %{{x}}<br>Signal: %{{y:.3f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     opacity=0.5, row=row, col=col)
    
    def _plot_volatility_stability(self, fig, rolling_stats, stability_analysis, colors, row, col):
        """Plot rolling volatility and stability metrics"""
        if 'rolling_vol_short' in rolling_stats:
            vol_data = rolling_stats['rolling_vol_short']
            for i, factor in enumerate(vol_data.columns):
                fig.add_trace(
                    go.Scatter(
                        x=vol_data.index,
                        y=vol_data[factor],
                        mode='lines',
                        name=f"{factor} Volatility",
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f"{factor} Volatility<br>Date: %{{x}}<br>Vol: %{{y:.4f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
    
    def _plot_cross_factor_relationships(self, fig, cross_factor_analysis, colors, row, col):
        """Plot cross-factor correlation evolution"""
        if 'rolling_correlations' in cross_factor_analysis:
            rolling_corrs = cross_factor_analysis['rolling_correlations']
            for i, pair in enumerate(rolling_corrs.columns):
                fig.add_trace(
                    go.Scatter(
                        x=rolling_corrs.index,
                        y=rolling_corrs[pair],
                        mode='lines',
                        name=f"Corr: {pair}",
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f"Correlation {pair}<br>Date: %{{x}}<br>Corr: %{{y:.3f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=row, col=col)
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.3, row=row, col=col)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.3, row=row, col=col)
    
    def _plot_correlation_evolution(self, fig, returns, normalized_scores, colors, row, col):
        """Plot how stock-factor correlations evolve"""
        # Calculate rolling correlations for top stocks with first factor
        if len(normalized_scores.columns) > 0 and len(returns.columns) > 0:
            first_factor = normalized_scores.iloc[:, 0]
            
            for i, stock in enumerate(returns.columns[:4]):  # Limit to 4 stocks for clarity
                rolling_corr = returns[stock].rolling(
                    window=self.window_long, min_periods=self.window_short
                ).corr(first_factor)
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        mode='lines',
                        name=f"{stock} vs PC1",
                        line=dict(width=1.5, color=colors[i % len(colors)]),
                        hovertemplate=f"{stock} vs PC1<br>Date: %{{x}}<br>Corr: %{{y:.3f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=row, col=col)
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", opacity=0.3, row=row, col=col)
        fig.add_hline(y=-0.7, line_dash="dash", line_color="red", opacity=0.3, row=row, col=col)
    
    def _plot_economic_impact(self, fig, normalized_scores, economic_data, colors, row, col):
        """Enhanced economic indicators impact analysis"""
        if economic_data is not None and len(economic_data) > 0:
            # Get metadata if available
            metadata = getattr(economic_data, 'metadata', {})
            
            # Plot economic indicators with enhanced information
            legend_names_used = set()
            for i, col_name in enumerate(economic_data.columns[:4]):  # Show up to 4 indicators
                # Normalize economic data for comparison
                econ_normalized = (economic_data[col_name] - economic_data[col_name].mean()) / economic_data[col_name].std()
                
                # Clean up column name for display
                clean_name = col_name.split('_')[-1] if '_' in col_name else col_name
                display_name = clean_name.replace('^', '').replace('=X', '').replace('-Y.NYB', ' Index')
                
                # Get category for additional context
                category = next((cat for cat in metadata.keys() if cat in col_name), "Other")
                if category != "Other":
                    category_icon = category.split(' ')[0] if ' ' in category else "📊"
                    display_name = f"{category_icon} {display_name}"
                
                # Ensure unique legend names
                if display_name in legend_names_used:
                    display_name = f"{display_name} ({i+1})"
                legend_names_used.add(display_name)
                
                fig.add_trace(
                    go.Scatter(
                        x=economic_data.index,
                        y=econ_normalized,
                        mode='lines',
                        name=display_name,
                        line=dict(width=2.5, color=colors[i % len(colors)]),
                        hovertemplate=f"<b>{display_name}</b><br>Date: %{{x}}<br>Normalized Value: %{{y:.3f}}<br><extra></extra>",
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            # Add correlation information if we have factor scores
            if len(normalized_scores.columns) > 0:
                first_factor = normalized_scores.iloc[:, 0]
                
                # Calculate and display correlations
                correlations = []
                for col_name in economic_data.columns[:4]:
                    common_dates = first_factor.index.intersection(economic_data.index)
                    if len(common_dates) > 10:
                        corr = first_factor.loc[common_dates].corr(economic_data[col_name].loc[common_dates])
                        if not np.isnan(corr):
                            clean_name = col_name.split('_')[-1] if '_' in col_name else col_name
                            correlations.append(f"{clean_name}: {corr:.2f}")
                
                # Add correlation annotation with correct subplot reference
                if correlations:
                    corr_text = "PC1 Correlations:<br>" + "<br>".join(correlations[:3])
                    # Calculate correct subplot index for annotations
                    subplot_index = (row - 1) * 2 + col
                    fig.add_annotation(
                        x=0.02, y=0.98,
                        xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                        text=corr_text,
                        showarrow=False,
                        font=dict(size=10, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="lightblue",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="top"
                    )
            
            # Add impact explanation if metadata available
            if metadata:
                impacts = []
                # Extract meaningful impact descriptions
                for category, info in list(metadata.items())[:2]:
                    impact = info.get('impact', 'Economic indicator impact')
                    # Truncate long impact descriptions
                    if len(impact) > 60:
                        impact = impact[:57] + "..."
                    impacts.append(f"• {impact}")
                
                if impacts:
                    impact_text = "Market Impact:<br>" + "<br>".join(impacts)
                    subplot_index = (row - 1) * 2 + col
                    fig.add_annotation(
                        x=0.02, y=0.25,
                        xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                        text=impact_text,
                        showarrow=False,
                        font=dict(size=9, color="darkgreen"),
                        bgcolor="rgba(240,255,240,0.9)",
                        bordercolor="lightgreen",
                        borderwidth=1,
                        xanchor="left",
                        yanchor="bottom"
                    )
                    
        else:
            # Enhanced placeholder with helpful information
            subplot_index = (row - 1) * 2 + col
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                text="📊 Economic Indicators Analysis<br><br>" +
                     "💡 Select economic indicators from the dropdown<br>" +
                     "to see their impact on your portfolio factors.<br><br>" +
                     "Available categories:<br>" +
                     "• Treasury Yields • Commodities<br>" +
                     "• Currency Markets • Volatility<br>" +
                     "• Credit Markets • Real Estate",
                showarrow=False,
                font=dict(size=11, color="gray"),
                bgcolor="rgba(248,249,250,0.9)",
                bordercolor="lightgray",
                borderwidth=1
            )
    
    def _plot_extreme_events(self, fig, normalized_scores, regime_analysis, colors, row, col):
        """Enhanced extreme events and recovery analysis with event specification"""
        if len(normalized_scores.columns) > 0:
            first_factor = normalized_scores.iloc[:, 0]
            
            # Identify extreme events (|z-score| > 2) with classification
            extreme_highs = first_factor[first_factor > 2]
            extreme_lows = first_factor[first_factor < -2]
            severe_highs = first_factor[first_factor > 3]  # Very extreme events
            severe_lows = first_factor[first_factor < -3]
            
            # Plot the factor
            fig.add_trace(
                go.Scatter(
                    x=first_factor.index,
                    y=first_factor.values,
                    mode='lines',
                    name="PC1 Factor Movement",
                    line=dict(width=2, color=colors[0]),
                    hovertemplate="PC1<br>Date: %{x}<br>Z-Score: %{y:.3f}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Highlight extreme events with better descriptions
            if len(extreme_highs) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=extreme_highs.index,
                        y=extreme_highs.values,
                        mode='markers',
                        name="🔺 Market Stress Events",
                        marker=dict(color='red', size=8, symbol='triangle-up'),
                        hovertemplate="🔺 Market Stress Event<br>Date: %{x}<br>Z-Score: %{y:.3f}<br>Type: Portfolio divergence from norm<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            if len(extreme_lows) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=extreme_lows.index,
                        y=extreme_lows.values,
                        mode='markers',
                        name="🔻 Market Correction Events", 
                        marker=dict(color='blue', size=8, symbol='triangle-down'),
                        hovertemplate="🔻 Market Correction Event<br>Date: %{x}<br>Z-Score: %{y:.3f}<br>Type: Portfolio oversold condition<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            # Highlight severe events (z > 3 or z < -3)
            if len(severe_highs) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=severe_highs.index,
                        y=severe_highs.values,
                        mode='markers',
                        name="⚡ Crisis Events",
                        marker=dict(color='darkred', size=12, symbol='star'),
                        hovertemplate="⚡ CRISIS EVENT<br>Date: %{x}<br>Z-Score: %{y:.3f}<br>Type: Severe market dislocation<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            if len(severe_lows) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=severe_lows.index,
                        y=severe_lows.values,
                        mode='markers',
                        name="💥 Crash Events",
                        marker=dict(color='darkblue', size=12, symbol='star'),
                        hovertemplate="💥 CRASH EVENT<br>Date: %{x}<br>Z-Score: %{y:.3f}<br>Type: Severe market crash<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            # Add event classification annotations
            subplot_index = (row - 1) * 2 + col
            event_summary = []
            
            if len(extreme_highs) > 0 or len(extreme_lows) > 0:
                total_extreme = len(extreme_highs) + len(extreme_lows)
                event_summary.append(f"📊 Total Events: {total_extreme}")
                
                if len(extreme_highs) > 0:
                    event_summary.append(f"🔺 Stress Events: {len(extreme_highs)}")
                if len(extreme_lows) > 0:
                    event_summary.append(f"🔻 Correction Events: {len(extreme_lows)}")
                if len(severe_highs) > 0:
                    event_summary.append(f"⚡ Crisis Events: {len(severe_highs)}")
                if len(severe_lows) > 0:
                    event_summary.append(f"💥 Crash Events: {len(severe_lows)}")
                    
                # Calculate recovery metrics
                if total_extreme > 0:
                    # Simple recovery metric: average time to return to normal range (-1, 1)
                    recovery_info = self._analyze_recovery_times(first_factor, extreme_highs, extreme_lows)
                    if recovery_info:
                        event_summary.append(f"🔄 Avg Recovery: {recovery_info}")
                
                event_text = "<br>".join(event_summary)
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                    text=event_text,
                    showarrow=False,
                    font=dict(size=10, color="darkred"),
                    bgcolor="rgba(255,245,245,0.9)",
                    bordercolor="red",
                    borderwidth=1,
                    xanchor="left",
                    yanchor="top"
                )
                
                # Add event explanation
                explanation_text = ("Event Types:<br>"
                                  "🔺 Z>2: Market stress/volatility<br>"
                                  "🔻 Z<-2: Market corrections<br>"
                                  "⚡💥 |Z|>3: Crisis/crash events")
                fig.add_annotation(
                    x=0.02, y=0.02,
                    xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                    text=explanation_text,
                    showarrow=False,
                    font=dict(size=9, color="darkblue"),
                    bgcolor="rgba(245,245,255,0.9)",
                    bordercolor="blue",
                    borderwidth=1,
                    xanchor="left",
                    yanchor="bottom"
                )
            else:
                # No extreme events found
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{subplot_index} domain", yref=f"y{subplot_index} domain",
                    text="📊 No Extreme Events Detected<br><br>✅ Portfolio showing stable behavior<br>All factor movements within ±2σ range",
                    showarrow=False,
                    font=dict(size=12, color="green"),
                    bgcolor="rgba(245,255,245,0.9)",
                    bordercolor="green",
                    borderwidth=1
                )
        
        # Add extreme event threshold lines with labels
        fig.add_hline(y=3, line_dash="dot", line_color="darkred", opacity=0.7, row=row, col=col)
        fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5, row=row, col=col)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=row, col=col)
        fig.add_hline(y=-2, line_dash="dash", line_color="blue", opacity=0.5, row=row, col=col)
        fig.add_hline(y=-3, line_dash="dot", line_color="darkblue", opacity=0.7, row=row, col=col)
    
    def _analyze_recovery_times(self, factor_scores, extreme_highs, extreme_lows):
        """Analyze average recovery time from extreme events"""
        try:
            recovery_days = []
            
            # Combine all extreme events
            all_extremes = pd.concat([extreme_highs, extreme_lows]).sort_index()
            
            for event_date in all_extremes.index[:5]:  # Analyze first 5 events to avoid performance issues
                # Find when factor returns to normal range (-1, 1)
                future_scores = factor_scores.loc[factor_scores.index > event_date]
                if len(future_scores) > 0:
                    # Find first occurrence where |z-score| < 1
                    recovery_point = future_scores[abs(future_scores) < 1]
                    if len(recovery_point) > 0:
                        recovery_date = recovery_point.index[0]
                        days_to_recovery = (recovery_date - event_date).days
                        recovery_days.append(days_to_recovery)
            
            if recovery_days:
                avg_recovery = int(np.mean(recovery_days))
                return f"{avg_recovery} days"
            
        except Exception:
            pass
        
        return None
    
    def _plot_performance_attribution(self, fig, analysis_results, colors, row, col):
        """Plot factor performance attribution"""
        if 'summary_stats' in analysis_results and 'performance_metrics' in analysis_results['summary_stats']:
            performance_metrics = analysis_results['summary_stats']['performance_metrics']
            
            factors = list(performance_metrics.keys())
            sharpe_ratios = [performance_metrics[f]['sharpe_ratio'] for f in factors]
            returns = [performance_metrics[f]['annualized_return'] * 100 for f in factors]
            volatilities = [performance_metrics[f]['annualized_volatility'] * 100 for f in factors]
            
            # Create risk-return scatter
            fig.add_trace(
                go.Scatter(
                    x=volatilities,
                    y=returns,
                    mode='markers+text',
                    text=factors,
                    textposition="top center",
                    marker=dict(
                        size=[abs(sr) * 10 + 5 for sr in sharpe_ratios],  # Size based on Sharpe ratio
                        color=sharpe_ratios,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio", x=1.1, len=0.3)
                    ),
                    name="Risk-Return",
                    hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Add quadrant lines
            avg_vol = np.mean(volatilities) if volatilities else 0
            avg_ret = np.mean(returns) if returns else 0
            fig.add_vline(x=avg_vol, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
            fig.add_hline(y=avg_ret, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    def _add_regime_backgrounds(self, fig, regime_series, row, col):
        """Add colored backgrounds for different market regimes"""
        regime_colors = {
            'bull': 'rgba(0, 255, 0, 0.1)',
            'bear': 'rgba(255, 0, 0, 0.1)', 
            'neutral': 'rgba(128, 128, 128, 0.05)'
        }
        
        current_regime = None
        start_date = None
        
        for date, regime in regime_series.items():
            if regime != current_regime:
                # End previous regime background
                if current_regime is not None and start_date is not None:
                    fig.add_vrect(
                        x0=start_date, x1=date,
                        fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.05)'),
                        layer="below", line_width=0,
                        row=row, col=col
                    )
                
                # Start new regime background
                current_regime = regime
                start_date = date
        
        # Add final regime background
        if current_regime is not None and start_date is not None:
            fig.add_vrect(
                x0=start_date, x1=regime_series.index[-1],
                fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.05)'),
                layer="below", line_width=0,
                row=row, col=col
            )


def create_enhanced_timeseries_tab(results):
    """
    Main function to create the enhanced time series analysis tab
    This replaces the original create_timeseries_tab function
    
    Args:
        results: PCA analysis results dictionary
        
    Returns:
        HTML div containing the enhanced time series analysis
    """
    try:
        # Extract required data
        factor_scores = results['metrics']['factor_scores']
        explained_variance = results['metrics']['explained_variance']
        returns = results['returns']
        economic_data = results.get('economic_data')
        
        # Initialize the enhanced analyzer
        analyzer = EnhancedTimeSeriesAnalyzer(
            window_short=30,
            window_long=60,
            volatility_window=20
        )
        
        # Perform comprehensive analysis
        logger.info("Running enhanced time series analysis...")
        ts_analysis = analyzer.analyze_factor_scores(factor_scores, explained_variance)
        
        # Analyze stock correlations
        correlation_analysis = analyzer.analyze_stock_correlations(returns, factor_scores)
        ts_analysis.update(correlation_analysis)
        
        # Analyze economic relationships if data available
        if economic_data is not None and len(economic_data) > 0:
            econ_analysis = analyzer.analyze_economic_relationships(factor_scores, economic_data)
            ts_analysis.update(econ_analysis)
        
        # Create enhanced visualizations
        main_fig = analyzer.create_enhanced_visualizations(
            ts_analysis, returns, economic_data
        )
        
        # Generate actionable insights
        insights = analyzer.generate_insights_report(ts_analysis)
        
        # Create summary cards with enhanced metrics
        summary_cards = create_enhanced_summary_cards(ts_analysis)
        
        # Create insights cards
        insights_cards = create_insights_display_cards(insights)
        
        # Create interpretation guide
        interpretation_guide = create_enhanced_interpretation_guide()
        
        return html.Div([
            # Enhanced summary metrics
            summary_cards,
            
            # Main visualization
            dcc.Graph(figure=main_fig, className="mb-4"),
            
            # Insights and recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-lightbulb me-2"),
                                "🎯 Key Insights & Recommendations"
                            ], className="mb-0 text-primary")
                        ]),
                        dbc.CardBody(insights_cards)
                    ])
                ], width=8),
                dbc.Col([
                    interpretation_guide
                ], width=4)
            ], className="mb-4"),
            
            # Technical details accordion
            create_technical_details_accordion(ts_analysis),
            
            # Export functionality
            create_export_section()
        ])
        
    except Exception as e:
        logger.error(f"Error in enhanced time series analysis: {str(e)}")
        return dbc.Alert([
            html.H5("⚠️ Enhanced Time Series Analysis Error", className="alert-heading"),
            html.P(f"Error occurred during enhanced analysis: {str(e)}"),
            html.P("Please try running the analysis again or check your data inputs.")
        ], color="danger", className="mt-4")


def create_enhanced_summary_cards(analysis_results):
    """Create enhanced summary cards for time series analysis"""
    summary_stats = analysis_results.get('summary_stats', {})
    regime_analysis = analysis_results.get('regime_analysis', {})
    
    # Extract key metrics
    performance_metrics = summary_stats.get('performance_metrics', {})
    current_regime = regime_analysis.get('current_regime', 'Unknown')
    regime_stability = regime_analysis.get('regime_stability', 0)
    
    # Calculate average metrics across factors
    if performance_metrics:
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in performance_metrics.values()])
        avg_volatility = np.mean([m.get('annualized_volatility', 0) for m in performance_metrics.values()])
        total_extreme_events = sum([m.get('extreme_events', 0) for m in performance_metrics.values()])
        best_factor = summary_stats.get('best_performing_factor', 'N/A')
    else:
        avg_sharpe = 0
        avg_volatility = 0
        total_extreme_events = 0
        best_factor = 'N/A'
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{current_regime.title()}", className="text-primary mb-1"),
                    html.P("📊 Current Market Regime", className="text-muted mb-0 small"),
                    html.Small(f"Stability: {regime_stability:.1f} days", className="text-info")
                ])
            ], className="text-center border-primary shadow-sm h-100")
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{avg_sharpe:.3f}", className="text-success mb-1"),
                    html.P("⚡ Avg Factor Sharpe", className="text-muted mb-0 small"),
                    html.Small("Risk-adjusted performance", className="text-success")
                ])
            ], className="text-center border-success shadow-sm h-100")
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{avg_volatility*100:.1f}%", className="text-warning mb-1"),
                    html.P("📈 Average Volatility", className="text-muted mb-0 small"),
                    html.Small("Annualized factor vol", className="text-warning")
                ])
            ], className="text-center border-warning shadow-sm h-100")
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_extreme_events}", className="text-danger mb-1"),
                    html.P("🚨 Extreme Events", className="text-muted mb-0 small"),
                    html.Small("|Z-Score| > 2 events", className="text-danger")
                ])
            ], className="text-center border-danger shadow-sm h-100")
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{best_factor}", className="text-info mb-1"),
                    html.P("🏆 Top Factor", className="text-muted mb-0 small"),
                    html.Small("Best risk-adj returns", className="text-info")
                ])
            ], className="text-center border-info shadow-sm h-100")
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4([
                        html.I(className="fas fa-chart-line")
                    ], className="text-secondary mb-1"),
                    html.P("📊 Analysis Status", className="text-muted mb-0 small"),
                    html.Small("Enhanced Complete", className="text-success")
                ])
            ], className="text-center border-secondary shadow-sm h-100")
        ], width=2)
    ], className="mb-4")
    
    return cards


def create_insights_display_cards(insights):
    """Create display cards for insights and recommendations"""
    if not insights:
        return html.P("No specific insights available at this time.", className="text-muted")
    
    insight_cards = []
    
    # Group insights by type
    insight_groups = {}
    for insight in insights:
        insight_type = insight.get('type', 'general')
        if insight_type not in insight_groups:
            insight_groups[insight_type] = []
        insight_groups[insight_type].append(insight)
    
    # Create cards for each group
    for group_name, group_insights in insight_groups.items():
        group_items = []
        for insight in group_insights:
            severity_class = {
                'success': 'text-success',
                'warning': 'text-warning', 
                'danger': 'text-danger',
                'info': 'text-info'
            }.get(insight.get('severity', 'info'), 'text-info')
            
            group_items.append(
                html.Li([
                    html.Strong(insight.get('title', 'Insight'), className=severity_class),
                    html.Br(),
                    html.Span(insight.get('description', ''), className="small text-muted")
                ], className="mb-2")
            )
        
        if group_items:
            insight_cards.append(
                html.Div([
                    html.H6(f"{group_name.replace('_', ' ').title()}", className="text-primary mb-2"),
                    html.Ul(group_items, className="list-unstyled")
                ], className="mb-3")
            )
    
    return html.Div(insight_cards) if insight_cards else html.P("Processing insights...", className="text-muted")


def create_enhanced_interpretation_guide():
    """Create enhanced interpretation guide for time series analysis"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-book me-2"),
                "📚 Enhanced Analysis Guide"
            ], className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div([
                        html.P("🎯 Understanding Factor Evolution:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li("Z-scores show relative factor performance vs historical mean"),
                            html.Li("Values > 2 or < -2 indicate extreme events"),
                            html.Li("Colored backgrounds show market regime classification"),
                            html.Li("Moving averages reveal underlying trends")
                        ], className="small mb-3"),
                        
                        html.P("📊 Regime Classification:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li([html.Span("Bull: ", className="text-success fw-bold"), "Positive momentum with Z > 1"]),
                            html.Li([html.Span("Bear: ", className="text-danger fw-bold"), "Negative momentum with Z < -1"]),
                            html.Li([html.Span("Neutral: ", className="text-muted fw-bold"), "Mixed or low-conviction signals"])
                        ], className="small")
                    ])
                ], title="🎯 Factor Scores & Regimes"),
                
                dbc.AccordionItem([
                    html.Div([
                        html.P("⚡ Momentum vs Mean Reversion:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li("Momentum signals: Short MA - Long MA"),
                            html.Li("Positive values suggest continued direction"),
                            html.Li("Mean reversion: Z-score relative to long-term mean"),
                            html.Li("Extreme values may signal reversal opportunity")
                        ], className="small")
                    ])
                ], title="🔄 Momentum Analysis"),
                
                dbc.AccordionItem([
                    html.Div([
                        html.P("🌍 Economic Context:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li("Economic indicators provide macro context"),
                            html.Li("Lead-lag relationships show predictive power"),
                            html.Li("Correlation changes signal regime shifts"),
                            html.Li("Use for timing and risk management")
                        ], className="small")
                    ])
                ], title="🌍 Economic Integration"),
                
                dbc.AccordionItem([
                    html.Div([
                        html.P("💼 Portfolio Applications:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li("Stable factors: Core portfolio allocation"),
                            html.Li("Momentum factors: Tactical positioning"),
                            html.Li("Mean-reverting factors: Contrarian opportunities"),
                            html.Li("Extreme events: Risk management triggers")
                        ], className="small")
                    ])
                ], title="💼 Investment Applications")
            ], start_collapsed=True, flush=True)
        ])
    ])


def create_technical_details_accordion(analysis_results):
    """Create accordion with technical analysis details"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-cogs me-2"),
                "⚙️ Technical Analysis Details"
            ], className="mb-0 text-secondary")
        ]),
        dbc.CardBody([
            dbc.Accordion([
                dbc.AccordionItem([
                    create_regime_analysis_details(analysis_results.get('regime_analysis', {}))
                ], title="📊 Regime Analysis Details"),
                
                dbc.AccordionItem([
                    create_stability_analysis_details(analysis_results.get('stability_analysis', {}))
                ], title="🛡️ Stability Analysis"),
                
                dbc.AccordionItem([
                    create_momentum_analysis_details(analysis_results.get('momentum_analysis', {}))
                ], title="⚡ Momentum & Mean Reversion"),
                
                dbc.AccordionItem([
                    create_correlation_analysis_details(analysis_results.get('cross_factor_analysis', {}))
                ], title="🔗 Cross-Factor Relationships"),
                
                dbc.AccordionItem([
                    create_performance_analysis_details(analysis_results.get('summary_stats', {}))
                ], title="📈 Performance Metrics")
            ], start_collapsed=True, flush=True)
        ])
    ])


def create_regime_analysis_details(regime_analysis):
    """Create detailed regime analysis information"""
    if not regime_analysis:
        return html.P("No regime analysis data available.", className="text-muted")
    
    current_regime = regime_analysis.get('current_regime', 'Unknown')
    regime_stability = regime_analysis.get('regime_stability', 0)
    transition_matrix = regime_analysis.get('transition_matrix', {})
    
    details = [
        html.P([
            html.Strong("Current Regime: "), 
            html.Span(current_regime.title(), className=f"badge bg-{'success' if current_regime == 'bull' else 'danger' if current_regime == 'bear' else 'secondary'}")
        ]),
        html.P([html.Strong("Average Regime Duration: "), f"{regime_stability:.1f} days"]),
    ]
    
    if transition_matrix:
        details.append(html.Hr())
        details.append(html.H6("Regime Transition Probabilities:", className="text-primary"))
        
        for from_regime, transitions in transition_matrix.items():
            details.append(html.P([
                html.Strong(f"From {from_regime.title()}: "),
                html.Br(),
                *[html.Span(f"→ {to_regime.title()}: {prob:.1%}  ", className="small text-muted") 
                  for to_regime, prob in transitions.items()]
            ]))
    
    return html.Div(details)


def create_stability_analysis_details(stability_analysis):
    """Create detailed stability analysis information"""
    if not stability_analysis:
        return html.P("No stability analysis data available.", className="text-muted")
    
    stable_factors = stability_analysis.get('stable_factors', [])
    unstable_factors = stability_analysis.get('unstable_factors', [])
    stability_metrics = stability_analysis.get('stability_metrics', {})
    
    details = []
    
    if stable_factors:
        details.extend([
            html.P([
                html.Strong("Stable Factors: ", className="text-success"),
                ", ".join(stable_factors)
            ]),
            html.P("These factors show consistent behavior and are suitable for core allocation.", className="small text-muted")
        ])
    
    if unstable_factors:
        details.extend([
            html.P([
                html.Strong("Unstable Factors: ", className="text-warning"),
                ", ".join(unstable_factors) 
            ]),
            html.P("These factors show high volatility and require careful position sizing.", className="small text-muted")
        ])
    
    if stability_metrics:
        details.append(html.Hr())
        details.append(html.H6("Detailed Stability Metrics:", className="text-primary"))
        
        for factor, metrics in stability_metrics.items():
            details.append(
                html.Div([
                    html.Strong(f"{factor}:"),
                    html.Ul([
                        html.Li(f"Volatility of Volatility: {metrics.get('vol_of_vol', 0):.4f}"),
                        html.Li(f"Autocorrelation: {metrics.get('autocorrelation', 0):.3f}"),
                        html.Li(f"Coefficient of Variation: {metrics.get('coefficient_of_variation', 0):.2f}")
                    ], className="small")
                ], className="mb-2")
            )
    
    return html.Div(details)


def create_momentum_analysis_details(momentum_analysis):
    """Create detailed momentum analysis information"""
    if not momentum_analysis:
        return html.P("No momentum analysis data available.", className="text-muted")
    
    strong_momentum_factors = momentum_analysis.get('strong_momentum_factors', [])
    mean_reverting_factors = momentum_analysis.get('mean_reverting_factors', [])
    
    details = []
    
    if strong_momentum_factors:
        details.extend([
            html.P([
                html.Strong("Strong Momentum Factors: ", className="text-info"),
                ", ".join(strong_momentum_factors)
            ]),
            html.P("These factors show persistent directional movement. Consider trend-following strategies.", 
                   className="small text-muted")
        ])
    
    if mean_reverting_factors:
        details.extend([
            html.P([
                html.Strong("Mean Reverting Factors: ", className="text-warning"),
                ", ".join(mean_reverting_factors)
            ]),
            html.P("These factors are at extreme levels and may reverse. Consider contrarian positioning.", 
                   className="small text-muted")
        ])
    
    # Add interpretation guide
    details.extend([
        html.Hr(),
        html.H6("Momentum Interpretation:", className="text-primary"),
        html.Ul([
            html.Li("Momentum Signal > 0.5: Strong upward trend"),
            html.Li("Momentum Signal < -0.5: Strong downward trend"), 
            html.Li("Mean Reversion |Z-Score| > 2: Extreme level, potential reversal"),
            html.Li("Combine signals for tactical allocation decisions")
        ], className="small")
    ])
    
    return html.Div(details)


def create_correlation_analysis_details(cross_factor_analysis):
    """Create detailed cross-factor correlation analysis"""
    if not cross_factor_analysis:
        return html.P("No cross-factor analysis data available.", className="text-muted")
    
    diversification_score = cross_factor_analysis.get('diversification_score', 0)
    
    details = [
        html.P([
            html.Strong("Portfolio Diversification Score: "),
            html.Span(f"{diversification_score:.3f}", 
                     className=f"badge bg-{'success' if diversification_score > 0.7 else 'warning' if diversification_score > 0.4 else 'danger'}")
        ]),
        html.P("Score closer to 1.0 indicates better diversification across factors.", className="small text-muted"),
        html.Hr(),
        html.H6("Diversification Interpretation:", className="text-primary"),
        html.Ul([
            html.Li("Score > 0.7: Excellent diversification", className="text-success small"),
            html.Li("Score 0.4-0.7: Good diversification", className="text-warning small"),
            html.Li("Score < 0.4: Poor diversification, factors highly correlated", className="text-danger small")
        ])
    ]
    
    return html.Div(details)


def create_performance_analysis_details(summary_stats):
    """Create detailed performance analysis information"""
    if not summary_stats:
        return html.P("No performance analysis data available.", className="text-muted")
    
    performance_metrics = summary_stats.get('performance_metrics', {})
    best_factor = summary_stats.get('best_performing_factor', 'N/A')
    worst_factor = summary_stats.get('worst_performing_factor', 'N/A')
    
    details = []
    
    if best_factor != 'N/A' and worst_factor != 'N/A':
        details.extend([
            html.P([
                html.Strong("Best Performing Factor: ", className="text-success"),
                best_factor
            ]),
            html.P([
                html.Strong("Worst Performing Factor: ", className="text-danger"),
                worst_factor
            ]),
            html.Hr()
        ])
    
    if performance_metrics:
        details.append(html.H6("Detailed Performance Metrics:", className="text-primary"))
        
        for factor, metrics in performance_metrics.items():
            details.append(
                dbc.Card([
                    dbc.CardHeader(html.H6(factor, className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.P([html.Strong("Annualized Return: "), f"{metrics.get('annualized_return', 0)*100:.2f}%"]),
                                html.P([html.Strong("Volatility: "), f"{metrics.get('annualized_volatility', 0)*100:.2f}%"]),
                                html.P([html.Strong("Sharpe Ratio: "), f"{metrics.get('sharpe_ratio', 0):.3f}"])
                            ], width=6),
                            dbc.Col([
                                html.P([html.Strong("Max Drawdown: "), f"{metrics.get('max_drawdown', 0)*100:.2f}%"]),
                                html.P([html.Strong("Extreme Events: "), f"{metrics.get('extreme_events', 0)}"]),
                                html.P([html.Strong("Current Z-Score: "), f"{metrics.get('current_z_score', 0):.2f}"])
                            ], width=6)
                        ])
                    ])
                ], className="mb-2 small")
            )
    
    return html.Div(details)


def create_export_section():
    """Create export functionality section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-download me-2"),
                "📤 Export Analysis"
            ], className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            html.P("Export your enhanced time series analysis for further research or presentation:", 
                   className="text-muted"),
            dbc.ButtonGroup([
                dbc.Button([
                    html.I(className="fas fa-file-csv me-2"),
                    "Export Data (CSV)"
                ], color="primary", size="sm", id="export-csv-btn"),
                dbc.Button([
                    html.I(className="fas fa-file-pdf me-2"), 
                    "Export Report (PDF)"
                ], color="success", size="sm", id="export-pdf-btn"),
                dbc.Button([
                    html.I(className="fas fa-image me-2"),
                    "Export Charts (PNG)"
                ], color="info", size="sm", id="export-charts-btn")
            ])
        ])
    ], className="mt-4")


# Additional utility functions for advanced analysis

def calculate_factor_attribution(factor_scores, returns, window=60):
    """
    Calculate factor attribution for individual stocks over time
    
    Args:
        factor_scores: Factor scores DataFrame
        returns: Stock returns DataFrame  
        window: Rolling window for attribution analysis
        
    Returns:
        Dictionary with attribution results
    """
    attribution_results = {}
    
    for stock in returns.columns:
        stock_attribution = {}
        
        for factor in factor_scores.columns:
            # Rolling attribution using correlation * factor performance
            rolling_corr = returns[stock].rolling(window).corr(factor_scores[factor])
            factor_return = factor_scores[factor].rolling(window).mean()
            attribution = rolling_corr * factor_return
            
            stock_attribution[f"{factor}_attribution"] = attribution
        
        attribution_results[stock] = pd.DataFrame(stock_attribution)
    
    return attribution_results


def detect_structural_breaks(factor_scores, min_regime_length=30):
    """
    Detect structural breaks in factor behavior using change point detection
    
    Args:
        factor_scores: Factor scores DataFrame
        min_regime_length: Minimum length of a regime
        
    Returns:
        Dictionary with break points and regime information
    """
    if not RUPTURES_AVAILABLE:
        return {}
        
    break_results = {}
    
    for factor in factor_scores.columns:
        series = factor_scores[factor].dropna()
        
        if len(series) < min_regime_length * 2:
            continue
        
        try:
            # Use Binary Segmentation for change point detection
            model = "rbf"  # Radial Basis Function kernel
            algo = Binseg(model=model, min_size=min_regime_length, jump=5)
            algo.fit(series.values.reshape(-1, 1))
            
            # Detect break points
            break_points = algo.predict(n_bkps=3)  # Maximum 3 break points
            break_dates = [series.index[bp-1] for bp in break_points[:-1]]  # Exclude last point
            
            break_results[factor] = {
                'break_points': break_points,
                'break_dates': break_dates,
                'n_regimes': len(break_points)
            }
        except Exception as e:
            logger.warning(f"Error detecting structural breaks for {factor}: {str(e)}")
            continue
    
    return break_results


def calculate_factor_loadings_stability(returns, factor_scores, window=120):
    """
    Calculate how stable factor loadings are over time
    
    Args:
        returns: Stock returns DataFrame
        factor_scores: Factor scores DataFrame
        window: Rolling window for stability analysis
        
    Returns:
        DataFrame with stability metrics
    """
    stability_results = {}
    
    for stock in returns.columns:
        stock_stability = {}
        
        for factor in factor_scores.columns:
            # Calculate rolling correlation (proxy for loading)
            rolling_loading = returns[stock].rolling(window).corr(factor_scores[factor])
            
            # Stability metrics
            loading_std = rolling_loading.std()
            loading_mean = rolling_loading.mean()
            stability_ratio = abs(loading_mean) / (loading_std + 0.01)  # Higher is more stable
            
            stock_stability[f"{factor}_stability"] = stability_ratio
            stock_stability[f"{factor}_avg_loading"] = loading_mean
            stock_stability[f"{factor}_loading_vol"] = loading_std
        
        stability_results[stock] = stock_stability
    
    return pd.DataFrame(stability_results).T


# Integration function for the main dashboard
def integrate_enhanced_timeseries_to_main():
    """
    Instructions for integrating the enhanced time series analysis 
    into the main dashboard application
    """
    integration_notes = """
    INTEGRATION STEPS:
    
    1. Replace the existing create_timeseries_tab function in main.py with:
       from enhanced_timeseries import create_enhanced_timeseries_tab
    
    2. Update the tab callback in main.py:
       elif active_tab == "timeseries-tab":
           return create_enhanced_timeseries_tab(results)
    
    3. Add the enhanced_timeseries.py file to your project directory
    
    4. Install additional dependencies if needed:
       pip install ruptures  # For structural break detection (optional)
    
    5. The enhanced version provides:
       - Advanced regime detection and classification
       - Momentum and mean reversion analysis  
       - Cross-factor relationship analysis
       - Economic indicator integration
       - Detailed performance attribution
       - Interactive export functionality
       - Comprehensive interpretation guides
    
    6. All existing functionality is preserved while adding:
       - 8 detailed visualization panels
       - Advanced statistical analysis
       - Actionable investment insights
       - Professional-grade risk metrics
       - Enhanced user experience
    """
    
    return integration_notes


if __name__ == "__main__":
    # Example usage and testing
    print("Enhanced Time Series Analysis Module")
    print("====================================")
    print(integrate_enhanced_timeseries_to_main())
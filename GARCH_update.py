"""
VAR/GARCH Analysis Module for Advanced Stock Portfolio Analysis
================================================================
This module provides comprehensive econometric analysis using:
- Vector Autoregression (VAR) for multi-asset dynamics
- GARCH models for volatility clustering
- Granger causality testing
- Impulse response functions
- Forecast error variance decomposition

Author: Advanced Analytics Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from scipy.optimize import minimize
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging

# Import statsmodels components with graceful fallback
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available. VAR analysis will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VARAnalyzer:
    """
    Vector Autoregression (VAR) Analysis for Multi-Asset Dynamics
    
    This class implements VAR modeling to understand:
    - Lead-lag relationships between assets
    - Granger causality networks
    - Impulse response functions
    - Variance decomposition
    """
    
    def __init__(self, max_lags: int = 10):
        """
        Initialize VAR Analyzer
        
        Args:
            max_lags: Maximum number of lags to consider
        """
        self.max_lags = max_lags
        self.var_model = None
        self.var_results = None
        self.data_prepared = None
        
    def prepare_data(self, returns_data: pd.DataFrame, 
                     economic_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare and validate data for VAR analysis
        
        Args:
            returns_data: DataFrame of asset returns
            economic_data: Optional DataFrame of economic indicators
            
        Returns:
            Prepared DataFrame ready for VAR modeling
        """
        logger.info("Preparing data for VAR analysis...")
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available, returning original data")
            return returns_data.dropna()
        
        # Combine stock returns with economic indicators if available
        if economic_data is not None and len(economic_data) > 0:
            # Convert economic indicators to returns if needed
            econ_returns = economic_data.pct_change().dropna()
            
            # Align dates
            common_dates = returns_data.index.intersection(econ_returns.index)
            if len(common_dates) > 0:
                returns_aligned = returns_data.loc[common_dates]
                econ_aligned = econ_returns.loc[common_dates]
                combined_data = pd.concat([returns_aligned, econ_aligned], axis=1)
            else:
                combined_data = returns_data.copy()
        else:
            combined_data = returns_data.copy()
        
        # Remove any remaining NaN values
        combined_data = combined_data.dropna()
        
        # Test for stationarity
        stationarity_results = self._test_stationarity(combined_data)
        
        # Apply differencing if needed
        stationary_data = self._ensure_stationarity(combined_data, stationarity_results)
        
        self.data_prepared = stationary_data
        logger.info(f"Prepared {len(stationary_data)} observations with {len(stationary_data.columns)} variables")
        
        return stationary_data
    
    def _test_stationarity(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Test stationarity using Augmented Dickey-Fuller test"""
        results = {}
        
        if not STATSMODELS_AVAILABLE:
            for column in data.columns:
                results[column] = {'is_stationary': True}
            return results
        
        for column in data.columns:
            try:
                adf_result = adfuller(data[column].dropna(), autolag='AIC')
                results[column] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                logger.warning(f"Stationarity test failed for {column}: {e}")
                results[column] = {'is_stationary': True}
        
        return results
    
    def _ensure_stationarity(self, data: pd.DataFrame, stationarity_results: Dict) -> pd.DataFrame:
        """Ensure all series are stationary through differencing if needed"""
        stationary_data = data.copy()
        
        for column in data.columns:
            if not stationarity_results.get(column, {}).get('is_stationary', True):
                stationary_data[column] = data[column].diff().dropna()
                logger.info(f"Applied differencing to {column} to achieve stationarity")
        
        return stationary_data.dropna()
    
    def fit(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit VAR model with optimal lag selection
        
        Args:
            data: Optional data to fit (uses prepared data if not provided)
            
        Returns:
            Dictionary containing model results and diagnostics
        """
        if data is None:
            data = self.data_prepared
        
        if data is None or data.empty:
            raise ValueError("No data available for VAR fitting")
        
        logger.info("Estimating VAR model...")
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available, VAR analysis skipped")
            return {}
        
        try:
            # Additional data validation
            if data.shape[1] < 2:
                raise ValueError("VAR model requires at least 2 variables")
            
            if data.shape[0] < 20:
                raise ValueError("Insufficient observations for reliable VAR estimation (minimum 20 required)")
            
            # Check for sufficient variation
            for col in data.columns:
                if data[col].std() < 1e-8:
                    logger.warning(f"Variable {col} has very low variation, results may be unreliable")
            
            # Create VAR model
            self.var_model = VAR(data)
            
            # Select optimal lag length with better bounds
            max_lags_allowed = min(self.max_lags, len(data)//4, 10)  # More conservative
            
            try:
                lag_selection = self.var_model.select_order(maxlags=max_lags_allowed)
                optimal_lags = lag_selection.aic  # Use AIC criterion
                
                # Ensure lag is valid
                if optimal_lags <= 0:
                    optimal_lags = 1
                    logger.warning("Selected optimal lags was 0, using 1 instead")
                
                logger.info(f"Selected optimal lags: {optimal_lags}")
                
            except Exception as lag_e:
                logger.warning(f"Lag selection failed: {lag_e}, using default lag=1")
                optimal_lags = 1
            
            # Fit VAR model
            self.var_results = self.var_model.fit(optimal_lags)
            
            # Validate that fitting was successful
            if not hasattr(self.var_results, 'params') or self.var_results.params is None:
                raise ValueError("VAR fitting failed - no parameters estimated")
            
            # Calculate comprehensive results
            logger.info("Calculating VAR analysis components...")
            
            results = {
                'model': self.var_results,
                'optimal_lags': optimal_lags,
                'data_shape': data.shape,
                'variable_names': list(data.columns)
            }
            
            # Add diagnostics safely
            try:
                results['diagnostics'] = self._calculate_diagnostics()
            except Exception as diag_e:
                logger.error(f"Diagnostics calculation failed: {diag_e}")
                results['diagnostics'] = {}
            
            # Add Granger causality safely
            try:
                results['granger_causality'] = self._granger_causality_tests(data, optimal_lags)
            except Exception as granger_e:
                logger.error(f"Granger causality tests failed: {granger_e}")
                results['granger_causality'] = {}
            
            # Add FEVD safely
            try:
                results['forecast_error_variance_decomposition'] = self._calculate_fevd()
            except Exception as fevd_e:
                logger.error(f"FEVD calculation failed: {fevd_e}")
                results['forecast_error_variance_decomposition'] = {}
            
            # Add IRF safely
            try:
                results['impulse_response'] = self._calculate_impulse_response()
            except Exception as irf_e:
                logger.error(f"IRF calculation failed: {irf_e}")
                results['impulse_response'] = {}
            
            logger.info("VAR analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"VAR model estimation failed: {e}")
            return {}
    
    def _calculate_diagnostics(self) -> Dict[str, Any]:
        """Calculate VAR model diagnostic statistics"""
        if self.var_results is None or not STATSMODELS_AVAILABLE:
            return {}
        
        try:
            residuals = self.var_results.resid
            
            # Ljung-Box test for serial correlation
            ljung_box_results = {}
            for col in residuals.columns:
                try:
                    lb_test = acorr_ljungbox(residuals[col].dropna(), lags=10, return_df=True)
                    ljung_box_results[col] = {
                        'statistic': lb_test['lb_stat'].iloc[-1],
                        'p_value': lb_test['lb_pvalue'].iloc[-1]
                    }
                except Exception:
                    ljung_box_results[col] = {'statistic': np.nan, 'p_value': np.nan}
            
            return {
                'aic': self.var_results.aic,
                'bic': self.var_results.bic,
                'log_likelihood': self.var_results.llf,
                'ljung_box_tests': ljung_box_results,
                'residual_correlation': residuals.corr().to_dict()
            }
            
        except Exception as e:
            logger.error(f"VAR diagnostics calculation failed: {e}")
            return {}
    
    def _granger_causality_tests(self, data: pd.DataFrame, max_lag: int) -> Dict[str, Dict]:
        """Perform Granger causality tests between all variable pairs"""
        if not STATSMODELS_AVAILABLE:
            return {}
        
        granger_results = {}
        variables = data.columns.tolist()
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    try:
                        test_data = data[[var1, var2]].dropna()
                        
                        if len(test_data) > max_lag * 3:
                            gc_test = grangercausalitytests(test_data, max_lag, verbose=False)
                            
                            p_values = []
                            for lag in range(1, max_lag + 1):
                                if lag in gc_test:
                                    p_val = gc_test[lag][0]['ssr_ftest'][1]
                                    p_values.append(p_val)
                            
                            min_p_value = min(p_values) if p_values else 1.0
                            granger_results[f"{var2}_causes_{var1}"] = {
                                'min_p_value': min_p_value,
                                'is_significant': min_p_value < 0.05,
                                'p_values_by_lag': p_values
                            }
                    except Exception as e:
                        logger.warning(f"Granger causality test failed for {var2} -> {var1}: {e}")
                        continue
        
        return granger_results
    
    def _calculate_fevd(self, periods: int = 20) -> Dict[str, pd.DataFrame]:
        """Calculate Forecast Error Variance Decomposition"""
        if self.var_results is None or not STATSMODELS_AVAILABLE:
            return {}
        
        try:
            # Calculate FEVD
            fevd = self.var_results.fevd(periods)
            fevd_dict = {}
            
            # Handle different statsmodels versions
            if hasattr(fevd, 'decomp'):
                # Older statsmodels versions
                fevd_data = fevd.decomp
            elif hasattr(fevd, 'summary'):
                # Newer versions - extract from summary
                fevd_data = fevd.summary().as_html()
                logger.warning("FEVD summary format detected, using alternative extraction")
                return {}  # Return empty for now, implement HTML parsing if needed
            else:
                # Try direct access
                fevd_data = np.array(fevd)
            
            # Create DataFrames for each variable
            for i, var in enumerate(self.var_results.names):
                if fevd_data.ndim == 3:
                    # Standard format: (periods, variables, target_variable)
                    fevd_dict[var] = pd.DataFrame(
                        fevd_data[:, :, i],
                        columns=self.var_results.names,
                        index=range(1, periods + 1)
                    )
                elif fevd_data.ndim == 2:
                    # Flattened format
                    n_vars = len(self.var_results.names)
                    start_idx = i * periods
                    end_idx = (i + 1) * periods
                    var_data = fevd_data[start_idx:end_idx, :]
                    fevd_dict[var] = pd.DataFrame(
                        var_data,
                        columns=self.var_results.names,
                        index=range(1, periods + 1)
                    )
            
            logger.info(f"FEVD calculated successfully for {len(fevd_dict)} variables")
            return fevd_dict
            
        except AttributeError as e:
            logger.error(f"FEVD attribute error (possibly statsmodels version issue): {e}")
            # Try alternative calculation
            try:
                return self._calculate_fevd_alternative(periods)
            except:
                return {}
        except Exception as e:
            logger.error(f"FEVD calculation failed: {e}")
            return {}
    
    def _calculate_fevd_alternative(self, periods: int = 20) -> Dict[str, pd.DataFrame]:
        """Alternative FEVD calculation for compatibility"""
        try:
            # Use impulse response to manually calculate FEVD
            irf = self.var_results.irf(periods)
            
            if not hasattr(irf, 'irfs'):
                return {}
                
            irfs = irf.irfs
            n_vars = irfs.shape[2]
            fevd_dict = {}
            
            for i, var in enumerate(self.var_results.names):
                fevd_data = np.zeros((periods, n_vars))
                
                # Calculate cumulative variance contributions
                for period in range(periods):
                    # Sum of squared impulse responses up to this period
                    cumulative_responses = np.sum(irfs[:period+1, i, :] ** 2, axis=0)
                    total_variance = np.sum(cumulative_responses)
                    
                    if total_variance > 0:
                        fevd_data[period, :] = cumulative_responses / total_variance
                    else:
                        # If no variance, assume each variable contributes equally
                        fevd_data[period, :] = 1.0 / n_vars
                
                fevd_dict[var] = pd.DataFrame(
                    fevd_data,
                    columns=self.var_results.names,
                    index=range(1, periods + 1)
                )
            
            logger.info("Alternative FEVD calculation successful")
            return fevd_dict
            
        except Exception as e:
            logger.error(f"Alternative FEVD calculation failed: {e}")
            return {}
    
    def _calculate_impulse_response(self, periods: int = 20) -> Dict[str, np.ndarray]:
        """Calculate Impulse Response Functions"""
        if self.var_results is None or not STATSMODELS_AVAILABLE:
            return {}
        
        try:
            irf = self.var_results.irf(periods)
            
            if not hasattr(irf, 'irfs'):
                logger.error("IRF object missing 'irfs' attribute")
                return {}
            
            responses = irf.irfs
            result = {
                'responses': responses,
                'variable_names': self.var_results.names
            }
            
            # Try to get confidence intervals
            try:
                if hasattr(irf, 'stderr'):
                    # Method 1: Direct stderr method
                    stderr = irf.stderr()
                    result['lower_bounds'] = responses - 1.96 * stderr
                    result['upper_bounds'] = responses + 1.96 * stderr
                elif hasattr(irf, 'cov'):
                    # Method 2: Using covariance matrix
                    # Calculate standard errors from covariance
                    stderr = np.sqrt(np.diag(irf.cov()).reshape(responses.shape))
                    result['lower_bounds'] = responses - 1.96 * stderr
                    result['upper_bounds'] = responses + 1.96 * stderr
                else:
                    # Method 3: Bootstrap confidence intervals if available
                    if hasattr(irf, 'cum_effects_stderr'):
                        stderr = irf.cum_effects_stderr
                        result['lower_bounds'] = responses - 1.96 * stderr
                        result['upper_bounds'] = responses + 1.96 * stderr
                    else:
                        # Use simple approximation based on response magnitude
                        stderr_approx = np.abs(responses) * 0.1  # 10% of response as rough stderr
                        result['lower_bounds'] = responses - 1.96 * stderr_approx
                        result['upper_bounds'] = responses + 1.96 * stderr_approx
                        logger.warning("Using approximate confidence intervals for IRF")
                        
            except Exception as conf_e:
                logger.warning(f"Could not calculate IRF confidence intervals: {conf_e}")
                # Use simple approximation
                stderr_approx = np.abs(responses) * 0.1
                result['lower_bounds'] = responses - 1.96 * stderr_approx
                result['upper_bounds'] = responses + 1.96 * stderr_approx
            
            logger.info(f"IRF calculated successfully with shape: {responses.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Impulse response calculation failed: {e}")
            return self._calculate_impulse_response_alternative(periods)
    
    def _calculate_impulse_response_alternative(self, periods: int = 20) -> Dict[str, np.ndarray]:
        """Alternative IRF calculation using VAR coefficients"""
        try:
            if not hasattr(self.var_results, 'coefs'):
                return {}
            
            n_vars = len(self.var_results.names)
            lags = self.var_results.k_ar
            
            # Initialize response matrix
            responses = np.zeros((periods, n_vars, n_vars))
            
            # Identity matrix for initial shock
            responses[0] = np.eye(n_vars)
            
            # Calculate recursive responses
            coefs = self.var_results.coefs
            for t in range(1, periods):
                for lag in range(min(t, lags)):
                    if lag < coefs.shape[0]:
                        responses[t] += np.dot(responses[t-1-lag], coefs[lag].T)
            
            # Simple confidence bands (±10% of response)
            stderr_approx = np.abs(responses) * 0.1
            
            return {
                'responses': responses,
                'lower_bounds': responses - 1.96 * stderr_approx,
                'upper_bounds': responses + 1.96 * stderr_approx,
                'variable_names': self.var_results.names
            }
            
        except Exception as e:
            logger.error(f"Alternative IRF calculation failed: {e}")
            return {}


class GARCHAnalyzer:
    """
    GARCH Model for Volatility Clustering Analysis
    
    This class implements GARCH(1,1) modeling to understand:
    - Volatility persistence
    - Shock impact on volatility
    - Conditional heteroskedasticity
    """
    
    def __init__(self):
        """Initialize GARCH Analyzer"""
        self.results = {}
        
    def fit(self, returns: pd.Series, model_type: str = 'GARCH') -> Dict[str, Any]:
        """
        Fit GARCH model to returns series
        
        Args:
            returns: Series of asset returns
            model_type: Type of GARCH model (currently only 'GARCH' supported)
            
        Returns:
            Dictionary containing model parameters and diagnostics
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                logger.warning(f"Insufficient data for GARCH: {len(returns_clean)} observations")
                return {}
            
            # Calculate volatility statistics
            volatility_stats = self._calculate_volatility_statistics(returns_clean)
            
            # Estimate GARCH(1,1) parameters
            garch_params = self._estimate_garch_parameters(returns_clean)
            
            # Generate conditional volatility
            conditional_volatility = self._generate_conditional_volatility(returns_clean, garch_params)
            
            # Calculate model diagnostics
            diagnostics = self._calculate_diagnostics(returns_clean, conditional_volatility)
            
            return {
                'parameters': garch_params,
                'conditional_volatility': conditional_volatility,
                'volatility_statistics': volatility_stats,
                'diagnostics': diagnostics,
                'log_likelihood': garch_params.get('log_likelihood', np.nan)
            }
            
        except Exception as e:
            logger.error(f"GARCH estimation failed: {e}")
            return {}
    
    def _calculate_volatility_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive volatility statistics"""
        return {
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera_stat': stats.jarque_bera(returns)[0],
            'jarque_bera_p': stats.jarque_bera(returns)[1],
            'arch_lm_stat': self._arch_lm_test(returns),
            'volatility_clustering': self._test_volatility_clustering(returns)
        }
    
    def _arch_lm_test(self, returns: pd.Series, lags: int = 5) -> float:
        """ARCH LM test for heteroskedasticity"""
        try:
            squared_returns = returns**2
            n = len(squared_returns)
            
            # Create lagged variables
            X = np.ones((n-lags, lags+1))
            for i in range(lags):
                X[:, i+1] = squared_returns.iloc[lags-1-i:-1-i].values
            
            y = squared_returns.iloc[lags:].values
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                r_squared = 1 - np.var(residuals) / np.var(y)
                
                # LM statistic
                lm_stat = n * r_squared
                return lm_stat
            except:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _test_volatility_clustering(self, returns: pd.Series) -> float:
        """Test for volatility clustering using autocorrelation of squared returns"""
        try:
            squared_returns = returns**2
            autocorr_1 = squared_returns.autocorr(lag=1)
            return autocorr_1 if not np.isnan(autocorr_1) else 0.0
        except:
            return 0.0
    
    def _estimate_garch_parameters(self, returns: pd.Series) -> Dict[str, float]:
        """
        Estimate GARCH(1,1) parameters using Maximum Likelihood Estimation
        
        GARCH(1,1) model:
        σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
        
        where:
        - ω (omega): baseline volatility
        - α (alpha): reaction to shocks
        - β (beta): persistence
        """
        
        def garch_likelihood(params, returns):
            """Negative log-likelihood for GARCH(1,1)"""
            omega, alpha, beta = params
            
            # Parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            T = len(returns)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(returns)  # Initial variance
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
            
            # Avoid numerical issues
            sigma2 = np.maximum(sigma2, 1e-8)
            
            # Log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            return -log_likelihood
        
        try:
            # Initial parameter values
            initial_params = [0.01, 0.1, 0.85]  # omega, alpha, beta
            
            # Optimization bounds
            bounds = [(1e-6, 1), (0, 1), (0, 1)]
            
            # Constraint: alpha + beta < 1 (stationarity)
            constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}
            
            result = minimize(
                garch_likelihood,
                initial_params,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                omega, alpha, beta = result.x
                return {
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta,
                    'persistence': alpha + beta,
                    'log_likelihood': -result.fun,
                    'converged': True
                }
            else:
                logger.warning("GARCH optimization did not converge")
                return self._fallback_parameters()
                
        except Exception as e:
            logger.error(f"GARCH parameter estimation failed: {e}")
            return self._fallback_parameters()
    
    def _fallback_parameters(self) -> Dict[str, float]:
        """Return fallback parameters when optimization fails"""
        return {
            'omega': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'persistence': np.nan,
            'log_likelihood': np.nan,
            'converged': False
        }
    
    def _generate_conditional_volatility(self, returns: pd.Series, 
                                        garch_params: Dict) -> pd.Series:
        """Generate conditional volatility series from GARCH parameters"""
        try:
            if not garch_params.get('converged', False):
                # Fallback to rolling volatility
                return returns.rolling(20).std()
            
            omega = garch_params['omega']
            alpha = garch_params['alpha']
            beta = garch_params['beta']
            
            T = len(returns)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(returns)
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
            
            return pd.Series(np.sqrt(sigma2), index=returns.index, name='Conditional_Volatility')
            
        except Exception as e:
            logger.error(f"Conditional volatility generation failed: {e}")
            return returns.rolling(20).std()
    
    def _calculate_diagnostics(self, returns: pd.Series, 
                              conditional_volatility: pd.Series) -> Dict[str, float]:
        """Calculate GARCH model diagnostics"""
        try:
            # Standardized residuals
            std_residuals = returns / conditional_volatility
            
            # Ljung-Box tests
            lb_stat, lb_p_value = self._ljung_box_test(std_residuals)
            lb_stat_sq, lb_p_value_sq = self._ljung_box_test(std_residuals**2)
            
            return {
                'ljung_box_stat': lb_stat,
                'ljung_box_p_value': lb_p_value,
                'ljung_box_squared_stat': lb_stat_sq,
                'ljung_box_squared_p_value': lb_p_value_sq,
                'mean_std_residual': std_residuals.mean(),
                'std_std_residual': std_residuals.std(),
                'skewness_std_residual': std_residuals.skew(),
                'kurtosis_std_residual': std_residuals.kurtosis()
            }
            
        except Exception as e:
            logger.error(f"GARCH diagnostics calculation failed: {e}")
            return {}
    
    def _ljung_box_test(self, series: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """Perform Ljung-Box test for serial correlation"""
        try:
            if STATSMODELS_AVAILABLE:
                lb_test = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
                return lb_test['lb_stat'].iloc[-1], lb_test['lb_pvalue'].iloc[-1]
            else:
                return np.nan, np.nan
        except:
            return np.nan, np.nan


class IntegratedVARGARCHAnalysis:
    """
    Integrated VAR-GARCH Analysis System
    
    This class combines VAR and GARCH models for comprehensive
    portfolio analysis including both multi-asset dynamics and
    individual volatility modeling.
    """
    
    def __init__(self, max_lags: int = 10):
        """
        Initialize integrated analysis system
        
        Args:
            max_lags: Maximum lags for VAR model
        """
        self.var_analyzer = VARAnalyzer(max_lags)
        self.garch_analyzer = GARCHAnalyzer()
        self.results = {}
        
    def analyze_portfolio(self, 
                          returns_data: pd.DataFrame,
                          economic_data: Optional[pd.DataFrame] = None,
                          analyze_var: bool = True,
                          analyze_garch: bool = True,
                          max_garch_stocks: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive VAR-GARCH analysis on portfolio
        
        Args:
            returns_data: DataFrame of asset returns
            economic_data: Optional economic indicators
            analyze_var: Whether to perform VAR analysis
            analyze_garch: Whether to perform GARCH analysis
            max_garch_stocks: Maximum number of stocks for GARCH analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {
            'var_results': {},
            'garch_results': {},
            'summary_statistics': {},
            'recommendations': []
        }
        
        # VAR Analysis
        if analyze_var and len(returns_data) > 50 and len(returns_data.columns) >= 2:
            logger.info("Starting VAR analysis...")
            try:
                # Prepare data
                prepared_data = self.var_analyzer.prepare_data(returns_data, economic_data)
                
                # Fit VAR model
                var_results = self.var_analyzer.fit(prepared_data)
                results['var_results'] = var_results
                
                # Generate VAR recommendations
                results['recommendations'].extend(
                    self._generate_var_recommendations(var_results)
                )
            except Exception as e:
                logger.error(f"VAR analysis failed: {e}")
        
        # GARCH Analysis
        if analyze_garch:
            logger.info("Starting GARCH analysis...")
            garch_results = {}
            
            for i, stock in enumerate(returns_data.columns):
                if i >= max_garch_stocks:
                    break
                    
                try:
                    logger.info(f"Fitting GARCH for {stock}...")
                    garch_result = self.garch_analyzer.fit(returns_data[stock])
                    
                    if garch_result:
                        garch_results[stock] = garch_result
                        
                except Exception as e:
                    logger.warning(f"GARCH failed for {stock}: {e}")
            
            results['garch_results'] = garch_results
            
            # Generate GARCH recommendations
            results['recommendations'].extend(
                self._generate_garch_recommendations(garch_results)
            )
        
        # Calculate summary statistics
        results['summary_statistics'] = self._calculate_summary_statistics(
            returns_data, results['var_results'], results['garch_results']
        )
        
        self.results = results
        return results
    
    def _generate_var_recommendations(self, var_results: Dict) -> List[str]:
        """Generate investment recommendations from VAR results"""
        recommendations = []
        
        if 'granger_causality' in var_results:
            significant_relationships = [
                rel for rel, data in var_results['granger_causality'].items()
                if data.get('is_significant', False)
            ]
            
            if significant_relationships:
                recommendations.append(
                    f"📊 Found {len(significant_relationships)} significant lead-lag relationships. "
                    "Consider using leading indicators for timing decisions."
                )
        
        if 'diagnostics' in var_results:
            aic = var_results['diagnostics'].get('aic')
            if aic and aic < -1000:
                recommendations.append(
                    "✅ VAR model shows good fit. Multi-asset dynamics are well captured."
                )
        
        return recommendations
    
    def _generate_garch_recommendations(self, garch_results: Dict) -> List[str]:
        """Generate investment recommendations from GARCH results"""
        recommendations = []
        
        high_persistence_stocks = []
        for stock, result in garch_results.items():
            if 'parameters' in result and result['parameters'].get('converged'):
                persistence = result['parameters'].get('persistence', 0)
                if persistence > 0.95:
                    high_persistence_stocks.append(stock)
        
        if high_persistence_stocks:
            recommendations.append(
                f"⚠️ High volatility persistence in: {', '.join(high_persistence_stocks)}. "
                "Consider reducing position sizes during volatile periods."
            )
        
        return recommendations
    
    def _calculate_summary_statistics(self, returns_data: pd.DataFrame,
                                     var_results: Dict,
                                     garch_results: Dict) -> Dict:
        """Calculate summary statistics for the analysis"""
        summary = {
            'n_assets': len(returns_data.columns),
            'n_observations': len(returns_data),
            'date_range': f"{returns_data.index[0]} to {returns_data.index[-1]}",
            'var_optimal_lags': var_results.get('optimal_lags', 'N/A'),
            'n_significant_causalities': 0,
            'avg_garch_persistence': 0
        }
        
        # Count significant Granger causalities
        if 'granger_causality' in var_results:
            summary['n_significant_causalities'] = sum(
                1 for data in var_results['granger_causality'].values()
                if data.get('is_significant', False)
            )
        
        # Calculate average GARCH persistence
        persistences = []
        for result in garch_results.values():
            if 'parameters' in result and result['parameters'].get('converged'):
                persistences.append(result['parameters'].get('persistence', 0))
        
        if persistences:
            summary['avg_garch_persistence'] = np.mean(persistences)
        
        return summary


# Utility functions for easy integration
def run_var_garch_analysis(returns_data: pd.DataFrame,
                           economic_data: Optional[pd.DataFrame] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run complete VAR-GARCH analysis
    
    Args:
        returns_data: DataFrame of asset returns
        economic_data: Optional economic indicators
        **kwargs: Additional parameters for the analysis
        
    Returns:
        Complete analysis results
    """
    analyzer = IntegratedVARGARCHAnalysis(
        max_lags=kwargs.get('max_lags', 10)
    )
    
    return analyzer.analyze_portfolio(
        returns_data,
        economic_data,
        analyze_var=kwargs.get('analyze_var', True),
        analyze_garch=kwargs.get('analyze_garch', True),
        max_garch_stocks=kwargs.get('max_garch_stocks', 5)
    )


# Export main classes and functions
__all__ = [
    'VARAnalyzer',
    'GARCHAnalyzer', 
    'IntegratedVARGARCHAnalysis',
    'run_var_garch_analysis',
    'STATSMODELS_AVAILABLE'
]


logger = logging.getLogger(__name__)


class VARGARCHVisualizer:
    """
    Create professional visualizations for VAR/GARCH analysis results
    """
    
    def __init__(self):
        """Initialize visualizer with default settings"""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
        self.fig = None
    
    def create_comprehensive_dashboard(self, var_results: Dict, 
                                      garch_results: Dict,
                                      returns_data: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive 4-panel dashboard for VAR/GARCH results
        
        Args:
            var_results: VAR analysis results
            garch_results: GARCH analysis results
            returns_data: Original returns data
            
        Returns:
            Plotly figure with 4 subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Granger Causality Network",
                "Impulse Response Functions",
                "GARCH Conditional Volatility",
                "Forecast Error Variance Decomposition"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Panel 1: Granger Causality Network
        self._add_granger_network(fig, var_results, row=1, col=1)
        
        # Panel 2: Impulse Response Functions
        self._add_impulse_response(fig, var_results, row=1, col=2)
        
        # Panel 3: GARCH Volatility
        self._add_garch_volatility(fig, garch_results, row=2, col=1)
        
        # Panel 4: Variance Decomposition
        self._add_variance_decomposition(fig, var_results, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="VAR/GARCH Analysis Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        self.fig = fig
        return fig
    
    def _add_granger_network(self, fig: go.Figure, var_results: Dict, 
                            row: int, col: int):
        """Add Granger causality network visualization"""
        if 'granger_causality' not in var_results:
            return
        
        granger_data = var_results['granger_causality']
        
        # Extract significant relationships
        significant_relationships = []
        for relationship, result in granger_data.items():
            if result.get('is_significant', False):
                parts = relationship.split('_causes_')
                if len(parts) == 2:
                    significant_relationships.append({
                        'from': parts[0],
                        'to': parts[1],
                        'strength': 1 - result['min_p_value']
                    })
        
        if not significant_relationships:
            # Add empty plot with message
            fig.add_trace(
                go.Scatter(x=[0.5], y=[0.5], 
                          text=["No significant causality found"],
                          mode='text', showlegend=False),
                row=row, col=col
            )
            return
        
        # Create network layout
        variables = list(set([r['from'] for r in significant_relationships] + 
                           [r['to'] for r in significant_relationships]))
        n_vars = len(variables)
        
        # Position nodes in a circle
        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Add edges (relationships)
        for rel in significant_relationships:
            from_idx = variables.index(rel['from'])
            to_idx = variables.index(rel['to'])
            
            # Add arrow
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[from_idx], x_pos[to_idx]],
                    y=[y_pos[from_idx], y_pos[to_idx]],
                    mode='lines',
                    line=dict(
                        width=rel['strength']*5,
                        color=self.color_scheme['danger']
                    ),
                    hovertemplate=f"{rel['from']} → {rel['to']}<br>Strength: {rel['strength']:.3f}",
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=x_pos, y=y_pos,
                mode='markers+text',
                text=variables,
                textposition="top center",
                marker=dict(
                    size=20,
                    color=self.color_scheme['primary'],
                    line=dict(width=2, color='white')
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(showgrid=False, showticklabels=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, showticklabels=False, row=row, col=col)
    
    def _add_impulse_response(self, fig: go.Figure, var_results: Dict,
                             row: int, col: int):
        """Add impulse response function visualization"""
        try:
            if 'impulse_response' not in var_results:
                self._add_placeholder_chart(fig, "Impulse Response Functions", 
                                          "IRF data not available", row, col)
                return
            
            irf_data = var_results['impulse_response']
            if not irf_data or 'responses' not in irf_data:
                self._add_placeholder_chart(fig, "Impulse Response Functions", 
                                          "IRF calculation failed", row, col)
                return
            
            responses = irf_data['responses']
            var_names = irf_data.get('variable_names', [])
            
            # Validate response data shape
            if responses.size == 0 or len(responses.shape) != 3:
                self._add_placeholder_chart(fig, "Impulse Response Functions", 
                                          "Invalid IRF data format", row, col)
                return
            
            # Plot first variable's response to shocks
            if len(var_names) > 0 and responses.shape[0] > 0:
                periods = list(range(responses.shape[0]))
                
                # Show response of first variable to all shocks
                colors = px.colors.qualitative.Set2
                n_shocks = min(3, responses.shape[2])  # Limit to 3 shocks for clarity
                
                for j in range(n_shocks):
                    var_name = var_names[j] if j < len(var_names) else f'Var{j+1}'
                    
                    # Get response data
                    response_data = responses[:, 0, j]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=periods,
                            y=response_data,
                            mode='lines+markers',
                            name=f"Shock: {var_name}",
                            line=dict(width=2, color=colors[j % len(colors)]),
                            marker=dict(size=4),
                            hovertemplate=f"Shock from {var_name}<br>Period: %{{x}}<br>Response: %{{y:.4f}}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                    
                    # Add confidence bands if available
                    if 'lower_bounds' in irf_data and 'upper_bounds' in irf_data:
                        lower_bounds = irf_data['lower_bounds']
                        upper_bounds = irf_data['upper_bounds']
                        
                        if lower_bounds.shape == responses.shape and upper_bounds.shape == responses.shape:
                            # Add confidence interval
                            fig.add_trace(
                                go.Scatter(
                                    x=periods + periods[::-1],
                                    y=np.concatenate([upper_bounds[:, 0, j], lower_bounds[::-1, 0, j]]),
                                    fill='toself',
                                    fillcolor=f'rgba({colors[j % len(colors)][4:-1]}, 0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ),
                                row=row, col=col
                            )
                
                # Add zero line for reference
                fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                            opacity=0.5, row=row, col=col)
            else:
                self._add_placeholder_chart(fig, "Impulse Response Functions", 
                                          "No valid IRF data to display", row, col)
                return
            
            # Update axes
            fig.update_xaxes(title_text="Periods", showgrid=True, row=row, col=col)
            fig.update_yaxes(title_text="Response", showgrid=True, row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding impulse response visualization: {e}")
            self._add_placeholder_chart(fig, "Impulse Response Functions", 
                                      f"Visualization error: {str(e)}", row, col)
    
    def _add_garch_volatility(self, fig: go.Figure, garch_results: Dict,
                             row: int, col: int):
        """Add GARCH conditional volatility visualization"""
        if not garch_results:
            return
        
        # Plot conditional volatility for up to 3 stocks
        colors = [self.color_scheme['primary'], 
                 self.color_scheme['secondary'], 
                 self.color_scheme['success']]
        
        for i, (stock, result) in enumerate(list(garch_results.items())[:3]):
            if 'conditional_volatility' not in result:
                continue
            
            cond_vol = result['conditional_volatility']
            if isinstance(cond_vol, pd.Series) and not cond_vol.empty:
                # Show recent volatility (last 6 months)
                recent_vol = cond_vol.tail(126)
                
                fig.add_trace(
                    go.Scatter(
                        x=recent_vol.index,
                        y=recent_vol.values * 100,  # Convert to percentage
                        mode='lines',
                        name=f"{stock}",
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f"{stock}<br>Date: %{{x}}<br>Volatility: %{{y:.2f}}%<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Conditional Volatility (%)", row=row, col=col)
    
    def _add_variance_decomposition(self, fig: go.Figure, var_results: Dict,
                                   row: int, col: int):
        """Add forecast error variance decomposition visualization"""
        try:
            if 'forecast_error_variance_decomposition' not in var_results:
                self._add_placeholder_chart(fig, "Forecast Error Variance Decomposition", 
                                          "FEVD data not available", row, col)
                return
            
            fevd_data = var_results['forecast_error_variance_decomposition']
            if not fevd_data:
                self._add_placeholder_chart(fig, "Forecast Error Variance Decomposition", 
                                          "FEVD calculation failed", row, col)
                return
            
            # Select first variable for display
            if not isinstance(fevd_data, dict) or len(fevd_data) == 0:
                self._add_placeholder_chart(fig, "Forecast Error Variance Decomposition", 
                                          "Invalid FEVD data format", row, col)
                return
            
            first_var = list(fevd_data.keys())[0]
            fevd_df = fevd_data[first_var]
            
            # Validate DataFrame
            if fevd_df.empty or fevd_df.shape[0] == 0:
                self._add_placeholder_chart(fig, "Forecast Error Variance Decomposition", 
                                          "No FEVD data to display", row, col)
                return
            
            # Create stacked area chart
            colors = px.colors.qualitative.Set3
            
            # Ensure data sums to 1 (or 100%)
            row_sums = fevd_df.sum(axis=1)
            if (row_sums < 0.9).any() or (row_sums > 1.1).any():
                logger.warning("FEVD data may not be properly normalized")
            
            for i, column in enumerate(fevd_df.columns):
                contribution_data = fevd_df[column] * 100  # Convert to percentage
                
                fig.add_trace(
                    go.Scatter(
                        x=fevd_df.index,
                        y=contribution_data,
                        mode='lines',
                        name=column,
                        stackgroup='one',
                        fillcolor=colors[i % len(colors)],
                        line=dict(width=0.5, color=colors[i % len(colors)]),
                        hovertemplate=f"{column}<br>Period: %{{x}}<br>Contribution: %{{y:.1f}}%<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            # Add 100% reference line
            fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                        opacity=0.5, row=row, col=col)
            
            # Update axes
            fig.update_xaxes(title_text="Forecast Horizon", showgrid=True, row=row, col=col)
            fig.update_yaxes(title_text="Variance Contribution (%)", 
                           range=[0, 105], showgrid=True, row=row, col=col)
            
        except Exception as e:
            logger.error(f"Error adding variance decomposition visualization: {e}")
            self._add_placeholder_chart(fig, "Forecast Error Variance Decomposition", 
                                      f"Visualization error: {str(e)}", row, col)
    
    def _add_placeholder_chart(self, fig: go.Figure, title: str, message: str, 
                             row: int, col: int):
        """Add placeholder chart when data is not available"""
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0.5, 0.5],
                mode='text',
                text=[message],
                textposition="middle center",
                textfont=dict(size=14, color="gray"),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="", showgrid=False, showticklabels=False, 
                        zeroline=False, row=row, col=col)
        fig.update_yaxes(title_text="", showgrid=False, showticklabels=False, 
                        zeroline=False, row=row, col=col)
    
    def create_volatility_comparison(self, garch_results: Dict,
                                    returns_data: pd.DataFrame) -> go.Figure:
        """
        Create detailed volatility comparison chart
        
        Args:
            garch_results: GARCH analysis results
            returns_data: Original returns data
            
        Returns:
            Plotly figure comparing different volatility measures
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Conditional vs Realized Volatility", 
                          "Volatility Persistence Parameters"],
            row_heights=[0.7, 0.3]
        )
        
        # Top panel: Volatility comparison
        for stock, result in list(garch_results.items())[:5]:
            if 'conditional_volatility' in result:
                cond_vol = result['conditional_volatility']
                
                # Add conditional volatility
                fig.add_trace(
                    go.Scatter(
                        x=cond_vol.index,
                        y=cond_vol.values * 100 * np.sqrt(252),  # Annualized
                        mode='lines',
                        name=f"{stock} GARCH",
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                # Add realized volatility
                if stock in returns_data.columns:
                    realized_vol = returns_data[stock].rolling(20).std() * 100 * np.sqrt(252)
                    fig.add_trace(
                        go.Scatter(
                            x=realized_vol.index,
                            y=realized_vol.values,
                            mode='lines',
                            name=f"{stock} Realized",
                            line=dict(width=1, dash='dot'),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        # Bottom panel: GARCH parameters
        param_data = []
        for stock, result in garch_results.items():
            if 'parameters' in result and result['parameters'].get('converged'):
                params = result['parameters']
                param_data.append({
                    'Stock': stock,
                    'Alpha': params.get('alpha', 0),
                    'Beta': params.get('beta', 0),
                    'Persistence': params.get('persistence', 0)
                })
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            
            # Create grouped bar chart
            x_pos = np.arange(len(param_df))
            width = 0.25
            
            fig.add_trace(
                go.Bar(
                    x=param_df['Stock'],
                    y=param_df['Alpha'],
                    name='α (Alpha)',
                    marker_color=self.color_scheme['primary'],
                    width=width
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=param_df['Stock'],
                    y=param_df['Beta'],
                    name='β (Beta)',
                    marker_color=self.color_scheme['secondary'],
                    width=width
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=param_df['Stock'],
                    y=param_df['Persistence'],
                    name='α+β (Persistence)',
                    marker_color=self.color_scheme['danger'],
                    width=width
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text="Volatility Analysis Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Volatility (%)", row=1, col=1)
        fig.update_xaxes(title_text="Stock", row=2, col=1)
        fig.update_yaxes(title_text="Parameter Value", row=2, col=1)
        
        return fig
    
    def create_granger_heatmap(self, var_results: Dict) -> go.Figure:
        """
        Create Granger causality heatmap
        
        Args:
            var_results: VAR analysis results
            
        Returns:
            Plotly heatmap figure
        """
        if 'granger_causality' not in var_results:
            return go.Figure()
        
        granger_data = var_results['granger_causality']
        
        # Extract all variables
        variables = set()
        for relationship in granger_data.keys():
            parts = relationship.split('_causes_')
            if len(parts) == 2:
                variables.update(parts)
        
        variables = sorted(list(variables))
        n_vars = len(variables)
        
        # Create matrix
        p_value_matrix = np.ones((n_vars, n_vars))
        
        for i, var_to in enumerate(variables):
            for j, var_from in enumerate(variables):
                if i != j:
                    key = f"{var_from}_causes_{var_to}"
                    if key in granger_data:
                        p_value_matrix[i, j] = granger_data[key]['min_p_value']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=p_value_matrix,
            x=variables,
            y=variables,
            colorscale='RdYlGn_r',
            zmid=0.05,
            text=np.round(p_value_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="P-value"),
            hovertemplate="From: %{x}<br>To: %{y}<br>P-value: %{z:.4f}<extra></extra>"
        ))
        
        # Add significance threshold line
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n_vars-0.5,
            y0=-0.5, y1=n_vars-0.5,
            line=dict(color="gray", width=2, dash="dot")
        )
        
        fig.update_layout(
            title="Granger Causality P-values (< 0.05 = significant)",
            xaxis_title="Causes (From)",
            yaxis_title="Effect (To)",
            height=600,
            width=700
        )
        
        return fig


def create_interpretation_cards(var_results: Dict, garch_results: Dict,
                               summary_stats: Dict) -> Dict[str, Any]:
    """
    Create interpretation cards for display in dashboard
    
    Args:
        var_results: VAR analysis results
        garch_results: GARCH analysis results
        summary_stats: Summary statistics
        
    Returns:
        Dictionary with interpretation content
    """
    interpretations = {
        'var_insights': [],
        'garch_insights': [],
        'recommendations': [],
        'risk_warnings': []
    }
    
    # VAR Insights
    if var_results and 'granger_causality' in var_results:
        granger_data = var_results['granger_causality']
        significant_count = sum(1 for d in granger_data.values() 
                              if d.get('is_significant', False))
        
        if significant_count > 0:
            interpretations['var_insights'].append({
                'title': 'Lead-Lag Relationships',
                'content': f'Found {significant_count} significant causal relationships',
                'type': 'info'
            })
    
    # GARCH Insights
    high_persistence_stocks = []
    for stock, result in garch_results.items():
        if 'parameters' in result and result['parameters'].get('converged'):
            persistence = result['parameters'].get('persistence', 0)
            if persistence > 0.95:
                high_persistence_stocks.append(stock)
    
    if high_persistence_stocks:
        interpretations['garch_insights'].append({
            'title': 'High Volatility Persistence',
            'content': f'Stocks with persistent volatility: {", ".join(high_persistence_stocks)}',
            'type': 'warning'
        })
    
    # Risk Warnings
    if summary_stats.get('avg_garch_persistence', 0) > 0.9:
        interpretations['risk_warnings'].append({
            'title': 'Portfolio Volatility Risk',
            'content': 'High average volatility persistence detected. Consider risk reduction strategies.',
            'type': 'danger'
        })
    
    # Recommendations
    if var_results.get('optimal_lags'):
        interpretations['recommendations'].append({
            'title': 'Optimal Trading Horizon',
            'content': f'VAR model suggests {var_results["optimal_lags"]} day lag for best predictions',
            'type': 'success'
        })
    
    return interpretations


# Export main classes and functions
__all__ = [
    'VARGARCHVisualizer',
    'create_interpretation_cards'
]
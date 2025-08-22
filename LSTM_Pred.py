"""
Enhanced LSTM Trading Model with Advanced Features
Comprehensive improvements for better performance and showcase value
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Bidirectional, GRU, Input, Attention, 
                                   MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import logging
from typing import Tuple, Dict, Optional, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ta  # Technical analysis library
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class AdvancedFeatureEngineering:
    """Advanced feature engineering for financial time series"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if price_col not in df.columns:
            # If specific price column not found, use the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for technical indicators")
        
        df = df.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'ema_{window}'] = df[price_col].ewm(span=window).mean()
        
        # Price ratios
        df['price_sma5_ratio'] = df[price_col] / df['sma_5']
        df['price_sma20_ratio'] = df[price_col] / df['sma_20']
        df['sma5_sma20_ratio'] = df['sma_5'] / df['sma_20']
        
        # Volatility measures
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'rolling_max_{window}'] = df[price_col].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df[price_col].rolling(window=window).min()
            df[f'price_position_{window}'] = (df[price_col] - df[f'rolling_min_{window}']) / \
                                           (df[f'rolling_max_{window}'] - df[f'rolling_min_{window}'] + 1e-8)
        
        # Momentum indicators
        for period in [5, 10, 14, 20]:
            df[f'rsi_{period}'] = AdvancedFeatureEngineering._calculate_rsi(df[price_col], period)
            df[f'momentum_{period}'] = df[price_col] / df[price_col].shift(period) - 1
        
        # MACD
        ema12 = df[price_col].ewm(span=12).mean()
        ema26 = df[price_col].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['sma_20']
        std20 = df[price_col].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume-based indicators (if volume exists)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df[price_col] * df['volume']
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Higher order features
        df['returns_squared'] = df['returns'] ** 2
        df['log_volume'] = np.log(df.get('volume', pd.Series(1, index=df.index)) + 1)
        
        # Regime detection
        df['high_volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(60).quantile(0.75)).astype(int)
        df['trend_strength'] = abs(df['sma_5'] - df['sma_20']) / df['sma_20']
        
        return df.dropna()
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Key interactions for financial data
        feature_pairs = [
            ('returns', 'volatility_5'),
            ('rsi_14', 'momentum_10'),
            ('macd', 'bb_position'),
            ('price_sma5_ratio', 'volume_ratio') if 'volume_ratio' in df.columns else None,
            ('trend_strength', 'volatility_10')
        ]
        
        for pair in feature_pairs:
            if pair and all(col in df.columns for col in pair):
                df[f'{pair[0]}_{pair[1]}_interaction'] = df[pair[0]] * df[pair[1]]
        
        return df


class EnhancedLSTMPredictor:
    """Enhanced LSTM with attention, better architecture, and advanced features"""
    
    def __init__(self, 
                 seq_length: int = 30,
                 lstm_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 use_attention: bool = True,
                 use_bidirectional: bool = True,
                 feature_selection: bool = True,
                 n_features: int = 20):
        
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_bidirectional = use_bidirectional
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()  # Use StandardScaler for better gradient flow
        self.feature_selector = None
        self.selected_features = None
        self.target_scaler = RobustScaler()  # Separate scaler for target
        
    def prepare_data(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced data preparation with feature engineering and selection"""
        logger.info("Starting enhanced data preparation...")
        
        # Feature engineering
        data_engineered = AdvancedFeatureEngineering.add_technical_indicators(data, target_col)
        data_engineered = AdvancedFeatureEngineering.create_interaction_features(data_engineered, target_col)
        
        # Remove infinite values and handle outliers
        data_engineered = data_engineered.replace([np.inf, -np.inf], np.nan)
        data_engineered = data_engineered.dropna()
        
        # Outlier removal using IQR method
        for col in data_engineered.select_dtypes(include=[np.number]).columns:
            Q1 = data_engineered[col].quantile(0.01)
            Q3 = data_engineered[col].quantile(0.99)
            data_engineered = data_engineered[(data_engineered[col] >= Q1) & (data_engineered[col] <= Q3)]
        
        # Separate target and features
        y = data_engineered[target_col].values
        X_features = data_engineered.drop(columns=[target_col])
        
        # Feature selection
        if self.feature_selection and len(X_features.columns) > self.n_features:
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
            X_selected = self.feature_selector.fit_transform(X_features, y)
            self.selected_features = X_features.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features[:10]}...")
        else:
            X_selected = X_features.values
            self.selected_features = X_features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Scale target separately for better reconstruction
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_scaled
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with improved temporal structure"""
        if len(X) <= self.seq_length:
            raise ValueError(f"Data length ({len(X)}) must be greater than sequence length ({self.seq_length})")
        
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:(i + self.seq_length)])
            y_seq.append(y[i + self.seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build enhanced model with attention mechanism"""
        logger.info(f"Building enhanced model with input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            
            lstm_layer = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate * 0.5,
                recurrent_dropout=0.1,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )
            
            if self.use_bidirectional:
                x = Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)
            
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        if self.use_attention and len(x.shape) == 3:
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=4,
                key_dim=x.shape[-1] // 4,
                dropout=self.dropout_rate
            )(x, x)
            
            # Add & Norm
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
            
            # Global average pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with residual connections
        for i, units in enumerate(self.dense_units):
            prev_x = x
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate * 0.5)(x)
            
            # Residual connection if dimensions match
            if prev_x.shape[-1] == units:
                x = Add()([prev_x, x])
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Enhanced model built with {model.count_params():,} parameters")
        return model
    
    def train(self, 
              data: pd.DataFrame,
              target_col: str,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 64,
              patience: int = 20) -> Dict[str, Any]:
        """Enhanced training with better callbacks and monitoring"""
        
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(data, target_col)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Build model
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                mode='min'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_train = self.model.predict(X_train, verbose=0)
        y_pred_val = self.model.predict(X_val, verbose=0)
        
        # Inverse transform predictions
        y_train_inv = self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_train_inv = self.target_scaler.inverse_transform(y_pred_train).flatten()
        y_val_inv = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_val_inv = self.target_scaler.inverse_transform(y_pred_val).flatten()
        
        return {
            'history': history,
            'train_actual': y_train_inv,
            'train_predicted': y_pred_train_inv,
            'val_actual': y_val_inv,
            'val_predicted': y_pred_val_inv,
            'model': self.model
        }
    
    def predict_future(self, 
                       data: pd.DataFrame,
                       target_col: str,
                       days_ahead: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced future prediction with confidence intervals"""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare the last sequence
        X_scaled, _ = self.prepare_data(data, target_col)
        last_sequence = X_scaled[-self.seq_length:].copy()
        
        predictions = []
        
        for _ in range(days_ahead):
            # Predict next value
            X_pred = last_sequence.reshape(1, self.seq_length, -1)
            next_pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Store prediction
            predictions.append(next_pred_scaled)
            
            # Update sequence (simple approach - in practice, you'd need to engineer features for the new timestep)
            new_features = last_sequence[-1].copy()
            if self.selected_features and target_col in self.selected_features:
                target_idx = self.selected_features.index(target_col) if target_col in self.selected_features else 0
                new_features[target_idx] = next_pred_scaled
            
            # Roll the sequence
            last_sequence = np.vstack([last_sequence[1:], new_features])
        
        # Convert predictions back to original scale
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        # Simple confidence intervals (could be enhanced with Monte Carlo dropout)
        std_error = np.std(predictions_inv) if len(predictions_inv) > 1 else np.abs(predictions_inv[0]) * 0.1
        confidence_lower = predictions_inv - 1.96 * std_error
        confidence_upper = predictions_inv + 1.96 * std_error
        
        return predictions_inv, confidence_lower, confidence_upper
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        # Basic metrics
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Financial metrics
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        
        # Directional accuracy
        if len(y_true_clean) > 1:
            true_direction = np.diff(y_true_clean) > 0
            pred_direction = np.diff(y_pred_clean) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Theil's U statistic
        theil_u = np.sqrt(np.mean((y_pred_clean - y_true_clean)**2)) / \
                  (np.sqrt(np.mean(y_true_clean**2)) + np.sqrt(np.mean(y_pred_clean**2)) + 1e-8)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Theil_U': theil_u
        }


class ModelShowcase:
    """Create compelling visualizations for model showcase"""
    
    @staticmethod
    def create_comprehensive_dashboard(predictor: EnhancedLSTMPredictor, 
                                     results: Dict[str, Any],
                                     metrics: Dict[str, float]) -> None:
        """Create a comprehensive dashboard showing model capabilities"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Training History
        ax1 = plt.subplot(3, 3, 1)
        history = results['history']
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Training Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Training)
        ax2 = plt.subplot(3, 3, 2)
        train_actual = results['train_actual']
        train_pred = results['train_predicted']
        plt.scatter(train_actual, train_pred, alpha=0.6, s=20)
        plt.plot([train_actual.min(), train_actual.max()], 
                [train_actual.min(), train_actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Training: R² = {metrics.get("R2", 0):.3f}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Prediction Time Series
        ax3 = plt.subplot(3, 3, 3)
        val_actual = results['val_actual']
        val_pred = results['val_predicted']
        plt.plot(val_actual, label='Actual', linewidth=2, alpha=0.8)
        plt.plot(val_pred, label='Predicted', linewidth=2, alpha=0.8)
        plt.title('Validation Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Residuals Analysis
        ax4 = plt.subplot(3, 3, 4)
        residuals = val_actual - val_pred
        plt.hist(residuals, bins=30, alpha=0.7, density=True)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 5. Feature Importance (if available)
        ax5 = plt.subplot(3, 3, 5)
        if hasattr(predictor, 'selected_features') and predictor.selected_features:
            feature_names = predictor.selected_features[:10]  # Top 10 features
            importance_scores = np.random.rand(len(feature_names))  # Placeholder
            plt.barh(range(len(feature_names)), importance_scores)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.title('Top Features', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3)
        
        # 6. Metrics Summary
        ax6 = plt.subplot(3, 3, 6)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
        
        bars = plt.bar(range(len(metric_names)), metric_values, color=colors)
        plt.xticks(range(len(metric_names)), metric_names, rotation=45)
        plt.title('Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 7. Error Distribution Over Time
        ax7 = plt.subplot(3, 3, 7)
        errors = np.abs(val_actual - val_pred)
        plt.plot(errors, linewidth=2, alpha=0.8)
        plt.axhline(y=np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean Error: {np.mean(errors):.3f}')
        plt.title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Directional Accuracy
        ax8 = plt.subplot(3, 3, 8)
        if len(val_actual) > 1:
            actual_direction = np.diff(val_actual) > 0
            pred_direction = np.diff(val_pred) > 0
            correct_direction = actual_direction == pred_direction
            
            # Rolling directional accuracy
            window = min(20, len(correct_direction) // 4)
            rolling_accuracy = pd.Series(correct_direction.astype(float)).rolling(window).mean() * 100
            
            plt.plot(rolling_accuracy, linewidth=2)
            plt.axhline(y=50, color='red', linestyle='--', label='Random (50%)')
            plt.title(f'Rolling Directional Accuracy\nOverall: {metrics.get("Directional_Accuracy", 0):.1f}%', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 9. Model Architecture Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create text summary of model
        model_info = f"""
        Model Architecture:
        • LSTM Units: {predictor.lstm_units}
        • Dense Units: {predictor.dense_units}
        • Attention: {'Yes' if predictor.use_attention else 'No'}
        • Bidirectional: {'Yes' if predictor.use_bidirectional else 'No'}
        • Sequence Length: {predictor.seq_length}
        • Features: {len(predictor.selected_features) if predictor.selected_features else 'Unknown'}
        
        Key Performance:
        • R² Score: {metrics.get('R2', 0):.3f}
        • RMSE: {metrics.get('RMSE', 0):.3f}
        • Direction Acc: {metrics.get('Directional_Accuracy', 0):.1f}%
        """
        
        ax9.text(0.1, 0.9, model_info, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.suptitle('Enhanced LSTM Model Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.show()
    
    @staticmethod
    def create_future_prediction_plot(future_pred: np.ndarray,
                                    conf_lower: np.ndarray,
                                    conf_upper: np.ndarray,
                                    historical_data: np.ndarray = None,
                                    title: str = "Future Predictions") -> None:
        """Create an attractive future prediction plot"""
        
        plt.figure(figsize=(15, 8))
        
        days_ahead = len(future_pred)
        future_dates = range(days_ahead)
        
        # Plot historical data if provided
        if historical_data is not None:
            hist_dates = range(-len(historical_data), 0)
            plt.plot(hist_dates, historical_data, 'b-', linewidth=2, 
                    label='Historical Data', alpha=0.8)
        
        # Plot predictions
        plt.plot(future_dates, future_pred, 'r-', linewidth=3, 
                label='Predicted', marker='o', markersize=4)
        
        # Plot confidence intervals
        plt.fill_between(future_dates, conf_lower, conf_upper, 
                        alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(1, np.mean([conf_lower.min(), conf_upper.max()]), 'Future Predictions', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_sample_data(n_samples: int = 1000, add_noise: bool = True) -> pd.DataFrame:
    """Create realistic sample financial data for testing"""
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate realistic price series with trends, volatility, and patterns
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, n_samples) + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    # Add volatility clustering
    volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 8*np.pi, n_samples))**2
    
    # Generate returns with autocorrelation
    returns = np.zeros(n_samples)
    for i in range(1, n_samples):
        returns[i] = 0.7 * returns[i-1] + volatility[i] * np.random.normal(0, 1)
    
    # Convert to prices
    prices = trend * np.exp(np.cumsum(returns))
    
    # Add some noise if requested
    if add_noise:
        noise = np.random.normal(0, prices.std() * 0.01, n_samples)
        prices += noise
    
    # Create volume data
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_samples))),
        'open': prices + np.random.normal(0, prices.std() * 0.01, n_samples)
    }, index=dates)
    
    return df


def run_enhanced_model_demo(data: pd.DataFrame = None, target_col: str = 'close') -> Dict[str, Any]:
    """Run a complete demonstration of the enhanced model"""
    
    print("🚀 Enhanced LSTM Trading Model Demo")
    print("=" * 50)
    
    # Use sample data if none provided
    if data is None:
        print("📊 Generating sample financial data...")
        data = create_sample_data(n_samples=800)
        print(f"✅ Created dataset with {len(data)} samples")
    
    # Initialize enhanced predictor
    print("\n🏗️ Building Enhanced LSTM Predictor...")
    predictor = EnhancedLSTMPredictor(
        seq_length=30,
        lstm_units=[128, 64],
        dense_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        use_attention=True,
        use_bidirectional=True,
        feature_selection=True,
        n_features=20
    )
    
    # Train the model
    print("\n🎯 Training Enhanced Model...")
    try:
        results = predictor.train(
            data=data,
            target_col=target_col,
            validation_split=0.2,
            epochs=50,  # Reduced for demo
            batch_size=64,
            patience=15
        )
        print("✅ Training completed successfully!")
        
        # Calculate metrics
        print("\n📈 Calculating Performance Metrics...")
        train_metrics = predictor.calculate_metrics(
            results['train_actual'], 
            results['train_predicted']
        )
        val_metrics = predictor.calculate_metrics(
            results['val_actual'], 
            results['val_predicted']
        )
        
        print(f"📊 Training R²: {train_metrics.get('R2', 0):.4f}")
        print(f"📊 Validation R²: {val_metrics.get('R2', 0):.4f}")
        print(f"📊 Directional Accuracy: {val_metrics.get('Directional_Accuracy', 0):.2f}%")
        
        # Generate future predictions
        print("\n🔮 Generating Future Predictions...")
        future_pred, conf_lower, conf_upper = predictor.predict_future(
            data=data,
            target_col=target_col,
            days_ahead=30
        )
        
        # Create comprehensive dashboard
        print("\n📊 Creating Performance Dashboard...")
        ModelShowcase.create_comprehensive_dashboard(
            predictor, results, val_metrics
        )
        
        # Create future prediction plot
        print("\n🎯 Creating Future Prediction Visualization...")
        historical_prices = data[target_col].tail(60).values
        ModelShowcase.create_future_prediction_plot(
            future_pred, conf_lower, conf_upper, 
            historical_prices, "30-Day Price Forecast"
        )
        
        # Print detailed results
        print("\n" + "="*50)
        print("🎉 ENHANCED MODEL RESULTS SUMMARY")
        print("="*50)
        
        print(f"📈 Model Performance:")
        for metric, value in val_metrics.items():
            print(f"   • {metric}: {value:.4f}")
        
        print(f"\n🔮 Future Predictions:")
        print(f"   • Next 7 days avg: ${future_pred[:7].mean():.2f}")
        print(f"   • Next 30 days avg: ${future_pred.mean():.2f}")
        print(f"   • Predicted trend: {'📈 Upward' if future_pred[-1] > future_pred[0] else '📉 Downward'}")
        
        print(f"\n🏗️ Model Architecture:")
        print(f"   • Total Parameters: {predictor.model.count_params():,}")
        print(f"   • Features Used: {len(predictor.selected_features)}")
        print(f"   • Sequence Length: {predictor.seq_length}")
        print(f"   • Attention Enabled: {predictor.use_attention}")
        
        return {
            'predictor': predictor,
            'results': results,
            'metrics': val_metrics,
            'future_predictions': future_pred,
            'confidence_intervals': (conf_lower, conf_upper)
        }
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Additional utility functions for production use
def optimize_hyperparameters(data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Run hyperparameter optimization"""
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    print("🔧 Starting Hyperparameter Optimization...")
    
    param_distributions = {
        'seq_length': randint(20, 60),
        'lstm_units': [[64], [128], [64, 32], [128, 64], [128, 64, 32]],
        'dropout_rate': uniform(0.1, 0.4),
        'learning_rate': uniform(0.0001, 0.01),
        'use_attention': [True, False],
        'use_bidirectional': [True, False]
    }
    
    best_score = np.inf
    best_params = None
    
    # Simple random search (would need more sophisticated approach for production)
    for i in range(10):  # Limited iterations for demo
        print(f"🔄 Trial {i+1}/10")
        
        # Sample random parameters
        params = {}
        for key, values in param_distributions.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
            else:
                params[key] = values.rvs()
        
        try:
            predictor = EnhancedLSTMPredictor(**params)
            results = predictor.train(data, target_col, epochs=20, patience=5)
            
            val_metrics = predictor.calculate_metrics(
                results['val_actual'], 
                results['val_predicted']
            )
            
            score = val_metrics.get('RMSE', np.inf)
            if score < best_score:
                best_score = score
                best_params = params
                print(f"✅ New best score: {score:.4f}")
        
        except Exception as e:
            print(f"❌ Trial failed: {str(e)}")
            continue
    
    print(f"\n🏆 Best Parameters Found:")
    for key, value in best_params.items():
        print(f"   • {key}: {value}")
    
    return {'best_params': best_params, 'best_score': best_score}


def create_trading_strategy(predictor: EnhancedLSTMPredictor, 
                          data: pd.DataFrame,
                          target_col: str) -> pd.DataFrame:
    """Create a simple trading strategy based on predictions"""
    
    print("💰 Creating Trading Strategy...")
    
    # Make predictions for the entire dataset
    X_scaled, _ = predictor.prepare_data(data, target_col)
    X_seq, _ = predictor.create_sequences(X_scaled, np.zeros(len(X_scaled)))
    
    predictions_scaled = predictor.model.predict(X_seq, verbose=0)
    predictions = predictor.target_scaler.inverse_transform(predictions_scaled).flatten()
    
    # Align predictions with dates
    prediction_dates = data.index[predictor.seq_length:]
    actual_prices = data[target_col].iloc[predictor.seq_length:].values
    
    # Create strategy DataFrame
    strategy_df = pd.DataFrame({
        'date': prediction_dates,
        'actual_price': actual_prices,
        'predicted_price': predictions,
        'predicted_return': np.diff(np.concatenate([[predictions[0]], predictions])),
        'actual_return': np.diff(np.concatenate([[actual_prices[0]], actual_prices]))
    })
    
    # Simple strategy: Buy when predicted return > threshold, Sell when < -threshold
    threshold = strategy_df['predicted_return'].std() * 0.5
    
    strategy_df['signal'] = np.where(
        strategy_df['predicted_return'] > threshold, 1,  # Buy
        np.where(strategy_df['predicted_return'] < -threshold, -1, 0)  # Sell or Hold
    )
    
    # Calculate strategy returns
    strategy_df['strategy_return'] = strategy_df['signal'].shift(1) * strategy_df['actual_return']
    strategy_df['cumulative_return'] = (1 + strategy_df['actual_return']).cumprod()
    strategy_df['cumulative_strategy_return'] = (1 + strategy_df['strategy_return'].fillna(0)).cumprod()
    
    # Calculate performance metrics
    total_return = strategy_df['cumulative_strategy_return'].iloc[-1] - 1
    buy_and_hold_return = strategy_df['cumulative_return'].iloc[-1] - 1
    
    print(f"📊 Strategy Performance:")
    print(f"   • Strategy Total Return: {total_return:.2%}")
    print(f"   • Buy & Hold Return: {buy_and_hold_return:.2%}")
    print(f"   • Excess Return: {total_return - buy_and_hold_return:.2%}")
    
    # Plot strategy performance
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(strategy_df['date'], strategy_df['cumulative_return'], 
             label='Buy & Hold', linewidth=2)
    plt.plot(strategy_df['date'], strategy_df['cumulative_strategy_return'], 
             label='LSTM Strategy', linewidth=2)
    plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    signal_colors = ['red' if x == -1 else 'green' if x == 1 else 'gray' for x in strategy_df['signal']]
    plt.scatter(strategy_df['date'], strategy_df['actual_price'], 
                c=signal_colors, alpha=0.6, s=20)
    plt.plot(strategy_df['date'], strategy_df['predicted_price'], 
             'b--', alpha=0.7, label='Predicted Price')
    plt.title('Trading Signals', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(['Predicted', 'Sell Signal', 'Hold', 'Buy Signal'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return strategy_df


# Example usage and main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced LSTM Trading Model Demonstration")
    
    # Run the main demo
    demo_results = run_enhanced_model_demo()
    
    if demo_results:
        print("\n🎯 Demo completed successfully!")
        
        # Optionally run hyperparameter optimization
        # print("\n🔧 Running Hyperparameter Optimization...")
        # optimization_results = optimize_hyperparameters(create_sample_data(), 'close')
        
        # Create trading strategy
        sample_data = create_sample_data()
        strategy_results = create_trading_strategy(
            demo_results['predictor'], 
            sample_data, 
            'close'
        )
    else:
        print("❌ Demo failed. Please check the error messages above.")
"""
Machine Learning Based Trading Strategies
機械学習ベースのトレード戦略
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def linear_regression_strategy(data: pd.DataFrame,
                              lookback_window: int = 30,
                              feature_lag: int = 1,
                              threshold: float = 0.01) -> pd.DataFrame:
    """
    Linear Regression Strategy
    線形回帰戦略
    
    Uses rolling linear regression to predict price direction
    """
    data = data.copy()
    
    # Create features
    data['returns'] = data['close'].pct_change()
    data['volume_ma'] = data['volume'].rolling(window=10).mean()
    data['price_ma'] = data['close'].rolling(window=10).mean()
    data['volatility'] = data['returns'].rolling(window=10).std()
    
    # Lag features
    for col in ['returns', 'volume_ma', 'price_ma', 'volatility']:
        for lag in range(1, feature_lag + 1):
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Rolling regression predictions
    data['prediction'] = np.nan
    data['prediction_error'] = np.nan
    
    feature_cols = [col for col in data.columns if 'lag' in col]
    
    for i in range(lookback_window + feature_lag, len(data)):
        try:
            # Prepare training data
            y_train = data['returns'].iloc[i-lookback_window:i].values
            X_train = data[feature_cols].iloc[i-lookback_window:i].values
            
            # Remove NaN rows
            valid_idx = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            if np.sum(valid_idx) > 10:
                X_train = X_train[valid_idx]
                y_train = y_train[valid_idx]
                
                # Simple linear regression using normal equations
                X_train = np.column_stack([np.ones(len(X_train)), X_train])  # Add bias term
                
                try:
                    # Solve using least squares
                    coeffs = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    
                    # Make prediction for current period
                    X_current = data[feature_cols].iloc[i].values
                    if not np.isnan(X_current).any():
                        X_current = np.concatenate([[1], X_current])  # Add bias
                        prediction = np.dot(X_current, coeffs)
                        data.iloc[i, data.columns.get_loc('prediction')] = prediction
                        
                        # Calculate prediction error
                        actual = data['returns'].iloc[i-1]  # Previous period actual
                        if not np.isnan(actual):
                            prev_pred = data['prediction'].iloc[i-1]
                            if not np.isnan(prev_pred):
                                error = abs(actual - prev_pred)
                                data.iloc[i, data.columns.get_loc('prediction_error')] = error
                                
                except np.linalg.LinAlgError:
                    pass
                    
        except Exception:
            pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when prediction is positive and above threshold
    buy_condition = data['prediction'] > threshold
    
    # Sell when prediction is negative and below threshold
    sell_condition = data['prediction'] < -threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def knn_strategy(data: pd.DataFrame,
                k_neighbors: int = 5,
                lookback_window: int = 50,
                feature_window: int = 5) -> pd.DataFrame:
    """
    K-Nearest Neighbors Strategy
    K近傍法戦略
    
    Uses KNN to find similar market patterns
    """
    data = data.copy()
    
    # Create features (price patterns)
    feature_cols = []
    for i in range(feature_window):
        col_name = f'return_lag_{i}'
        data[col_name] = data['close'].pct_change().shift(i)
        feature_cols.append(col_name)
    
    # Add volume features
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=10).mean()
    for i in range(feature_window):
        col_name = f'volume_lag_{i}'
        data[col_name] = data['volume_ratio'].shift(i)
        feature_cols.append(col_name)
    
    # Target variable (future return)
    data['future_return'] = data['close'].pct_change().shift(-1)
    
    # Rolling KNN predictions
    data['knn_prediction'] = np.nan
    
    for i in range(lookback_window + feature_window, len(data) - 1):
        try:
            # Prepare training data
            train_end = i
            train_start = i - lookback_window
            
            X_train = data[feature_cols].iloc[train_start:train_end].values
            y_train = data['future_return'].iloc[train_start:train_end].values
            
            # Current pattern
            X_current = data[feature_cols].iloc[i].values.reshape(1, -1)
            
            # Remove NaN rows
            valid_idx = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            if np.sum(valid_idx) > k_neighbors:
                X_train = X_train[valid_idx]
                y_train = y_train[valid_idx]
                
                # Calculate distances to all training points
                if not np.isnan(X_current).any():
                    distances = np.sqrt(np.sum((X_train - X_current) ** 2, axis=1))
                    
                    # Find k nearest neighbors
                    nearest_indices = np.argsort(distances)[:k_neighbors]
                    nearest_returns = y_train[nearest_indices]
                    
                    # Predict as average of nearest neighbors
                    prediction = np.mean(nearest_returns)
                    data.iloc[i, data.columns.get_loc('knn_prediction')] = prediction
                    
        except Exception:
            pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when KNN predicts positive return
    buy_condition = data['knn_prediction'] > 0.005
    
    # Sell when KNN predicts negative return
    sell_condition = data['knn_prediction'] < -0.005
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def ensemble_ml_strategy(data: pd.DataFrame,
                        lookback_window: int = 40,
                        ensemble_size: int = 3) -> pd.DataFrame:
    """
    Ensemble ML Strategy
    アンサンブル機械学習戦略
    
    Combines multiple simple ML models
    """
    data = data.copy()
    
    # Create comprehensive features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['high_low_ratio'] = data['high'] / data['low']
    data['volume_price_trend'] = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # Technical indicators as features
    for window in [5, 10, 20]:
        data[f'sma_{window}'] = data['close'].rolling(window).mean()
        data[f'price_sma_ratio_{window}'] = data['close'] / data[f'sma_{window}']
        data[f'volume_sma_{window}'] = data['volume'].rolling(window).mean()
        data[f'volatility_{window}'] = data['returns'].rolling(window).std()
    
    # Lag features
    feature_base = ['returns', 'log_returns', 'high_low_ratio', 'volume_price_trend']
    feature_base += [col for col in data.columns if any(x in col for x in ['sma', 'ratio', 'volatility'])]
    
    feature_cols = []
    for col in feature_base:
        for lag in range(1, 4):
            lag_col = f'{col}_lag_{lag}'
            data[lag_col] = data[col].shift(lag)
            feature_cols.append(lag_col)
    
    # Target
    data['target'] = data['returns'].shift(-1)
    
    # Ensemble predictions
    data['ensemble_pred'] = np.nan
    
    for i in range(lookback_window + 5, len(data) - 1):
        try:
            # Prepare data
            train_end = i
            train_start = i - lookback_window
            
            X_train = data[feature_cols].iloc[train_start:train_end]
            y_train = data['target'].iloc[train_start:train_end]
            X_current = data[feature_cols].iloc[i:i+1]
            
            # Remove NaN
            valid_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            if valid_mask.sum() > 15 and not X_current.isnull().any().any():
                X_train_clean = X_train[valid_mask].values
                y_train_clean = y_train[valid_mask].values
                X_current_clean = X_current.values
                
                predictions = []
                
                # Model 1: Linear Regression
                try:
                    X_with_bias = np.column_stack([np.ones(len(X_train_clean)), X_train_clean])
                    coeffs = np.linalg.lstsq(X_with_bias, y_train_clean, rcond=None)[0]
                    X_current_bias = np.column_stack([np.ones(1), X_current_clean])
                    pred1 = np.dot(X_current_bias, coeffs)[0]
                    predictions.append(pred1)
                except:
                    pass
                
                # Model 2: Simple Moving Average of Similar Patterns
                try:
                    # Find most similar patterns using correlation
                    correlations = []
                    for j in range(len(X_train_clean)):
                        corr = np.corrcoef(X_train_clean[j], X_current_clean[0])[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)
                    
                    # Use top 5 most similar patterns
                    top_indices = np.argsort(correlations)[-5:]
                    pred2 = np.mean(y_train_clean[top_indices])
                    predictions.append(pred2)
                except:
                    pass
                
                # Model 3: Trend Following
                try:
                    recent_trend = np.mean(y_train_clean[-5:])
                    pred3 = recent_trend * 0.8  # Momentum with decay
                    predictions.append(pred3)
                except:
                    pass
                
                # Ensemble average
                if predictions:
                    ensemble_pred = np.mean(predictions)
                    data.iloc[i, data.columns.get_loc('ensemble_pred')] = ensemble_pred
                    
        except Exception:
            pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when ensemble predicts positive return
    buy_condition = data['ensemble_pred'] > 0.003
    
    # Sell when ensemble predicts negative return
    sell_condition = data['ensemble_pred'] < -0.003
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def neural_network_simple_strategy(data: pd.DataFrame,
                                  lookback_window: int = 30,
                                  hidden_size: int = 5,
                                  learning_rate: float = 0.01) -> pd.DataFrame:
    """
    Simple Neural Network Strategy
    シンプルニューラルネットワーク戦略
    
    Implements a basic neural network from scratch
    """
    data = data.copy()
    
    # Prepare features
    data['returns'] = data['close'].pct_change()
    feature_cols = []
    
    # Create lagged features
    for lag in range(1, 6):
        col = f'return_lag_{lag}'
        data[col] = data['returns'].shift(lag)
        feature_cols.append(col)
    
    for lag in range(1, 4):
        col = f'volume_lag_{lag}'
        data[col] = (data['volume'] / data['volume'].rolling(10).mean()).shift(lag)
        feature_cols.append(col)
    
    # Target
    data['target'] = data['returns'].shift(-1)
    
    # Simple neural network implementation
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    data['nn_prediction'] = np.nan
    
    for i in range(lookback_window + 10, len(data) - 1):
        try:
            # Prepare training data
            X_train = data[feature_cols].iloc[i-lookback_window:i].values
            y_train = data['target'].iloc[i-lookback_window:i].values
            
            # Remove NaN and normalize
            valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            if np.sum(valid_mask) > 15:
                X_train = X_train[valid_mask]
                y_train = y_train[valid_mask]
                
                # Normalize features
                X_mean = np.mean(X_train, axis=0)
                X_std = np.std(X_train, axis=0) + 1e-8
                X_train_norm = (X_train - X_mean) / X_std
                
                # Initialize weights
                np.random.seed(42 + i)  # For reproducibility
                input_size = X_train_norm.shape[1]
                
                # Simple single hidden layer network
                W1 = np.random.normal(0, 0.1, (input_size, hidden_size))
                b1 = np.zeros((1, hidden_size))
                W2 = np.random.normal(0, 0.1, (hidden_size, 1))
                b2 = np.zeros((1, 1))
                
                # Training loop (simplified)
                for epoch in range(10):  # Limited epochs for speed
                    # Forward pass
                    z1 = np.dot(X_train_norm, W1) + b1
                    a1 = sigmoid(z1)
                    z2 = np.dot(a1, W2) + b2
                    predictions = z2
                    
                    # Loss
                    loss = np.mean((predictions.flatten() - y_train) ** 2)
                    
                    # Backward pass
                    dz2 = 2 * (predictions.flatten() - y_train).reshape(-1, 1) / len(y_train)
                    dW2 = np.dot(a1.T, dz2)
                    db2 = np.sum(dz2, axis=0, keepdims=True)
                    
                    da1 = np.dot(dz2, W2.T)
                    dz1 = da1 * sigmoid_derivative(a1)
                    dW1 = np.dot(X_train_norm.T, dz1)
                    db1 = np.sum(dz1, axis=0, keepdims=True)
                    
                    # Update weights
                    W2 -= learning_rate * dW2
                    b2 -= learning_rate * db2
                    W1 -= learning_rate * dW1
                    b1 -= learning_rate * db1
                
                # Make prediction for current point
                X_current = data[feature_cols].iloc[i].values.reshape(1, -1)
                if not np.isnan(X_current).any():
                    X_current_norm = (X_current - X_mean) / X_std
                    
                    # Forward pass for prediction
                    z1_pred = np.dot(X_current_norm, W1) + b1
                    a1_pred = sigmoid(z1_pred)
                    z2_pred = np.dot(a1_pred, W2) + b2
                    
                    prediction = z2_pred[0, 0]
                    data.iloc[i, data.columns.get_loc('nn_prediction')] = prediction
                    
        except Exception:
            pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when NN predicts positive return
    buy_condition = data['nn_prediction'] > 0.002
    
    # Sell when NN predicts negative return
    sell_condition = data['nn_prediction'] < -0.002
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data
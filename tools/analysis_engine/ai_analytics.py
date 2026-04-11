"""
AI Analytics Module for NEPSE Analysis Tool
Advanced machine learning and AI-powered analytics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class AIAnalytics:
    """AI-powered analytics and predictions for stock analysis"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        try:
            if data is None or data.empty:
                return pd.DataFrame()
                
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns'] = data['Close'].pct_change()
            features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['high_low_ratio'] = data['High'] / data['Low']
            features['close_open_ratio'] = data['Close'] / data['Open']
            
            # Moving averages
            features['ma_5'] = data['Close'].rolling(window=5).mean()
            features['ma_10'] = data['Close'].rolling(window=10).mean()
            features['ma_20'] = data['Close'].rolling(window=20).mean()
            features['ma_50'] = data['Close'].rolling(window=50).mean()
            
            # Technical indicators as features
            features['rsi'] = self._calculate_rsi(data['Close'])
            features['macd'], features['macd_signal'], _ = self._calculate_macd(data['Close'])
            features['bollinger_upper'], features['bollinger_middle'], features['bollinger_lower'] = self._calculate_bollinger_bands(data['Close'])
            
            # Volume features
            features['volume_sma'] = data['Volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = data['Close'].shift(lag)
                features[f'return_lag_{lag}'] = features['returns'].shift(lag)
                
            # Target variable (next day's return)
            features['target'] = features['returns'].shift(-1)
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            return pd.DataFrame()
            
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for feature engineering"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD for feature engineering"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
        
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands for feature engineering"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
        
    def train_price_prediction_model(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train machine learning model for price prediction"""
        try:
            features = self.prepare_features(data)
            if features.empty:
                return {'error': 'No features available for training'}
                
            # Prepare training data
            feature_columns = [col for col in features.columns if col != 'target']
            X = features[feature_columns]
            y = features['target']
            
            if len(X) < 50:  # Need minimum data for training
                return {'error': 'Insufficient data for training'}
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {}
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            models['random_forest'] = {
                'model': rf_model,
                'mse': mean_squared_error(y_test, rf_pred),
                'r2': r2_score(y_test, rf_pred),
                'predictions': rf_pred
            }
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            models['linear_regression'] = {
                'model': lr_model,
                'mse': mean_squared_error(y_test, lr_pred),
                'r2': r2_score(y_test, lr_pred),
                'predictions': lr_pred
            }
            
            # Store models and scaler
            self.models[symbol] = models
            self.scalers[symbol] = scaler
            
            # Calculate ensemble predictions
            ensemble_pred = (rf_pred + lr_pred) / 2
            models['ensemble'] = {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'r2': r2_score(y_test, ensemble_pred),
                'predictions': ensemble_pred
            }
            
            self.is_trained = True
            
            return {
                'symbol': symbol,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'models': models,
                'feature_importance': self._get_feature_importance(rf_model, feature_columns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train prediction model for {symbol}: {e}")
            return {'error': str(e)}
            
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_columns, importance))
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return {}
            
    def predict_next_day_return(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Predict next day's return for a stock"""
        try:
            if symbol not in self.models or not self.is_trained:
                return {'error': 'Model not trained for this symbol'}
                
            features = self.prepare_features(data)
            if features.empty:
                return {'error': 'No features available for prediction'}
                
            # Get latest features
            feature_columns = [col for col in features.columns if col != 'target']
            latest_features = features[feature_columns].iloc[-1:].values
            
            # Scale features
            scaler = self.scalers[symbol]
            scaled_features = scaler.transform(latest_features)
            
            # Make predictions with all models
            predictions = {}
            for model_name, model_info in self.models[symbol].items():
                if model_name != 'ensemble':
                    model = model_info['model']
                    pred = model.predict(scaled_features)[0]
                    predictions[model_name] = pred
                    
            # Ensemble prediction
            if len(predictions) > 1:
                ensemble_pred = np.mean(list(predictions.values()))
                predictions['ensemble'] = ensemble_pred
                
            return {
                'symbol': symbol,
                'predicted_return': predictions.get('ensemble', 0),
                'model_predictions': predictions,
                'confidence': self._calculate_prediction_confidence(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to predict for {symbol}: {e}")
            return {'error': str(e)}
            
    def _calculate_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence score for predictions"""
        try:
            if len(predictions) < 2:
                return 0.0
                
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            
            # Higher confidence when models agree more
            confidence = 1.0 / (1.0 + std_dev)
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence: {e}")
            return 0.0
            
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous price movements using isolation forest"""
        try:
            features = self.prepare_features(data)
            if features.empty:
                return {'error': 'No features available for anomaly detection'}
                
            # Use only numerical features for anomaly detection
            numerical_features = features.select_dtypes(include=[np.number])
            
            # Train isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(numerical_features)
            
            # Add anomaly labels to original data
            data_with_anomalies = data.copy()
            data_with_anomalies = data_with_anomalies.iloc[len(data_with_anomalies) - len(anomaly_labels):]
            data_with_anomalies['anomaly'] = anomaly_labels
            data_with_anomalies['anomaly_score'] = iso_forest.decision_function(numerical_features)
            
            # Get anomaly statistics
            anomalies = data_with_anomalies[data_with_anomalies['anomaly'] == -1]
            
            return {
                'total_anomalies': len(anomalies),
                'anomaly_percentage': (len(anomalies) / len(data_with_anomalies)) * 100,
                'anomaly_dates': anomalies.index.tolist(),
                'anomaly_data': anomalies[['Close', 'Volume']].to_dict('records'),
                'recent_anomalies': anomalies.tail(5).to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies: {e}")
            return {'error': str(e)}
            
    def cluster_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Cluster stocks based on their characteristics"""
        try:
            if len(stock_data) < 3:
                return {'error': 'Need at least 3 stocks for clustering'}
                
            # Prepare features for clustering
            stock_features = []
            stock_symbols = []
            
            for symbol, data in stock_data.items():
                if data is not None and len(data) > 30:
                    # Calculate stock characteristics
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    avg_return = returns.mean() * 252  # Annualized
                    avg_volume = data['Volume'].mean()
                    price_level = data['Close'].mean()
                    
                    features = [volatility, avg_return, avg_volume, price_level]
                    stock_features.append(features)
                    stock_symbols.append(symbol)
                    
            if len(stock_features) < 3:
                return {'error': 'Insufficient data for clustering'}
                
            # Perform clustering
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(stock_features)
            
            # Determine optimal number of clusters (2-5)
            cluster_results = {}
            for n_clusters in range(2, min(6, len(stock_features))):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                cluster_results[n_clusters] = {
                    'labels': cluster_labels.tolist(),
                    'centers': kmeans.cluster_centers_.tolist(),
                    'inertia': kmeans.inertia_,
                    'clusters': {}
                }
                
                # Group stocks by cluster
                for i, symbol in enumerate(stock_symbols):
                    cluster_id = cluster_labels[i]
                    if cluster_id not in cluster_results[n_clusters]['clusters']:
                        cluster_results[n_clusters]['clusters'][cluster_id] = []
                    cluster_results[n_clusters]['clusters'][cluster_id].append(symbol)
                    
            # Select best number of clusters using elbow method
            best_n_clusters = 3  # Default
            if len(cluster_results) > 1:
                inertias = [result['inertia'] for result in cluster_results.values()]
                # Simple elbow detection
                if len(inertias) > 2:
                    diffs = np.diff(inertias)
                    best_n_clusters = np.argmin(diffs) + 2  # +2 because we start from 2 clusters
                    
            return {
                'best_n_clusters': best_n_clusters,
                'cluster_results': cluster_results,
                'stock_clusters': cluster_results[best_n_clusters]['clusters'],
                'cluster_centers': cluster_results[best_n_clusters]['centers']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cluster stocks: {e}")
            return {'error': str(e)}
            
    def generate_portfolio_optimization(self, portfolio_data: Dict[str, Dict], stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate portfolio optimization suggestions"""
        try:
            if not portfolio_data or not stock_data:
                return {'error': 'Insufficient data for optimization'}
                
            # Calculate expected returns and volatility for each stock
            stock_stats = {}
            
            for symbol, holdings in portfolio_data.items():
                if symbol in stock_data and len(stock_data[symbol]) > 30:
                    data = stock_data[symbol]
                    returns = data['Close'].pct_change().dropna()
                    
                    expected_return = returns.mean() * 252  # Annualized
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    
                    stock_stats[symbol] = {
                        'expected_return': expected_return,
                        'volatility': volatility,
                        'shares': holdings['shares'],
                        'current_price': holdings['current_price'],
                        'weight': 0  # Will be calculated
                    }
                    
            if not stock_stats:
                return {'error': 'No valid stock data for optimization'}
                
            # Calculate current weights
            total_value = sum(stats['shares'] * stats['current_price'] for stats in stock_stats.values())
            for symbol in stock_stats:
                stock_value = stock_stats[symbol]['shares'] * stock_stats[symbol]['current_price']
                stock_stats[symbol]['weight'] = stock_value / total_value
                
            # Generate optimization suggestions
            suggestions = []
            
            # Risk parity suggestion
            equal_weight = 1.0 / len(stock_stats)
            for symbol, stats in stock_stats.items():
                current_weight = stats['weight']
                if abs(current_weight - equal_weight) > 0.1:  # 10% threshold
                    action = 'reduce' if current_weight > equal_weight else 'increase'
                    suggestions.append({
                        'type': 'risk_parity',
                        'symbol': symbol,
                        'action': action,
                        'current_weight': current_weight,
                        'suggested_weight': equal_weight,
                        'reason': 'Balance portfolio risk'
                    })
                    
            # High volatility reduction
            avg_volatility = np.mean([stats['volatility'] for stats in stock_stats.values()])
            for symbol, stats in stock_stats.items():
                if stats['volatility'] > avg_volatility * 1.5:
                    suggestions.append({
                        'type': 'volatility_reduction',
                        'symbol': symbol,
                        'action': 'reduce',
                        'current_volatility': stats['volatility'],
                        'avg_volatility': avg_volatility,
                        'reason': 'High volatility relative to portfolio'
                    })
                    
            return {
                'current_portfolio_stats': stock_stats,
                'optimization_suggestions': suggestions,
                'total_suggestions': len(suggestions),
                'portfolio_volatility': np.average([stats['volatility'] for stats in stock_stats.values()], 
                                                 weights=[stats['weight'] for stats in stock_stats.values()]),
                'portfolio_expected_return': np.average([stats['expected_return'] for stats in stock_stats.values()], 
                                                    weights=[stats['weight'] for stats in stock_stats.values()])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate portfolio optimization: {e}")
            return {'error': str(e)}

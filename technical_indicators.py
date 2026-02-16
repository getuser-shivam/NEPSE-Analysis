"""
Technical Indicators Module for NEPSE Analysis Tool
Contains all technical analysis calculations and indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging


class TechnicalIndicators:
    """Technical analysis indicators and calculations"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI
            
        except Exception as e:
            self.logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series()
            
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"Failed to calculate MACD: {e}")
            return pd.Series(), pd.Series(), pd.Series()
            
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            return upper_band, rolling_mean, lower_band
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Bollinger Bands: {e}")
            return pd.Series(), pd.Series(), pd.Series()
            
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            high_max = data['High'].rolling(window=k_period).max()
            low_min = data['Low'].rolling(window=k_period).min()
            
            k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent, d_percent
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Stochastic: {e}")
            return pd.Series(), pd.Series()
            
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            high_max = data['High'].rolling(window=period).max()
            low_min = data['Low'].rolling(window=period).min()
            
            wr = -100 * ((high_max - data['Close']) / (high_max - low_min))
            
            return wr
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Williams %R: {e}")
            return pd.Series()
            
    def calculate_moving_averages(self, prices: pd.Series, short_period: int = 20, long_period: int = 50) -> Tuple[pd.Series, pd.Series]:
        """Calculate Simple Moving Averages"""
        try:
            short_ma = prices.rolling(window=short_period).mean()
            long_ma = prices.rolling(window=long_period).mean()
            
            return short_ma, long_ma
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Moving Averages: {e}")
            return pd.Series(), pd.Series()
            
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        try:
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'] / volume_sma
            
            return {
                'volume_sma': volume_sma,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volume indicators: {e}")
            return {}
            
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate price volatility (standard deviation)"""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volatility: {e}")
            return pd.Series()
            
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift(1))
            low_close = np.abs(data['Low'] - data['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Failed to calculate ATR: {e}")
            return pd.Series()
            
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        try:
            rolling_max = data['High'].rolling(window=window).max()
            rolling_min = data['Low'].rolling(window=window).min()
            
            return {
                'resistance': rolling_max,
                'support': rolling_min
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate support/resistance: {e}")
            return {}
            
    def calculate_trend_strength(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        try:
            def calculate_slope(series):
                if len(series) < period:
                    return 0.0
                x = np.arange(len(series))
                y = series.values
                slope = np.polyfit(x, y, 1)[0]
                return slope
                
            return prices.rolling(window=period).apply(calculate_slope)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trend strength: {e}")
            return pd.Series()
            
    def get_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators for given data"""
        try:
            if data is None or data.empty:
                return {}
                
            close_prices = data['Close']
            
            indicators = {
                'rsi': self.calculate_rsi(close_prices),
                'macd': self.calculate_macd(close_prices),
                'bollinger_bands': self.calculate_bollinger_bands(close_prices),
                'stochastic': self.calculate_stochastic(data),
                'williams_r': self.calculate_williams_r(data),
                'moving_averages': self.calculate_moving_averages(close_prices),
                'volume_indicators': self.calculate_volume_indicators(data),
                'volatility': self.calculate_volatility(close_prices),
                'atr': self.calculate_atr(data),
                'support_resistance': self.calculate_support_resistance(data),
                'trend_strength': self.calculate_trend_strength(close_prices)
            }
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Failed to calculate all indicators: {e}")
            return {}
            
    def get_trading_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic trading signals based on indicators"""
        try:
            indicators = self.get_all_indicators(data)
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'neutral_signals': []
            }
            
            if not indicators:
                return signals
                
            # RSI signals
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1] if len(indicators['rsi']) > 0 else 50
                if rsi < 30:
                    signals['buy_signals'].append('RSI oversold')
                elif rsi > 70:
                    signals['sell_signals'].append('RSI overbought')
                    
            # MACD signals
            if 'macd' in indicators:
                macd_line, signal_line, histogram = indicators['macd']
                if len(macd_line) > 1 and len(signal_line) > 1:
                    macd_current = macd_line.iloc[-1]
                    macd_previous = macd_line.iloc[-2]
                    signal_current = signal_line.iloc[-1]
                    
                    if macd_previous < signal_previous and macd_current > signal_current:
                        signals['buy_signals'].append('MACD bullish crossover')
                    elif macd_previous > signal_previous and macd_current < signal_current:
                        signals['sell_signals'].append('MACD bearish crossover')
                        
            # Moving average signals
            if 'moving_averages' in indicators:
                short_ma, long_ma = indicators['moving_averages']
                if len(short_ma) > 0 and len(long_ma) > 0:
                    short_current = short_ma.iloc[-1]
                    long_current = long_ma.iloc[-1]
                    
                    if short_current > long_current:
                        signals['buy_signals'].append('MA bullish (short > long)')
                    else:
                        signals['sell_signals'].append('MA bearish (short < long)')
                        
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to generate trading signals: {e}")
            return {'buy_signals': [], 'sell_signals': [], 'neutral_signals': []}

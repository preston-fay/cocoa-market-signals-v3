"""
Zen Consensus Service
Runs the consensus orchestration and stores predictions/signals in database
100% REAL predictions - NO FAKE DATA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import logging
from typing import Dict, List, Optional, Any
from sqlmodel import Session, select
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.database import engine
from app.models.price_data import PriceData
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.model_performance import ModelPerformance
from src.models.simple_zen_consensus import SimpleZenConsensus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZenConsensusService:
    """
    Service to run Zen Consensus and store all results
    """
    
    def __init__(self):
        self.consensus = SimpleZenConsensus()
        self.model_version = "1.0"
        
    def get_recent_price_data(self, days: int = 100) -> pd.DataFrame:
        """
        Get recent price data from database
        """
        with Session(engine) as session:
            prices = session.exec(
                select(PriceData)
                .where(PriceData.source == "Yahoo Finance")
                .order_by(PriceData.date.desc())
                .limit(days)
            ).all()
            
            if not prices:
                raise ValueError("No price data found in database")
            
            # Convert to DataFrame (reverse for chronological order)
            prices = list(reversed(prices))
            df = pd.DataFrame([
                {'date': p.date, 'price': p.price}
                for p in prices
            ])
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"Loaded {len(df)} days of price data")
            return df
    
    def run_daily_consensus(self) -> Dict[str, Any]:
        """
        Run consensus and store all predictions/signals
        """
        logger.info("Running daily Zen Consensus...")
        
        # Get data
        df = self.get_recent_price_data()
        
        # Run consensus
        consensus_result = self.consensus.run_consensus(df)
        
        # Store predictions
        predictions_stored = self._store_predictions(consensus_result, df)
        
        # Generate and store signals
        signals_stored = self._store_signals(consensus_result, df)
        
        # Log summary
        logger.info(f"Consensus complete:")
        logger.info(f"  Price: ${consensus_result['consensus_forecast']:,.0f}")
        logger.info(f"  Signal: {consensus_result['consensus_signal']}")
        logger.info(f"  Confidence: {consensus_result['confidence_score']:.1%}")
        logger.info(f"  Predictions stored: {predictions_stored}")
        logger.info(f"  Signals stored: {signals_stored}")
        
        return {
            'consensus': consensus_result,
            'predictions_stored': predictions_stored,
            'signals_stored': signals_stored,
            'timestamp': datetime.now()
        }
    
    def _store_predictions(self, consensus: Dict, df: pd.DataFrame) -> int:
        """
        Store consensus predictions for multiple horizons
        """
        stored = 0
        current_price = df['price'].iloc[-1]
        
        with Session(engine) as session:
            # Store predictions for different horizons
            horizons = [1, 7, 30]
            
            for horizon in horizons:
                # Adjust prediction based on horizon
                if 'price_change_rate' in consensus:
                    daily_change = consensus['price_change_rate'] / 7
                    predicted_price = current_price * (1 + daily_change * horizon)
                else:
                    predicted_price = consensus['consensus_forecast']
                
                # Confidence decreases with horizon
                confidence = consensus['confidence_score'] * (1 - horizon * 0.01)
                
                prediction = Prediction(
                    model_name="zen_consensus",
                    target_date=date.today() + timedelta(days=horizon),
                    prediction_horizon=horizon,
                    predicted_price=predicted_price,
                    confidence_score=confidence,
                    prediction_type="point",
                    current_price=current_price,
                    model_version=self.model_version,
                    features_used=json.dumps({
                        'models': list(consensus['role_contributions'].keys()),
                        'total_models': consensus['total_models']
                    }),
                    market_regime=self._detect_market_regime(df)
                )
                
                # Add prediction range if available
                if 'prediction_range' in consensus:
                    prediction.predicted_low = consensus['prediction_range']['min']
                    prediction.predicted_high = consensus['prediction_range']['max']
                
                session.add(prediction)
                stored += 1
            
            session.commit()
        
        return stored
    
    def _store_signals(self, consensus: Dict, df: pd.DataFrame) -> int:
        """
        Generate and store trading signals
        """
        stored = 0
        current_price = df['price'].iloc[-1]
        
        with Session(engine) as session:
            # Main consensus signal
            signal_strength = abs(consensus['price_change_rate'])
            
            main_signal = Signal(
                signal_date=datetime.combine(date.today(), datetime.min.time()),
                signal_type='consensus',
                signal_name=f"zen_{consensus['consensus_signal']}",
                signal_direction='bullish' if 'buy' in consensus['consensus_signal'] else 'bearish' if 'sell' in consensus['consensus_signal'] else 'neutral',
                signal_strength=float(signal_strength * 10),  # Scale to -10 to +10
                signal_value=float(consensus['consensus_forecast']),
                description=f"Zen Consensus: {consensus['consensus_signal'].upper()} signal ({consensus['price_change_rate']*100:+.1f}% expected move)",
                source='zen_consensus',
                detector='simple_zen_consensus',
                confidence=consensus['confidence_score']
            )
            session.add(main_signal)
            stored += 1
            
            # Role-specific signals
            for role_name, role_data in consensus['role_contributions'].items():
                if role_data.get('predictions'):
                    role_avg = np.mean(list(role_data['predictions'].values()))
                    role_change = (role_avg - current_price) / current_price
                    
                    if abs(role_change) > 0.01:  # Only significant signals
                        role_signal = Signal(
                            signal_date=datetime.combine(date.today(), datetime.min.time()),
                            signal_type='role',
                            signal_name=f"{role_name}_signal",
                            signal_direction='bullish' if role_change > 0 else 'bearish',
                            signal_strength=float(role_change * 10),  # Scale to -10 to +10
                            signal_value=float(role_avg),
                            description=f"{role_name}: {role_data['stance']} - expecting {role_change*100:+.1f}% move",
                            source=role_name,
                            detector='simple_zen_consensus',
                            confidence=role_data.get('weight', 0.3)
                        )
                        session.add(role_signal)
                        stored += 1
            
            # Market context signals
            market_context = self.consensus.get_market_context(df)
            if market_context['volatility_regime'] == 'high':
                vol_signal = Signal(
                    signal_date=datetime.combine(date.today(), datetime.min.time()),
                    signal_type='volatility',
                    signal_name='high_volatility_alert',
                    signal_direction='neutral',
                    signal_strength=float(min(10, market_context['current_volatility'] * 10)),  # Cap at 10
                    signal_value=float(market_context['current_volatility'] * 100),
                    description=f"High volatility alert: {market_context['current_volatility']*100:.0f}% annualized",
                    source='market_analysis',
                    detector='volatility_monitor',
                    confidence=0.9
                )
                session.add(vol_signal)
                stored += 1
            
            session.commit()
        
        return stored
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime
        """
        returns = df['price'].pct_change().dropna()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        # Trend
        sma20 = df['price'].rolling(20).mean().iloc[-1]
        sma50 = df['price'].rolling(50).mean().iloc[-1]
        current_price = df['price'].iloc[-1]
        
        if current_price > sma20 > sma50:
            trend = "uptrend"
        elif current_price < sma20 < sma50:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        # Volatility
        current_vol = volatility.iloc[-1]
        avg_vol = volatility.mean()
        
        if current_vol > avg_vol * 1.5:
            vol_regime = "high_vol"
        elif current_vol < avg_vol * 0.7:
            vol_regime = "low_vol"
        else:
            vol_regime = "normal_vol"
        
        return f"{trend}_{vol_regime}"
    
    def evaluate_past_predictions(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Evaluate accuracy of past predictions
        """
        logger.info(f"Evaluating predictions from last {days_back} days...")
        
        with Session(engine) as session:
            # Get predictions that should have materialized
            cutoff_date = date.today() - timedelta(days=days_back)
            
            predictions = session.exec(
                select(Prediction)
                .where(Prediction.model_name == "zen_consensus")
                .where(Prediction.target_date <= date.today())
                .where(Prediction.created_at >= datetime.combine(cutoff_date, datetime.min.time()))
                .where(Prediction.actual_price.is_(None))
            ).all()
            
            updated = 0
            errors = []
            
            for pred in predictions:
                # Get actual price for target date
                actual = session.exec(
                    select(PriceData)
                    .where(PriceData.date == pred.target_date)
                    .where(PriceData.source == "Yahoo Finance")
                ).first()
                
                if actual:
                    pred.actual_price = actual.price
                    pred.error = actual.price - pred.predicted_price
                    pred.error_percentage = abs(pred.error) / actual.price * 100
                    
                    errors.append({
                        'horizon': pred.prediction_horizon,
                        'error_pct': pred.error_percentage,
                        'direction_correct': (pred.predicted_price > pred.current_price) == (actual.price > pred.current_price)
                    })
                    
                    updated += 1
            
            session.commit()
            
            # Calculate performance metrics
            if errors:
                by_horizon = {}
                for h in [1, 7, 30]:
                    horizon_errors = [e for e in errors if e['horizon'] == h]
                    if horizon_errors:
                        by_horizon[h] = {
                            'mape': np.mean([e['error_pct'] for e in horizon_errors]),
                            'directional_accuracy': np.mean([e['direction_correct'] for e in horizon_errors]),
                            'count': len(horizon_errors)
                        }
                
                # Store overall performance
                overall_mape = np.mean([e['error_pct'] for e in errors])
                overall_direction = np.mean([e['direction_correct'] for e in errors])
                
                perf = ModelPerformance(
                    model_name="zen_consensus",
                    evaluation_date=date.today(),
                    period_start=cutoff_date,
                    period_end=date.today(),
                    period_days=days_back,
                    metric_name="overall_accuracy",
                    metric_value=overall_mape,
                    sample_size=len(errors),
                    metadata=json.dumps({
                        'directional_accuracy': float(overall_direction),
                        'by_horizon': by_horizon,
                        'model_version': self.model_version
                    })
                )
                session.add(perf)
                session.commit()
                
                logger.info(f"Evaluation complete:")
                logger.info(f"  Predictions evaluated: {updated}")
                logger.info(f"  Overall MAPE: {overall_mape:.1f}%")
                logger.info(f"  Directional accuracy: {overall_direction:.1%}")
                
                return {
                    'evaluated': updated,
                    'overall_mape': overall_mape,
                    'directional_accuracy': overall_direction,
                    'by_horizon': by_horizon
                }
            
        return {'evaluated': 0, 'message': 'No predictions to evaluate'}
    
    def get_current_signals(self) -> List[Dict[str, Any]]:
        """
        Get today's signals
        """
        with Session(engine) as session:
            signals = session.exec(
                select(Signal)
                .where(Signal.signal_date == date.today())
                .where(Signal.source.in_(['zen_consensus', 'trend_follower', 'mean_reverter', 'momentum_trader']))
                .order_by(Signal.confidence.desc())
            ).all()
            
            return [
                {
                    'type': s.signal_type,
                    'direction': s.signal_direction,
                    'strength': s.signal_strength,
                    'confidence': s.confidence,
                    'description': s.description
                }
                for s in signals
            ]

def run_zen_consensus():
    """
    Main function to run Zen Consensus
    """
    service = ZenConsensusService()
    
    try:
        # Run consensus
        result = service.run_daily_consensus()
        
        # Evaluate past predictions
        evaluation = service.evaluate_past_predictions(days_back=7)
        
        # Get current signals
        signals = service.get_current_signals()
        
        print("\n" + "="*60)
        print("ZEN CONSENSUS DAILY RUN COMPLETE")
        print("="*60)
        
        consensus = result['consensus']
        print(f"\nConsensus Forecast:")
        print(f"  Current Price: ${consensus['current_price']:,.0f}")
        print(f"  7-Day Target: ${consensus['consensus_forecast']:,.0f}")
        print(f"  Expected Move: {consensus['price_change_rate']*100:+.1f}%")
        print(f"  Signal: {consensus['consensus_signal'].upper()}")
        print(f"  Confidence: {consensus['confidence_score']:.1%}")
        
        if evaluation.get('evaluated', 0) > 0:
            print(f"\nModel Performance:")
            print(f"  Recent MAPE: {evaluation['overall_mape']:.1f}%")
            print(f"  Directional Accuracy: {evaluation['directional_accuracy']:.1%}")
        
        print(f"\nActive Signals ({len(signals)}):")
        for signal in signals[:3]:  # Top 3 signals
            print(f"  - {signal['description']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running consensus: {str(e)}")
        raise

if __name__ == "__main__":
    run_zen_consensus()
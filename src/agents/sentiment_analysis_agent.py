#!/usr/bin/env python3
"""
Sentiment Analysis Agent with Zen Consensus Integration
Orchestrates NLP analysis with other data sources
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import numpy as np
from sqlmodel import Session, select
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.core.database import engine
from app.models.news_article import NewsArticle
from src.nlp.sentiment_analysis_engine import CocoaSentimentAnalyzer
from src.services.zen_consensus_service import ZenConsensusService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisAgent:
    """
    Agent that performs sentiment analysis and integrates with Zen Consensus
    for comprehensive market signal generation
    """
    
    def __init__(self):
        self.sentiment_analyzer = CocoaSentimentAnalyzer()
        self.zen_consensus = ZenConsensusService()
        self.analysis_results = []
        
    async def analyze_and_orchestrate(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze sentiment and orchestrate with other data sources
        using Zen Consensus
        """
        logger.info(f"🧠 Starting sentiment analysis for past {days_back} days")
        
        # 1. Process unanalyzed articles
        self.sentiment_analyzer.process_all_articles()
        
        # 2. Get sentiment summary
        sentiment_summary = self.sentiment_analyzer.get_market_sentiment_summary(days_back)
        
        # 3. Get analyzed articles for detailed processing
        with Session(engine) as session:
            analyzed_articles = session.exec(
                select(NewsArticle)
                .where(NewsArticle.sentiment_score.is_not(None))
                .where(NewsArticle.published_date >= datetime.now() - timedelta(days=days_back))
                .order_by(NewsArticle.published_date.desc())
            ).all()
        
        # 4. Extract temporal sentiment patterns
        temporal_patterns = self._extract_temporal_patterns(analyzed_articles)
        
        # 5. Identify key events and their impact
        key_events = self._identify_key_events(analyzed_articles)
        
        # 6. Generate sentiment-based features for Zen Consensus
        sentiment_features = self._generate_sentiment_features(
            sentiment_summary, 
            temporal_patterns, 
            key_events
        )
        
        # 7. Integrate with Zen Consensus
        zen_prediction = await self._integrate_with_zen_consensus(sentiment_features)
        
        # 8. Generate comprehensive analysis
        analysis = {
            'timestamp': datetime.now(),
            'period_analyzed': days_back,
            'sentiment_summary': sentiment_summary,
            'temporal_patterns': temporal_patterns,
            'key_events': key_events,
            'sentiment_features': sentiment_features,
            'zen_consensus_integration': zen_prediction,
            'recommendations': self._generate_recommendations(
                sentiment_summary, 
                temporal_patterns, 
                zen_prediction
            )
        }
        
        return analysis
    
    def _extract_temporal_patterns(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Extract temporal patterns from sentiment scores"""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        # Group by date
        daily_sentiments = {}
        for article in articles:
            date_key = article.published_date.date()
            if date_key not in daily_sentiments:
                daily_sentiments[date_key] = []
            daily_sentiments[date_key].append(article.sentiment_score)
        
        # Calculate daily averages
        daily_averages = {
            str(date): np.mean(scores) 
            for date, scores in daily_sentiments.items()
        }
        
        # Calculate volatility
        all_scores = [a.sentiment_score for a in articles]
        volatility = np.std(all_scores) if all_scores else 0
        
        # Detect trend changes
        trend_changes = self._detect_trend_changes(daily_averages)
        
        return {
            'daily_averages': daily_averages,
            'volatility': volatility,
            'trend_changes': trend_changes,
            'momentum': self._calculate_momentum(daily_averages)
        }
    
    def _detect_trend_changes(self, daily_averages: Dict[str, float]) -> List[Dict]:
        """Detect significant trend changes in sentiment"""
        changes = []
        dates = sorted(daily_averages.keys())
        
        if len(dates) < 3:
            return changes
        
        for i in range(2, len(dates)):
            prev_avg = daily_averages[dates[i-2]]
            curr_avg = daily_averages[dates[i]]
            
            # Significant change threshold
            if abs(curr_avg - prev_avg) > 0.3:
                changes.append({
                    'date': dates[i],
                    'change': curr_avg - prev_avg,
                    'direction': 'positive' if curr_avg > prev_avg else 'negative',
                    'magnitude': abs(curr_avg - prev_avg)
                })
        
        return changes
    
    def _calculate_momentum(self, daily_averages: Dict[str, float]) -> float:
        """Calculate sentiment momentum"""
        if len(daily_averages) < 2:
            return 0.0
        
        dates = sorted(daily_averages.keys())
        recent = dates[-7:] if len(dates) >= 7 else dates
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate slope of recent sentiment
        x = list(range(len(recent)))
        y = [daily_averages[d] for d in recent]
        
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _identify_key_events(self, articles: List[NewsArticle]) -> List[Dict]:
        """Identify key events from articles"""
        key_events = []
        
        # High impact articles (extreme sentiment)
        for article in articles:
            if abs(article.sentiment_score) > 0.7:
                event = {
                    'date': article.published_date,
                    'title': article.title,
                    'sentiment_score': article.sentiment_score,
                    'impact': 'high' if abs(article.sentiment_score) > 0.8 else 'medium',
                    'topics': article.topics.split(',') if article.topics else [],
                    'market_impact': article.market_impact or 'unknown'
                }
                key_events.append(event)
        
        # Sort by date
        key_events.sort(key=lambda x: x['date'], reverse=True)
        
        return key_events[:10]  # Top 10 events
    
    def _generate_sentiment_features(self, summary: Dict, patterns: Dict, events: List[Dict]) -> Dict:
        """Generate features for Zen Consensus integration"""
        features = {
            # Overall sentiment
            'sentiment_mean': summary.get('average_sentiment', 0),
            'sentiment_std': summary.get('sentiment_std', 0),
            'sentiment_trend': summary.get('sentiment_trend', 'stable'),
            
            # Temporal features
            'sentiment_momentum': patterns.get('momentum', 0),
            'sentiment_volatility': patterns.get('volatility', 0),
            'trend_change_count': len(patterns.get('trend_changes', [])),
            
            # Event features
            'high_impact_event_count': len([e for e in events if e['impact'] == 'high']),
            'negative_event_ratio': len([e for e in events if e['sentiment_score'] < 0]) / max(len(events), 1),
            
            # Topic distribution
            'dominant_topics': summary.get('dominant_topics', []),
            
            # Market signals
            'bullish_signal_strength': self._calculate_signal_strength(summary, 'bullish'),
            'bearish_signal_strength': self._calculate_signal_strength(summary, 'bearish')
        }
        
        return features
    
    def _calculate_signal_strength(self, summary: Dict, signal_type: str) -> float:
        """Calculate strength of bullish/bearish signals"""
        avg_sentiment = summary.get('average_sentiment', 0)
        outlook = summary.get('market_outlook', 'neutral')
        
        if signal_type == 'bullish':
            if outlook == 'bullish':
                return min(1.0, 0.5 + avg_sentiment)
            else:
                return max(0.0, avg_sentiment)
        else:  # bearish
            if outlook == 'bearish':
                return min(1.0, 0.5 - avg_sentiment)
            else:
                return max(0.0, -avg_sentiment)
    
    async def _integrate_with_zen_consensus(self, sentiment_features: Dict) -> Dict:
        """Integrate sentiment features with Zen Consensus"""
        try:
            # Run Zen Consensus
            zen_result = self.zen_consensus.run_daily_consensus()
            
            # Extract relevant data
            consensus = zen_result.get('consensus', {})
            
            # Combine with sentiment features for comprehensive prediction
            integrated_result = {
                'consensus_prediction': consensus.get('price_change_rate', 0),
                'consensus_signal': consensus.get('consensus_signal', 'neutral'),
                'confidence': consensus.get('confidence_score', 0),
                'price_forecast': consensus.get('consensus_forecast', 0),
                'sentiment_adjusted_confidence': min(
                    consensus.get('confidence_score', 0.5),
                    sentiment_features.get('sentiment_volatility', 1) * 0.8
                ),
                'integrated_features': {
                    'zen_consensus': consensus,
                    'sentiment': sentiment_features
                }
            }
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error integrating with Zen Consensus: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, summary: Dict, patterns: Dict, zen_prediction: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on sentiment trend
        trend = summary.get('sentiment_trend', 'stable')
        if trend == 'improving':
            recommendations.append("Sentiment is improving - consider increasing position sizes")
        elif trend == 'deteriorating':
            recommendations.append("Sentiment is deteriorating - consider reducing exposure")
        
        # Based on volatility
        volatility = patterns.get('volatility', 0)
        if volatility > 0.5:
            recommendations.append("High sentiment volatility detected - implement tighter risk controls")
        
        # Based on Zen Consensus
        if 'consensus_prediction' in zen_prediction:
            prediction = zen_prediction['consensus_prediction']
            confidence = zen_prediction.get('confidence', 0)
            
            if confidence > 0.7:
                if prediction > 0:
                    recommendations.append(f"Strong bullish consensus (confidence: {confidence:.2f})")
                else:
                    recommendations.append(f"Strong bearish consensus (confidence: {confidence:.2f})")
        
        # Based on key events
        if summary.get('dominant_topics'):
            topics = summary['dominant_topics']
            if 'weather' in str(topics):
                recommendations.append("Weather-related events detected - monitor supply impact")
            if 'policy' in str(topics):
                recommendations.append("Policy changes detected - assess regulatory impact")
        
        return recommendations

async def run_sentiment_agent():
    """Run the sentiment analysis agent"""
    print("🚀 Initializing Sentiment Analysis Agent with Zen Consensus...")
    
    agent = SentimentAnalysisAgent()
    
    # Run analysis for different time periods
    periods = [7, 30, 90]
    
    for days in periods:
        print(f"\n📊 Analyzing {days}-day period...")
        
        analysis = await agent.analyze_and_orchestrate(days_back=days)
        
        # Display results
        print(f"\n=== {days}-Day Analysis Results ===")
        print(f"Average Sentiment: {analysis['sentiment_summary'].get('average_sentiment', 0):.3f}")
        print(f"Sentiment Trend: {analysis['sentiment_summary'].get('sentiment_trend', 'unknown')}")
        print(f"Articles Analyzed: {analysis['sentiment_summary'].get('articles_analyzed', 0)}")
        
        if 'temporal_patterns' in analysis:
            print(f"\nSentiment Momentum: {analysis['temporal_patterns'].get('momentum', 0):.3f}")
            print(f"Volatility: {analysis['temporal_patterns'].get('volatility', 0):.3f}")
        
        if 'key_events' in analysis and analysis['key_events']:
            print(f"\n🔥 Key Events ({len(analysis['key_events'])} found):")
            for event in analysis['key_events'][:3]:
                print(f"  - {event['date'].strftime('%Y-%m-%d')}: {event['title'][:60]}...")
                print(f"    Sentiment: {event['sentiment_score']:.2f}, Impact: {event['impact']}")
        
        if 'zen_consensus_integration' in analysis:
            zen = analysis['zen_consensus_integration']
            if 'consensus_prediction' in zen:
                print(f"\n🎯 Zen Consensus Prediction: {zen['consensus_prediction']:.3f}")
                print(f"   Confidence: {zen.get('confidence', 0):.2f}")
        
        if 'recommendations' in analysis:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
    
    return True

if __name__ == "__main__":
    asyncio.run(run_sentiment_agent())
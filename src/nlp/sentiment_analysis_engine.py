#!/usr/bin/env python3
"""
NLP Sentiment Analysis Engine for Cocoa Market Intelligence
Analyzes news articles and unstructured text to extract market sentiment
Designed specifically for commodity/financial text analysis
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from sqlmodel import Session, select
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.core.database import engine
from app.models.news_article import NewsArticle
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CocoaSentimentAnalyzer:
    """
    Specialized sentiment analysis for cocoa market news
    Combines multiple NLP techniques for accurate financial sentiment
    """
    
    def __init__(self):
        # Initialize NLP models
        try:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("Spacy model not found. Install with: pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
                self.nlp = None
            
        self.vader = SentimentIntensityAnalyzer()
        
        # Cocoa-specific sentiment modifiers
        self.positive_terms = {
            # Production
            'bumper harvest': 2.0,
            'record production': 2.0,
            'increased yield': 1.5,
            'good weather': 1.5,
            'favorable conditions': 1.5,
            'strong demand': 1.8,
            'supply shortage': 1.5,  # Positive for prices
            
            # Market
            'price rally': 2.0,
            'bullish': 1.8,
            'uptrend': 1.5,
            'support': 1.2,
            'breakout': 1.5,
            'recovery': 1.3,
            
            # Industry
            'sustainability': 1.0,
            'premium': 1.2,
            'quality improvement': 1.1
        }
        
        self.negative_terms = {
            # Production issues
            'drought': -2.0,
            'flood': -2.0,
            'black pod': -2.5,
            'disease outbreak': -2.5,
            'poor harvest': -2.0,
            'crop failure': -3.0,
            'swollen shoot': -2.5,
            
            # Market
            'bearish': -1.8,
            'downtrend': -1.5,
            'resistance': -1.2,
            'breakdown': -1.5,
            'oversupply': -1.5,
            'weak demand': -1.8,
            
            # Geopolitical
            'political unrest': -2.0,
            'strike': -1.8,
            'export ban': -2.5,
            'child labor': -1.5
        }
        
        # Context modifiers
        self.intensifiers = {
            'very': 1.5,
            'extremely': 2.0,
            'significantly': 1.8,
            'slightly': 0.5,
            'somewhat': 0.7,
            'massive': 2.0,
            'severe': 2.0
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'neither', 'none', 'nobody', 
                         'nothing', 'nowhere', 'isn\'t', 'aren\'t', 'wasn\'t',
                         'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t'}
    
    def preprocess_text(self, text: str) -> str:
        """Clean and prepare text for analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep numbers (important for prices, quantities)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\%]', '', text)
        
        return text.strip()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities relevant to cocoa markets"""
        entities = {
            'countries': [],
            'organizations': [],
            'dates': [],
            'money': [],
            'quantities': []
        }
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Countries, cities
                entities['countries'].append(ent.text)
            elif ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities['dates'].append(ent.text)
            elif ent.label_ == "MONEY":
                entities['money'].append(ent.text)
            elif ent.label_ in ["QUANTITY", "PERCENT"]:
                entities['quantities'].append(ent.text)
        
        return entities
    
    def calculate_domain_sentiment(self, text: str) -> float:
        """Calculate sentiment using domain-specific terms"""
        text_lower = text.lower()
        score = 0.0
        
        # Check for positive terms
        for term, weight in self.positive_terms.items():
            if term in text_lower:
                # Check for negation
                term_pos = text_lower.find(term)
                preceding_text = text_lower[max(0, term_pos-50):term_pos]
                
                if any(neg in preceding_text.split() for neg in self.negations):
                    score -= weight  # Reverse sentiment
                else:
                    score += weight
        
        # Check for negative terms
        for term, weight in self.negative_terms.items():
            if term in text_lower:
                term_pos = text_lower.find(term)
                preceding_text = text_lower[max(0, term_pos-50):term_pos]
                
                if any(neg in preceding_text.split() for neg in self.negations):
                    score -= weight  # Reverse sentiment (double negative = positive)
                else:
                    score += weight
        
        # Apply intensifiers
        for intensifier, multiplier in self.intensifiers.items():
            count = text_lower.count(intensifier)
            if count > 0 and abs(score) > 0:
                score *= (1 + (multiplier - 1) * 0.5)  # Moderate the intensifier effect
        
        # Normalize to [-1, 1]
        return np.tanh(score / 10)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Comprehensive sentiment analysis combining multiple methods
        Returns scores from different analyzers and a weighted consensus
        """
        clean_text = self.preprocess_text(text)
        
        results = {
            'domain_sentiment': 0.0,
            'vader_sentiment': 0.0,
            'textblob_sentiment': 0.0,
            'textblob_subjectivity': 0.0,
            'consensus_sentiment': 0.0,
            'confidence': 0.0
        }
        
        # 1. Domain-specific sentiment
        results['domain_sentiment'] = self.calculate_domain_sentiment(clean_text)
        
        # 2. VADER sentiment (good for social media and news)
        vader_scores = self.vader.polarity_scores(clean_text)
        results['vader_sentiment'] = vader_scores['compound']
        
        # 3. TextBlob sentiment
        try:
            blob = TextBlob(clean_text)
            results['textblob_sentiment'] = blob.sentiment.polarity
            results['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            pass
        
        # 4. Calculate weighted consensus
        # Give more weight to domain-specific analysis for commodity news
        weights = {
            'domain': 0.5,
            'vader': 0.3,
            'textblob': 0.2
        }
        
        consensus = (
            results['domain_sentiment'] * weights['domain'] +
            results['vader_sentiment'] * weights['vader'] +
            results['textblob_sentiment'] * weights['textblob']
        )
        
        results['consensus_sentiment'] = consensus
        
        # 5. Calculate confidence based on agreement
        sentiments = [results['domain_sentiment'], results['vader_sentiment'], results['textblob_sentiment']]
        std_dev = np.std(sentiments)
        results['confidence'] = max(0, 1 - std_dev)
        
        return results
    
    def extract_market_signals(self, text: str, entities: Dict) -> Dict[str, Any]:
        """Extract specific market signals from text"""
        signals = {
            'price_direction': None,
            'supply_impact': None,
            'demand_impact': None,
            'weather_impact': None,
            'time_horizon': None,
            'affected_regions': []
        }
        
        text_lower = text.lower()
        
        # Price direction signals
        if any(word in text_lower for word in ['rally', 'surge', 'jump', 'soar', 'climb']):
            signals['price_direction'] = 'bullish'
        elif any(word in text_lower for word in ['fall', 'drop', 'plunge', 'decline', 'slump']):
            signals['price_direction'] = 'bearish'
        
        # Supply signals
        if any(word in text_lower for word in ['shortage', 'deficit', 'tight supply']):
            signals['supply_impact'] = 'negative'
        elif any(word in text_lower for word in ['surplus', 'oversupply', 'abundant']):
            signals['supply_impact'] = 'positive'
        
        # Demand signals
        if any(word in text_lower for word in ['strong demand', 'increased consumption']):
            signals['demand_impact'] = 'positive'
        elif any(word in text_lower for word in ['weak demand', 'reduced consumption']):
            signals['demand_impact'] = 'negative'
        
        # Weather impact
        if any(word in text_lower for word in ['drought', 'dry', 'flood', 'excessive rain']):
            signals['weather_impact'] = 'negative'
        elif any(word in text_lower for word in ['favorable weather', 'good rainfall', 'ideal conditions']):
            signals['weather_impact'] = 'positive'
        
        # Time horizon
        if any(word in text_lower for word in ['immediate', 'today', 'this week']):
            signals['time_horizon'] = 'short_term'
        elif any(word in text_lower for word in ['this month', 'coming weeks']):
            signals['time_horizon'] = 'medium_term'
        elif any(word in text_lower for word in ['this year', 'next season', 'long term']):
            signals['time_horizon'] = 'long_term'
        
        # Affected regions (from entities)
        cocoa_regions = ['ghana', 'ivory coast', 'nigeria', 'cameroon', 'ecuador', 'brazil']
        for country in entities.get('countries', []):
            if country.lower() in cocoa_regions:
                signals['affected_regions'].append(country)
        
        return signals
    
    def analyze_article(self, article: NewsArticle) -> Dict[str, Any]:
        """Complete analysis of a news article"""
        # Combine title and content for analysis
        full_text = f"{article.title} {article.content}"
        
        # Extract entities
        entities = self.extract_entities(full_text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(full_text)
        
        # Extract market signals
        signals = self.extract_market_signals(full_text, entities)
        
        # Compile results
        analysis = {
            'article_id': article.id,
            'published_date': article.published_date,
            'sentiment_score': sentiment['consensus_sentiment'],
            'sentiment_confidence': sentiment['confidence'],
            'sentiment_components': sentiment,
            'entities': entities,
            'market_signals': signals,
            'topics': self.extract_topics(full_text),
            'analyzed_at': datetime.now()
        }
        
        return analysis
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'production': ['harvest', 'production', 'yield', 'crop'],
            'weather': ['weather', 'rain', 'drought', 'climate'],
            'disease': ['disease', 'black pod', 'swollen shoot', 'pest'],
            'market': ['price', 'futures', 'trading', 'market'],
            'supply_chain': ['export', 'shipping', 'logistics', 'port'],
            'sustainability': ['sustainable', 'certification', 'fairtrade'],
            'processing': ['grinding', 'processing', 'chocolate'],
            'policy': ['government', 'regulation', 'policy', 'tax']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def process_all_articles(self):
        """Process all unanalyzed articles in database"""
        with Session(engine) as session:
            # Get unprocessed articles
            unprocessed = session.exec(
                select(NewsArticle)
                .where(NewsArticle.processed == False)
                .limit(1000)  # Process larger batches
            ).all()
            
            logger.info(f"Processing {len(unprocessed)} articles...")
            
            for article in unprocessed:
                try:
                    # Analyze article
                    analysis = self.analyze_article(article)
                    
                    # Update article with sentiment
                    article.sentiment_score = analysis['sentiment_score']
                    article.sentiment_label = self.get_sentiment_label(analysis['sentiment_score'])
                    article.topics = ','.join(analysis['topics'])
                    article.market_impact = analysis['market_signals'].get('price_direction', '')
                    article.processed = True
                    article.processing_notes = f"Analyzed on {datetime.now().isoformat()}"
                    
                    # Store detailed analysis (could be separate table)
                    if analysis['entities']['countries']:
                        article.mentioned_countries = ','.join(analysis['entities']['countries'][:5])
                    if analysis['entities']['organizations']:
                        article.mentioned_companies = ','.join(analysis['entities']['organizations'][:5])
                    
                    session.add(article)
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.id}: {str(e)}")
                    article.processed = True
                    article.processing_notes = f"Error: {str(e)}"
                    session.add(article)
            
            session.commit()
            logger.info("Processing complete")
    
    def get_sentiment_label(self, score: float) -> str:
        """Convert numeric sentiment to label"""
        if score >= 0.6:
            return "very_positive"
        elif score >= 0.2:
            return "positive"
        elif score >= -0.2:
            return "neutral"
        elif score >= -0.6:
            return "negative"
        else:
            return "very_negative"
    
    def get_market_sentiment_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get sentiment summary for recent articles"""
        with Session(engine) as session:
            # Get recent analyzed articles
            recent_articles = session.exec(
                select(NewsArticle)
                .where(NewsArticle.sentiment_score.is_not(None))
                .where(NewsArticle.published_date >= datetime.now() - timedelta(days=days_back))
            ).all()
            
            if not recent_articles:
                return {"error": "No analyzed articles found"}
            
            sentiments = [a.sentiment_score for a in recent_articles]
            
            summary = {
                'period_days': days_back,
                'articles_analyzed': len(recent_articles),
                'average_sentiment': np.mean(sentiments),
                'sentiment_std': np.std(sentiments),
                'sentiment_trend': self.calculate_trend(recent_articles),
                'dominant_topics': self.get_dominant_topics(recent_articles),
                'key_entities': self.get_key_entities(recent_articles),
                'market_outlook': self.determine_outlook(sentiments)
            }
            
            return summary
    
    def calculate_trend(self, articles: List[NewsArticle]) -> str:
        """Calculate sentiment trend"""
        if len(articles) < 2:
            return "insufficient_data"
        
        # Sort by date
        sorted_articles = sorted(articles, key=lambda x: x.published_date)
        
        # Split into halves
        mid = len(sorted_articles) // 2
        first_half = [a.sentiment_score for a in sorted_articles[:mid]]
        second_half = [a.sentiment_score for a in sorted_articles[mid:]]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if second_avg > first_avg + 0.1:
            return "improving"
        elif second_avg < first_avg - 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def get_dominant_topics(self, articles: List[NewsArticle]) -> List[Tuple[str, int]]:
        """Get most common topics"""
        all_topics = []
        for article in articles:
            if article.topics:
                all_topics.extend(article.topics.split(','))
        
        return Counter(all_topics).most_common(5)
    
    def get_key_entities(self, articles: List[NewsArticle]) -> Dict[str, List[str]]:
        """Get most mentioned entities"""
        countries = []
        companies = []
        
        for article in articles:
            if article.mentioned_countries:
                countries.extend(article.mentioned_countries.split(','))
            if article.mentioned_companies:
                companies.extend(article.mentioned_companies.split(','))
        
        return {
            'countries': [c[0] for c in Counter(countries).most_common(3)],
            'companies': [c[0] for c in Counter(companies).most_common(3)]
        }
    
    def determine_outlook(self, sentiments: List[float]) -> str:
        """Determine market outlook based on sentiment"""
        avg = np.mean(sentiments)
        
        if avg >= 0.3:
            return "bullish"
        elif avg <= -0.3:
            return "bearish"
        else:
            return "neutral"


def main():
    """Run sentiment analysis on all articles"""
    print("🧠 Initializing Cocoa Sentiment Analysis Engine...")
    
    analyzer = CocoaSentimentAnalyzer()
    
    print("📰 Processing unanalyzed articles...")
    analyzer.process_all_articles()
    
    print("\n📊 Market Sentiment Summary (Last 30 days):")
    summary = analyzer.get_market_sentiment_summary(days_back=30)
    
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
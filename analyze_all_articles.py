#!/usr/bin/env python3
"""
Analyze ALL articles with sentiment analysis
Shows real progress and results
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import Session, select, func
from app.core.database import engine
from app.models.news_article import NewsArticle
from src.nlp.sentiment_analysis_engine import CocoaSentimentAnalyzer
import numpy as np
from datetime import datetime

def analyze_all_articles():
    """Process all articles with progress tracking"""
    
    analyzer = CocoaSentimentAnalyzer()
    
    with Session(engine) as session:
        # Get total unprocessed
        total_unprocessed = session.scalar(
            select(func.count(NewsArticle.id))
            .where(NewsArticle.processed == False)
        )
        
        print(f"📰 Found {total_unprocessed:,} unprocessed articles")
        print("🧠 Starting sentiment analysis...\n")
        
        batch_size = 500
        processed_total = 0
        
        while True:
            # Get batch of unprocessed articles
            unprocessed = session.exec(
                select(NewsArticle)
                .where(NewsArticle.processed == False)
                .limit(batch_size)
            ).all()
            
            if not unprocessed:
                break
            
            print(f"Processing batch of {len(unprocessed)} articles...")
            
            for i, article in enumerate(unprocessed):
                try:
                    # Analyze article
                    analysis = analyzer.analyze_article(article)
                    
                    # Update article with sentiment
                    article.sentiment_score = analysis['sentiment_score']
                    article.sentiment_label = analyzer.get_sentiment_label(analysis['sentiment_score'])
                    article.topics = ','.join(analysis['topics'])
                    article.market_impact = analysis['market_signals'].get('price_direction', '')
                    article.processed = True
                    article.processing_notes = f"Analyzed on {datetime.now().isoformat()}"
                    
                    # Store detailed analysis
                    if analysis['entities']['countries']:
                        article.mentioned_countries = ','.join(analysis['entities']['countries'][:5])
                    if analysis['entities']['organizations']:
                        article.mentioned_companies = ','.join(analysis['entities']['organizations'][:5])
                    
                    session.add(article)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(unprocessed)} in current batch")
                        session.commit()  # Commit periodically
                    
                except Exception as e:
                    print(f"  Error processing article {article.id}: {str(e)}")
                    article.processed = True
                    article.processing_notes = f"Error: {str(e)}"
                    session.add(article)
            
            session.commit()
            processed_total += len(unprocessed)
            print(f"✅ Completed batch. Total processed: {processed_total:,}\n")
    
    print("\n📊 ANALYSIS COMPLETE!")
    show_results()

def show_results():
    """Show comprehensive results"""
    
    with Session(engine) as session:
        # Get all articles with sentiment
        analyzed = session.exec(
            select(NewsArticle)
            .where(NewsArticle.sentiment_score.is_not(None))
        ).all()
        
        if not analyzed:
            print("No articles analyzed!")
            return
        
        print(f"\n📈 SENTIMENT ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Total articles analyzed: {len(analyzed):,}")
        
        # Overall sentiment stats
        sentiments = [a.sentiment_score for a in analyzed]
        print(f"\nOverall Sentiment:")
        print(f"  Mean: {np.mean(sentiments):.3f}")
        print(f"  Std Dev: {np.std(sentiments):.3f}")
        print(f"  Range: [{min(sentiments):.3f}, {max(sentiments):.3f}]")
        
        # By time period
        print("\nSentiment by Quarter:")
        quarters = [
            ('2023 Q3', '2023-07-01', '2023-09-30'),
            ('2023 Q4', '2023-10-01', '2023-12-31'),
            ('2024 Q1', '2024-01-01', '2024-03-31'),
            ('2024 Q2', '2024-04-01', '2024-06-30'),
            ('2024 Q3', '2024-07-01', '2024-09-30'),
            ('2024 Q4', '2024-10-01', '2024-12-31'),
            ('2025 Q1', '2025-01-01', '2025-03-31'),
            ('2025 Q2', '2025-04-01', '2025-06-30'),
            ('2025 Q3', '2025-07-01', '2025-09-30')
        ]
        
        for quarter, start, end in quarters:
            quarter_articles = [
                a for a in analyzed
                if start <= str(a.published_date.date()) <= end
            ]
            if quarter_articles:
                quarter_sentiments = [a.sentiment_score for a in quarter_articles]
                avg_sentiment = np.mean(quarter_sentiments)
                
                # Get top topics
                all_topics = []
                for a in quarter_articles:
                    if a.topics:
                        all_topics.extend(a.topics.split(','))
                
                from collections import Counter
                top_topics = Counter(all_topics).most_common(3)
                topics_str = ', '.join([t[0] for t in top_topics]) if top_topics else 'none'
                
                print(f"  {quarter}: {len(quarter_articles):4d} articles, "
                      f"avg sentiment: {avg_sentiment:+.3f}, topics: {topics_str}")
        
        # Sentiment distribution
        print("\nSentiment Distribution:")
        label_counts = {}
        for a in analyzed:
            label = a.sentiment_label or 'unknown'
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(analyzed)
        for label in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
            count = label_counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {label:13s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # Key topics
        print("\nTop Topics Mentioned:")
        all_topics = []
        for a in analyzed:
            if a.topics:
                all_topics.extend(a.topics.split(','))
        
        topic_counts = Counter(all_topics).most_common(10)
        for topic, count in topic_counts:
            print(f"  {topic}: {count:,} mentions")
        
        # Countries mentioned
        print("\nCountries Most Mentioned:")
        all_countries = []
        for a in analyzed:
            if a.mentioned_countries:
                all_countries.extend(a.mentioned_countries.split(','))
        
        country_counts = Counter(all_countries).most_common(10)
        for country, count in country_counts[:5]:
            print(f"  {country}: {count:,} mentions")

if __name__ == "__main__":
    analyze_all_articles()
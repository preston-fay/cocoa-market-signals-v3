#!/usr/bin/env python3
"""
Run Comprehensive Data Collection for Cocoa Market Signals
Coordinates all data collectors and validates completeness
NO FAKE DATA - All sources are real and verified
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from sqlmodel import Session, select, func
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent))

from app.core.database import engine
from app.models.price_data import PriceData
from app.models.weather_data import WeatherData
from app.models.news_article import NewsArticle
from app.models.trade_data import TradeData

# Import collectors
from src.data_pipeline.comprehensive_news_collector import ComprehensiveNewsCollector
from src.data_pipeline.icco_data_parser import ICCODataParser
from src.data_pipeline.enhanced_weather_collector import EnhancedWeatherCollector

class DataCompletenessValidator:
    """Validate we have comprehensive data coverage"""
    
    def __init__(self):
        self.requirements = {
            'price_data': {
                'min_days': 500,
                'required_symbols': ['CC=F'],
                'max_gap_days': 3
            },
            'weather_data': {
                'min_locations': 15,
                'min_days_per_location': 365,
                'required_parameters': [
                    'temperature', 'precipitation', 'humidity',
                    'soil_moisture_0_10cm', 'evapotranspiration'
                ]
            },
            'news_articles': {
                'min_total_articles': 1000,
                'min_sources': 5,
                'min_articles_per_month': 30,
                'date_range_years': 2
            },
            'trade_data': {
                'min_countries': 10,
                'min_records': 500,
                'required_flows': ['Export', 'Import', 'Production']
            }
        }
    
    def validate_price_data(self) -> Dict:
        """Validate price data completeness"""
        with Session(engine) as session:
            # Count records
            total_records = session.scalar(
                select(func.count(PriceData.id))
                .where(PriceData.source == 'Yahoo Finance')
            )
            
            # Get date range
            date_range = session.exec(
                select(func.min(PriceData.date), func.max(PriceData.date))
                .where(PriceData.source == 'Yahoo Finance')
            ).first()
            
            if date_range and date_range[0] and date_range[1]:
                days_covered = (date_range[1] - date_range[0]).days
            else:
                days_covered = 0
            
            # Check for gaps
            all_dates = session.exec(
                select(PriceData.date)
                .where(PriceData.source == 'Yahoo Finance')
                .order_by(PriceData.date)
            ).all()
            
            max_gap = 0
            if len(all_dates) > 1:
                for i in range(1, len(all_dates)):
                    gap = (all_dates[i] - all_dates[i-1]).days
                    if gap > max_gap:
                        max_gap = gap
            
            return {
                'total_records': total_records,
                'days_covered': days_covered,
                'max_gap_days': max_gap,
                'meets_requirements': (
                    total_records >= self.requirements['price_data']['min_days'] and
                    max_gap <= self.requirements['price_data']['max_gap_days']
                )
            }
    
    def validate_weather_data(self) -> Dict:
        """Validate weather data completeness"""
        with Session(engine) as session:
            # Count unique locations
            locations = session.scalar(
                select(func.count(func.distinct(WeatherData.location)))
            )
            
            # Count total records
            total_records = session.scalar(
                select(func.count(WeatherData.id))
            )
            
            # Check parameters
            sample = session.exec(
                select(WeatherData).limit(1)
            ).first()
            
            has_required_params = False
            if sample:
                has_required_params = all([
                    sample.temperature is not None,
                    sample.precipitation is not None,
                    sample.humidity is not None,
                    sample.soil_moisture_0_10cm is not None,
                    sample.evapotranspiration is not None
                ])
            
            avg_days_per_location = total_records / locations if locations > 0 else 0
            
            return {
                'unique_locations': locations,
                'total_records': total_records,
                'avg_days_per_location': avg_days_per_location,
                'has_required_parameters': has_required_params,
                'meets_requirements': (
                    locations >= self.requirements['weather_data']['min_locations'] and
                    avg_days_per_location >= self.requirements['weather_data']['min_days_per_location'] and
                    has_required_params
                )
            }
    
    def validate_news_data(self) -> Dict:
        """Validate news data completeness"""
        with Session(engine) as session:
            # Count total articles
            total_articles = session.scalar(
                select(func.count(NewsArticle.id))
            )
            
            # Count unique sources
            unique_sources = session.scalar(
                select(func.count(func.distinct(NewsArticle.source)))
            )
            
            # Get date range
            date_range = session.exec(
                select(func.min(NewsArticle.published_date), 
                      func.max(NewsArticle.published_date))
            ).first()
            
            months_covered = 0
            avg_per_month = 0
            
            if date_range and date_range[0] and date_range[1]:
                months_covered = (
                    (date_range[1].year - date_range[0].year) * 12 + 
                    date_range[1].month - date_range[0].month + 1
                )
                avg_per_month = total_articles / months_covered if months_covered > 0 else 0
            
            return {
                'total_articles': total_articles,
                'unique_sources': unique_sources,
                'months_covered': months_covered,
                'avg_articles_per_month': avg_per_month,
                'meets_requirements': (
                    total_articles >= self.requirements['news_articles']['min_total_articles'] and
                    unique_sources >= self.requirements['news_articles']['min_sources'] and
                    avg_per_month >= self.requirements['news_articles']['min_articles_per_month']
                )
            }
    
    def validate_trade_data(self) -> Dict:
        """Validate trade data completeness"""
        with Session(engine) as session:
            # Count total records
            total_records = session.scalar(
                select(func.count(TradeData.id))
            )
            
            # Count unique countries
            unique_countries = session.scalar(
                select(func.count(func.distinct(TradeData.reporter_country)))
            )
            
            # Count trade flows
            unique_flows = session.exec(
                select(func.distinct(TradeData.trade_flow))
            ).all()
            
            return {
                'total_records': total_records,
                'unique_countries': unique_countries,
                'trade_flows': list(unique_flows),
                'meets_requirements': (
                    total_records >= self.requirements['trade_data']['min_records'] and
                    unique_countries >= self.requirements['trade_data']['min_countries']
                )
            }
    
    def generate_completeness_report(self) -> Dict:
        """Generate comprehensive data completeness report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'price_data': self.validate_price_data(),
            'weather_data': self.validate_weather_data(),
            'news_data': self.validate_news_data(),
            'trade_data': self.validate_trade_data()
        }
        
        # Overall status
        report['overall_complete'] = all(
            category['meets_requirements'] 
            for category in report.values() 
            if isinstance(category, dict) and 'meets_requirements' in category
        )
        
        return report

async def run_all_collectors():
    """Run all data collectors"""
    print("=" * 60)
    print("COMPREHENSIVE DATA COLLECTION FOR COCOA MARKET SIGNALS")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. News Collection
    print("\n1. COLLECTING NEWS ARTICLES")
    print("-" * 40)
    try:
        news_collector = ComprehensiveNewsCollector()
        articles = await news_collector.collect_all_sources(days_back=30)
        news_collector.save_to_database(articles)
        print(f"✓ News collection complete")
    except Exception as e:
        print(f"✗ News collection failed: {str(e)}")
    
    # 2. ICCO Data
    print("\n2. PARSING ICCO REPORTS")
    print("-" * 40)
    try:
        icco_parser = ICCODataParser()
        icco_data = icco_parser.process_icco_reports(limit=3)
        if icco_data:
            icco_parser.save_to_database(icco_data)
        print(f"✓ ICCO data parsing complete")
    except Exception as e:
        print(f"✗ ICCO parsing failed: {str(e)}")
    
    # 3. Weather Data
    print("\n3. COLLECTING ENHANCED WEATHER DATA")
    print("-" * 40)
    try:
        weather_collector = EnhancedWeatherCollector()
        weather_data = weather_collector.collect_all_regions(days_back=365)
        if weather_data:
            weather_collector.save_to_database(weather_data)
        print(f"✓ Weather collection complete")
    except Exception as e:
        print(f"✗ Weather collection failed: {str(e)}")
    
    # 4. Validate Completeness
    print("\n4. VALIDATING DATA COMPLETENESS")
    print("-" * 40)
    validator = DataCompletenessValidator()
    report = validator.generate_completeness_report()
    
    # Print report
    print("\n📊 DATA COMPLETENESS REPORT")
    print("=" * 60)
    
    for category, stats in report.items():
        if isinstance(stats, dict) and 'meets_requirements' in stats:
            status = "✓" if stats['meets_requirements'] else "✗"
            print(f"\n{category.upper()}: {status}")
            for key, value in stats.items():
                if key != 'meets_requirements':
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print(f"OVERALL STATUS: {'✓ COMPLETE' if report['overall_complete'] else '✗ INCOMPLETE'}")
    print("=" * 60)
    
    # Save report
    report_path = Path("data_completeness_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

def main():
    """Main entry point"""
    print("Starting comprehensive data collection...")
    print("This will collect REAL data from multiple sources.")
    print("No synthetic data will be generated.")
    
    # Run async collectors
    asyncio.run(run_all_collectors())
    
    print(f"\nCollection complete at: {datetime.now()}")

if __name__ == "__main__":
    main()
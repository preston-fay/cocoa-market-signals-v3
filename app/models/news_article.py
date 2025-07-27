"""
News Article Model - For CommonCrawl and news data
"""
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class NewsArticle(SQLModel, table=True):
    """News articles and market reports from CommonCrawl and other sources"""
    __tablename__ = "news_articles"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Article metadata
    published_date: datetime = Field(index=True)
    fetched_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Content
    url: str = Field(unique=True, index=True)
    title: str = Field()
    content: str = Field(description="Full article text")
    summary: Optional[str] = Field(default=None, description="Article summary")
    
    # Source information
    source: str = Field(index=True, description="Website/publication")
    source_type: str = Field(description="news, report, analysis, etc")
    author: Optional[str] = Field(default=None)
    
    # Relevance and sentiment
    relevance_score: float = Field(description="Relevance to cocoa market 0-1")
    sentiment_score: Optional[float] = Field(default=None, description="Sentiment -1 to 1")
    sentiment_label: Optional[str] = Field(default=None, description="positive/negative/neutral")
    
    # Key entities and topics
    mentioned_countries: Optional[str] = Field(default=None, description="JSON array of countries")
    mentioned_companies: Optional[str] = Field(default=None, description="JSON array of companies")
    topics: Optional[str] = Field(default=None, description="JSON array of topics")
    
    # Impact assessment
    market_impact: Optional[str] = Field(default=None, description="high/medium/low/none")
    event_type: Optional[str] = Field(default=None, description="weather/disease/policy/market/other")
    
    # Processing status
    processed: bool = Field(default=False)
    processing_notes: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
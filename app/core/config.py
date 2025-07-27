"""
Application configuration following preston-dev-setup standards
"""
from pydantic import BaseSettings, validator
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_NAME: str = "Cocoa Market Signals"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: Optional[str] = None
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_SECRET: str = os.getenv("JWT_SECRET", SECRET_KEY)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # External APIs
    YAHOO_FINANCE_API: Optional[str] = None
    OPENMETEO_API: Optional[str] = None
    
    # Model settings
    MODEL_CACHE_TTL: int = 3600  # 1 hour
    PREDICTION_HORIZON_DAYS: int = 7
    BACKTEST_DAYS: int = 90
    
    # Observability
    SENTRY_DSN: Optional[str] = None
    OTEL_ENDPOINT: Optional[str] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        """Build database URL from components if not provided directly"""
        if isinstance(v, str):
            return v
        # Default to SQLite for development
        return "sqlite:///./cocoa_market_signals.db"
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str) and v:
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return []
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
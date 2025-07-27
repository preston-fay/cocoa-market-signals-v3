"""
Database configuration using SQLModel
Following preston-dev-setup standards
"""
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from typing import AsyncGenerator

# Database URL - use PostgreSQL in production, SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cocoa_market_signals.db")

# For async operations, convert to async URL
if DATABASE_URL.startswith("sqlite"):
    ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./cocoa_market_signals.db"
elif DATABASE_URL.startswith("postgresql"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql", "postgresql+asyncpg")
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Create engines
engine = create_engine(DATABASE_URL, echo=False)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)
AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine, 
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session"""
    async with AsyncSessionLocal() as session:
        yield session

def get_sync_session() -> Session:
    """Dependency to get sync database session"""
    with SessionLocal() as session:
        yield session

async def init_db():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        # Import all models to ensure they're registered
        from app.models import price_data, prediction, signal, model_performance
        
        # Create tables
        await conn.run_sync(SQLModel.metadata.create_all)

async def check_db_health() -> bool:
    """Check if database is accessible"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
        return True
    except Exception:
        return False
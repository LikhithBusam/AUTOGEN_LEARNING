# """
# Database models for the Financial Analyst system
# """
# from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.sql import func
# from datetime import datetime
# from config.settings import settings

# Base = declarative_base()

# class UserQuery(Base):
#     """Store user queries and responses"""
#     __tablename__ = "user_queries"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     query_text = Column(Text, nullable=False)
#     response_data = Column(JSON)
#     created_at = Column(DateTime, default=func.now())
#     processing_time = Column(Float)  # in seconds
#     agents_involved = Column(JSON)  # list of agents that processed this query

# class Portfolio(Base):
#     """Store user portfolio data"""
#     __tablename__ = "portfolios"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     portfolio_name = Column(String, nullable=False)
#     holdings = Column(JSON)  # {"AAPL": {"shares": 100, "avg_cost": 150.0}, ...}
#     created_at = Column(DateTime, default=func.now())
#     updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
#     is_active = Column(Boolean, default=True)

# class StockData(Base):
#     """Cache stock data to reduce API calls"""
#     __tablename__ = "stock_data"
    
#     id = Column(Integer, primary_key=True, index=True)
#     symbol = Column(String, index=True, nullable=False)
#     data_type = Column(String, nullable=False)  # "daily", "intraday", "fundamentals"
#     data = Column(JSON, nullable=False)
#     fetched_at = Column(DateTime, default=func.now())
#     expires_at = Column(DateTime)

# class NewsData(Base):
#     """Store news data and sentiment analysis"""
#     __tablename__ = "news_data"
    
#     id = Column(Integer, primary_key=True, index=True)
#     symbol = Column(String, index=True)
#     headline = Column(Text, nullable=False)
#     content = Column(Text)
#     source = Column(String)
#     published_at = Column(DateTime)
#     sentiment_score = Column(Float)  # -1 to 1
#     sentiment_label = Column(String)  # "positive", "negative", "neutral"
#     fetched_at = Column(DateTime, default=func.now())

# class PriceAlert(Base):
#     """Store price alerts for users"""
#     __tablename__ = "price_alerts"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     symbol = Column(String, nullable=False)
#     alert_type = Column(String, nullable=False)  # "above", "below"
#     target_price = Column(Float, nullable=False)
#     current_price = Column(Float)
#     is_triggered = Column(Boolean, default=False)
#     created_at = Column(DateTime, default=func.now())
#     triggered_at = Column(DateTime)

# class AnalysisReport(Base):
#     """Store generated analysis reports"""
#     __tablename__ = "analysis_reports"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     report_type = Column(String, nullable=False)  # "comparison", "portfolio", "risk_analysis"
#     symbols = Column(JSON)  # list of symbols analyzed
#     report_data = Column(JSON)
#     file_path = Column(String)  # path to generated PDF/HTML
#     created_at = Column(DateTime, default=func.now())

# # Database connection
# engine = create_engine(settings.database_url)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def create_tables():
#     """Create all database tables"""
#     Base.metadata.create_all(bind=engine)

# def get_db():
#     """Get database session"""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Database utility functions
# class DatabaseManager:
#     """Database manager for common operations"""
    
#     def __init__(self):
#         self.engine = engine
#         self.SessionLocal = SessionLocal
    
#     async def init_db(self):
#         """Initialize database and create tables"""
#         try:
#             Base.metadata.create_all(bind=self.engine)
#             return True
#         except Exception as e:
#             raise Exception(f"Failed to initialize database: {str(e)}")
    
#     def save_user_query(self, user_id: str, query_text: str, response_data: dict, 
#                        processing_time: float, agents_involved: list):
#         """Save a user query and response"""
#         db = SessionLocal()
#         try:
#             query = UserQuery(
#                 user_id=user_id,
#                 query_text=query_text,
#                 response_data=response_data,
#                 processing_time=processing_time,
#                 agents_involved=agents_involved
#             )
#             db.add(query)
#             db.commit()
#             return query.id
#         finally:
#             db.close()
    
#     def get_user_portfolio(self, user_id: str, portfolio_name: str = "default"):
#         """Get user's portfolio"""
#         db = SessionLocal()
#         try:
#             portfolio = db.query(Portfolio).filter(
#                 Portfolio.user_id == user_id,
#                 Portfolio.portfolio_name == portfolio_name,
#                 Portfolio.is_active == True
#             ).first()
#             return portfolio
#         finally:
#             db.close()
    
#     def update_portfolio(self, user_id: str, holdings: dict, portfolio_name: str = "default"):
#         """Update user's portfolio"""
#         db = SessionLocal()
#         try:
#             portfolio = self.get_user_portfolio(user_id, portfolio_name)
#             if portfolio:
#                 portfolio.holdings = holdings
#                 portfolio.updated_at = datetime.now()
#             else:
#                 portfolio = Portfolio(
#                     user_id=user_id,
#                     portfolio_name=portfolio_name,
#                     holdings=holdings
#                 )
#                 db.add(portfolio)
#             db.commit()
#             return portfolio
#         finally:
#             db.close()
    
#     def cache_stock_data(self, symbol: str, data_type: str, data: dict, expires_at: datetime):
#         """Cache stock data"""
#         db = SessionLocal()
#         try:
#             stock_data = StockData(
#                 symbol=symbol,
#                 data_type=data_type,
#                 data=data,
#                 expires_at=expires_at
#             )
#             db.add(stock_data)
#             db.commit()
#             return stock_data.id
#         finally:
#             db.close()
    
#     def get_cached_stock_data(self, symbol: str, data_type: str):
#         """Get cached stock data if not expired"""
#         db = SessionLocal()
#         try:
#             stock_data = db.query(StockData).filter(
#                 StockData.symbol == symbol,
#                 StockData.data_type == data_type,
#                 StockData.expires_at > datetime.now()
#             ).first()
#             return stock_data.data if stock_data else None
#         finally:
#             db.close()

# # Global database manager instance
# db_manager = DatabaseManager()

# # Convenience functions for sync operations
# def create_tables():
#     """Create database tables (sync version)"""
#     engine = create_engine(f"sqlite:///{DATABASE_PATH}")
#     Base.metadata.create_all(engine)

# async def get_user_history(user_id: str):
#     """Get user query history"""
#     return await db_manager.get_user_history(user_id)

# async def save_user_query(user_id: str, query: str, result: dict):
#     """Save user query and result"""
#     return await db_manager.save_user_query(user_id, query, result)



from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from config.settings import settings

Base = declarative_base()

class UserQuery(Base):
    __tablename__ = "user_queries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    query_text = Column(Text, nullable=False)
    response_data = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    processing_time = Column(Float)  # in seconds
    agents_involved = Column(JSON)  # list of agents that processed this query

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    portfolio_name = Column(String, nullable=False)
    holdings = Column(JSON)  # {"AAPL": {"shares": 100, "avg_cost": 150.0}, ...}
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

class StockData(Base):
    __tablename__ = "stock_data"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    data_type = Column(String, nullable=False)  # "daily", "intraday", "fundamentals"
    data = Column(JSON, nullable=False)
    fetched_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)

class NewsData(Base):
    __tablename__ = "news_data"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    headline = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String)
    published_at = Column(DateTime)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String)  # "positive", "negative", "neutral"
    fetched_at = Column(DateTime, default=func.now())

class PriceAlert(Base):
    __tablename__ = "price_alerts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    symbol = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)  # "above", "below"
    target_price = Column(Float, nullable=False)
    current_price = Column(Float)
    is_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    triggered_at = Column(DateTime)

class AnalysisReport(Base):
    __tablename__ = "analysis_reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    report_type = Column(String, nullable=False)  # "comparison", "portfolio", "risk_analysis"
    symbols = Column(JSON)  # list of symbols analyzed
    report_data = Column(JSON)
    file_path = Column(String)  # path to generated PDF/HTML
    created_at = Column(DateTime, default=func.now())

# Async engine and sessionmaker pattern
DATABASE_URL = settings.database_url  # Must use async driver, e.g. "postgresql+asyncpg://..."

engine = create_async_engine(DATABASE_URL, echo=False, future=True)

async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession, autoflush=False)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.async_session = async_session

    async def save_user_query(self, user_id: str, query_text: str, response_data: dict,
                             processing_time: float, agents_involved: list):
        async with self.async_session() as session:
            query = UserQuery(
                user_id=user_id,
                query_text=query_text,
                response_data=response_data,
                processing_time=processing_time,
                agents_involved=agents_involved
            )
            session.add(query)
            await session.commit()
            await session.refresh(query)
            return query.id

    async def get_user_portfolio(self, user_id: str, portfolio_name: str = "default"):
        async with self.async_session() as session:
            result = await session.execute(
                Portfolio.__table__.select().where(
                    Portfolio.user_id == user_id,
                    Portfolio.portfolio_name == portfolio_name,
                    Portfolio.is_active == True
                )
            )
            portfolio = result.first()
            return portfolio

    async def update_portfolio(self, user_id: str, holdings: dict, portfolio_name: str = "default"):
        async with self.async_session() as session:
            result = await session.execute(
                Portfolio.__table__.select().where(
                    Portfolio.user_id == user_id,
                    Portfolio.portfolio_name == portfolio_name,
                    Portfolio.is_active == True
                )
            )
            portfolio = result.scalar()
            if portfolio:
                portfolio.holdings = holdings
                portfolio.updated_at = datetime.now()
            else:
                portfolio = Portfolio(
                    user_id=user_id,
                    portfolio_name=portfolio_name,
                    holdings=holdings
                )
                session.add(portfolio)
            await session.commit()
            return portfolio

    async def cache_stock_data(self, symbol: str, data_type: str, data: dict, expires_at: datetime):
        async with self.async_session() as session:
            stock_data = StockData(
                symbol=symbol,
                data_type=data_type,
                data=data,
                expires_at=expires_at
            )
            session.add(stock_data)
            await session.commit()
            await session.refresh(stock_data)
            return stock_data.id

    async def get_cached_stock_data(self, symbol: str, data_type: str):
        async with self.async_session() as session:
            result = await session.execute(
                StockData.__table__.select().where(
                    StockData.symbol == symbol,
                    StockData.data_type == data_type,
                    StockData.expires_at > datetime.now()
                )
            )
            stock_data = result.first()
            return stock_data.data if stock_data else None

# Global database manager instance
db_manager = DatabaseManager()

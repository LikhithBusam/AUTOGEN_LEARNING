# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, select
# from sqlalchemy.orm import declarative_base
# from sqlalchemy.sql import func
# from datetime import datetime
# from config.settings import settings

# Base = declarative_base()

# class UserQuery(Base):
#     __tablename__ = "user_queries"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     query_text = Column(Text, nullable=False)
#     response_data = Column(JSON)
#     created_at = Column(DateTime, default=func.now())
#     processing_time = Column(Float)  # in seconds
#     agents_involved = Column(JSON)  # list of agents that processed this query

# class Portfolio(Base):
#     __tablename__ = "portfolios"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     portfolio_name = Column(String, nullable=False)
#     holdings = Column(JSON)  # {"AAPL": {"shares": 100, "avg_cost": 150.0}, ...}
#     created_at = Column(DateTime, default=func.now())
#     updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
#     is_active = Column(Boolean, default=True)

# class StockData(Base):
#     __tablename__ = "stock_data"
#     id = Column(Integer, primary_key=True, index=True)
#     symbol = Column(String, index=True, nullable=False)
#     data_type = Column(String, nullable=False)  # "daily", "intraday", "fundamentals"
#     data = Column(JSON, nullable=False)
#     fetched_at = Column(DateTime, default=func.now())
#     expires_at = Column(DateTime)

# class NewsData(Base):
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
#     __tablename__ = "analysis_reports"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     report_type = Column(String, nullable=False)  # "comparison", "portfolio", "risk_analysis"
#     symbols = Column(JSON)  # list of symbols analyzed
#     report_data = Column(JSON)
#     file_path = Column(String)  # path to generated PDF/HTML
#     created_at = Column(DateTime, default=func.now())

# # Async engine and sessionmaker pattern
# DATABASE_URL = settings.database_url  # Must use async driver, e.g. "postgresql+asyncpg://..."

# engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession, autoflush=False)

# async def create_tables():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# class DatabaseManager:
#     def __init__(self):
#         self.engine = engine
#         self.async_session = async_session

#     async def save_user_query(self, user_id: str, query_text: str, response_data: dict,
#                              processing_time: float = 0.0, agents_involved: list = None):
#         async with self.async_session() as session:
#             query = UserQuery(
#                 user_id=user_id,
#                 query_text=query_text,
#                 response_data=response_data,
#                 processing_time=processing_time,
#                 agents_involved=agents_involved or []
#             )
#             session.add(query)
#             await session.commit()
#             await session.refresh(query)
#             return query.id

#     async def get_user_portfolio(self, user_id: str, portfolio_name: str = "default"):
#         async with self.async_session() as session:
#             result = await session.execute(
#                 select(Portfolio).where(
#                     Portfolio.user_id == user_id,
#                     Portfolio.portfolio_name == portfolio_name,
#                     Portfolio.is_active == True
#                 )
#             )
#             portfolio = result.scalar_one_or_none()
#             return portfolio

#     async def update_portfolio(self, user_id: str, holdings: dict, portfolio_name: str = "default"):
#         async with self.async_session() as session:
#             result = await session.execute(
#                 select(Portfolio).where(
#                     Portfolio.user_id == user_id,
#                     Portfolio.portfolio_name == portfolio_name,
#                     Portfolio.is_active == True
#                 )
#             )
#             portfolio = result.scalar_one_or_none()
#             if portfolio:
#                 portfolio.holdings = holdings
#                 portfolio.updated_at = datetime.now()
#             else:
#                 portfolio = Portfolio(
#                     user_id=user_id,
#                     portfolio_name=portfolio_name,
#                     holdings=holdings
#                 )
#                 session.add(portfolio)
#             await session.commit()
#             return portfolio

#     async def cache_stock_data(self, symbol: str, data_type: str, data: dict, expires_at: datetime):
#         async with self.async_session() as session:
#             stock_data = StockData(
#                 symbol=symbol,
#                 data_type=data_type,
#                 data=data,
#                 expires_at=expires_at
#             )
#             session.add(stock_data)
#             await session.commit()
#             await session.refresh(stock_data)
#             return stock_data.id

#     async def get_cached_stock_data(self, symbol: str, data_type: str):
#         async with self.async_session() as session:
#             result = await session.execute(
#                 select(StockData).where(
#                     StockData.symbol == symbol,
#                     StockData.data_type == data_type,
#                     StockData.expires_at > datetime.now()
#                 )
#             )
#             stock_data = result.scalar_one_or_none()
#             return stock_data.data if stock_data else None

# # Global database manager instance
# db_manager = DatabaseManager()

    
# # Provide a convenience init function expected by main.py
# async def init_db():
#     """Initialize database tables (convenience wrapper)"""
#     await create_tables()

# # Backwards-compatible: attach init_db method to db_manager instance
# setattr(db_manager, 'init_db', init_db)



from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, select
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


# Async engine and sessionmaker
DATABASE_URL = settings.database_url   # should be something like "postgresql+asyncpg://user:pass@host/dbname"

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
                              processing_time: float = 0.0, agents_involved: list = None):
        async with self.async_session() as session:
            query = UserQuery(
                user_id=user_id,
                query_text=query_text,
                response_data=response_data,
                processing_time=processing_time,
                agents_involved=agents_involved or []
            )
            session.add(query)
            await session.commit()
            await session.refresh(query)
            return query.id

    async def get_user_portfolio(self, user_id: str, portfolio_name: str = "default"):
        async with self.async_session() as session:
            result = await session.execute(
                select(Portfolio).where(
                    Portfolio.user_id == user_id,
                    Portfolio.portfolio_name == portfolio_name,
                    Portfolio.is_active.is_(True)
                )
            )
            portfolio = result.scalar_one_or_none()
            return portfolio

    async def update_portfolio(self, user_id: str, holdings: dict, portfolio_name: str = "default"):
        async with self.async_session() as session:
            result = await session.execute(
                select(Portfolio).where(
                    Portfolio.user_id == user_id,
                    Portfolio.portfolio_name == portfolio_name,
                    Portfolio.is_active.is_(True)
                )
            )
            portfolio = result.scalar_one_or_none()
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
                select(StockData).where(
                    StockData.symbol == symbol,
                    StockData.data_type == data_type,
                    StockData.expires_at > datetime.now()
                )
            )
            stock_data = result.scalar_one_or_none()
            return stock_data.data if stock_data else None


db_manager = DatabaseManager()


# Convenience init function
async def init_db():
    """Initialize database tables"""
    await create_tables()

# Attach init_db to db_manager instance for convenience
setattr(db_manager, 'init_db', init_db)

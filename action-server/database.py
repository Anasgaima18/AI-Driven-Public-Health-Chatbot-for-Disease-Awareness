"""
Database models for AI-Driven Public Health Chatbot
Handles user data, health queries, and outbreak information
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    phone_number = Column(String(20), unique=True, index=True)
    language_preference = Column(String(10), default='en')
    location = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    opted_in_alerts = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

class HealthQuery(Base):
    __tablename__ = 'health_queries'
    
    id = Column(Integer, primary_key=True)
    user_phone = Column(String(20), index=True)
    query_text = Column(Text)
    language = Column(String(10))
    symptoms = Column(Text)
    response_text = Column(Text)
    accuracy_rating = Column(Float)  # User feedback rating
    created_at = Column(DateTime, default=datetime.utcnow)

class DiseaseOutbreak(Base):
    __tablename__ = 'disease_outbreaks'
    
    id = Column(Integer, primary_key=True)
    disease_name = Column(String(100))
    location = Column(String(100))
    severity = Column(String(20))  # low, moderate, high
    cases_count = Column(Integer)
    prevention_advice = Column(Text)
    status = Column(String(20), default='active')  # active, resolved
    reported_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class VaccinationSchedule(Base):
    __tablename__ = 'vaccination_schedules'
    
    id = Column(Integer, primary_key=True)
    vaccine_name = Column(String(100))
    location = Column(String(100))
    scheduled_date = Column(DateTime)
    age_group = Column(String(50))
    doses_available = Column(Integer)
    center_name = Column(String(200))
    contact_number = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class HealthAlert(Base):
    __tablename__ = 'health_alerts'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50))  # outbreak, vaccination, general
    title = Column(String(200))
    message = Column(Text)
    target_location = Column(String(100))
    target_language = Column(String(10))
    sent_count = Column(Integer, default=0)
    delivery_status = Column(String(20), default='pending')  # pending, sent, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)

class ChatbotMetrics(Base):
    __tablename__ = 'chatbot_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_queries = Column(Integer, default=0)
    successful_responses = Column(Integer, default=0)
    average_accuracy = Column(Float, default=0.0)
    language_distribution = Column(Text)  # JSON string
    top_health_topics = Column(Text)  # JSON string
    unique_users = Column(Integer, default=0)

# Database connection setup
def get_database_url():
    """Get database URL from environment or use default SQLite for development"""
    return os.getenv('DATABASE_URL', 'sqlite:///health_chatbot.db')

def create_database_engine():
    """Create database engine with proper configuration"""
    database_url = get_database_url()
    
    # Handle Railway's PostgreSQL URL format
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(database_url)
    return engine

def init_database():
    """Initialize database tables"""
    engine = create_database_engine()
    Base.metadata.create_all(bind=engine)
    return engine

def get_session():
    """Get database session"""
    engine = create_database_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# Usage example:
# from database import init_database, get_session, User, HealthQuery
# 
# # Initialize database
# init_database()
# 
# # Create session and add user
# session = get_session()
# new_user = User(phone_number="+919876543210", language_preference="hi", location="Mumbai")
# session.add(new_user)
# session.commit()
# session.close()
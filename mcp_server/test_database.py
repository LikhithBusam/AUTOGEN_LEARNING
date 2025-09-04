#!/usr/bin/env python3
"""
Test database functionality
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_database():
    print("Testing database functionality...")
    
    try:
        from data.database import db_manager, create_tables
        print("✅ Database module imported")
        
        # Test database initialization
        await create_tables()
        print("✅ Database tables created successfully")
        
        # Test database manager
        await db_manager.init_db()
        print("✅ Database manager initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_database())
    if result:
        print("🎉 Database test passed!")
    else:
        print("⚠️ Database test failed!")

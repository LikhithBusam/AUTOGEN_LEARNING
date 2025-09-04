#!/usr/bin/env python3
"""
Simple config test
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing configuration import...")

try:
    from config.settings import settings
    print("✅ Settings imported successfully")
    print(f"Database URL: {settings.database_url}")
    print(f"Log level: {settings.log_level}")
    print(f"Reports dir: {settings.reports_output_dir}")
except Exception as e:
    print(f"❌ Settings import failed: {e}")
    import traceback
    traceback.print_exc()

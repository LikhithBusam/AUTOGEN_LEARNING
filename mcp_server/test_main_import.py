try:
    import main
    print("✅ SUCCESS: Main application imported successfully!")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

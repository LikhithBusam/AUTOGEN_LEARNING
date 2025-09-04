"""
Debug ModelInfo to understand required fields
"""
try:
    from autogen_ext.models.openai._model_info import ModelInfo, ModelFamily
    print("✅ ModelInfo and ModelFamily imported successfully")
    
    # Check ModelFamily attributes
    print("\n🔍 ModelFamily attributes:")
    family_attrs = [attr for attr in dir(ModelFamily) if not attr.startswith('_')]
    print(family_attrs)
    
    # Try to create a basic ModelInfo to see what's required
    print("\n🔍 Testing ModelInfo creation:")
    try:
        info = ModelInfo(
            family="unknown",
            vision=True,
            function_calling=True,
            json_output=True
        )
        print("✅ ModelInfo created successfully with family='unknown'")
    except Exception as e:
        print(f"❌ ModelInfo creation failed: {e}")
        
    # Try with different family values
    for family in ["openai", "anthropic", "google", "mistral", "meta"]:
        try:
            info = ModelInfo(
                family=family,
                vision=True,
                function_calling=True,
                json_output=True
            )
            print(f"✅ ModelInfo created successfully with family='{family}'")
            break
        except Exception as e:
            print(f"❌ ModelInfo creation failed with family='{family}': {e}")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

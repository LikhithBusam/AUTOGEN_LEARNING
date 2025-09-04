import sys, traceback, importlib
sys.path.append(r'c:\Users\Likith\OneDrive\Desktop\fin\AUTOGEN_LEARNING\mcp_server')
modules=['main','data.database','mcp.financial_data_server','agents.orchestrator_agent']
for m in modules:
    try:
        importlib.import_module(m)
        print(m + ' OK')
    except Exception:
        print(m + ' FAIL')
        traceback.print_exc()

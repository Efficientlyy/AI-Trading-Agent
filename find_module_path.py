try:
    import ai_trading_agent_rs
    print(f"ai_trading_agent_rs module file: {ai_trading_agent_rs.__file__}")
except ImportError:
    print("ai_trading_agent_rs module not found.")
except AttributeError:
    print("ai_trading_agent_rs module is likely found, but __file__ attribute is missing (could be a namespace package or built-in module).")
    import sys
    if 'ai_trading_agent_rs' in sys.modules:
        print(f"Module object: {sys.modules['ai_trading_agent_rs']}")
    else:
        print("Module not in sys.modules either.")

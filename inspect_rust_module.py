import ai_trading_agent_rs
import inspect

print("--- dir(ai_trading_agent_rs) ---")
print(dir(ai_trading_agent_rs))

print("\n--- Functions in ai_trading_agent_rs ---")
for name, obj in inspect.getmembers(ai_trading_agent_rs):
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        print(name)

# Try to access the specific function to see the error if it still occurs
try:
    print("\n--- Attempting to access create_stochastic_oscillator_rs ---")
    func = ai_trading_agent_rs.create_stochastic_oscillator_rs
    print(f"Successfully accessed: {func}")
except AttributeError as e:
    print(f"Error accessing: {e}")

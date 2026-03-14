import subprocess
import os

try:
    # Set encoding to utf-8 explicitly to avoid platform issues
    with open('test_debug.txt', 'w', encoding='utf-8') as f:
        subprocess.run(['pytest', 'tests/test_price_tools_sql.py'], stdout=f, stderr=subprocess.STDOUT, text=True)
except Exception as e:
    with open('test_debug.txt', 'a', encoding='utf-8') as f:
        f.write(f"\nExecution failed: {e}")

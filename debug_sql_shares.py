import os
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# We need the actual engine to test
try:
    from core.query_executor import ENGINE
    from agent.sql_executor import BALANCING_SHARE_PIVOT_SQL
    
    print(f"Testing BALANCING_SHARE_PIVOT_SQL...")
    print(f"Query:\n{BALANCING_SHARE_PIVOT_SQL}")
    
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(BALANCING_SHARE_PIVOT_SQL), conn)
        print(f"\nResult rows: {len(df)}")
        if not df.empty:
            print("\nFirst 5 rows:")
            print(df.head())
        else:
            print("\nNo rows returned!")
            
            # Check table columns
            print("\nChecking columns in trade_derived_entities...")
            cols_df = pd.read_sql(text("SELECT * FROM trade_derived_entities LIMIT 0"), conn)
            print(f"Columns: {list(cols_df.columns)}")
            
            # Check segments
            print("\nChecking distinct segments...")
            segs_df = pd.read_sql(text("SELECT DISTINCT segment FROM trade_derived_entities"), conn)
            print(f"Segments: {segs_df['segment'].tolist()}")

except Exception as e:
    print(f"\nError: {e}")

# Refactoring Guide: Breaking Down Monolithic main.py

**Current State:** main.py = 3,900 lines (unmaintainable)
**Target State:** Modular architecture with files <500 lines each

---

## Recommended Architecture

```
langchain_railway/
├── main.py                          # FastAPI app only (~200 lines)
├── config.py                        # Environment & constants (~100 lines)
├── models.py                        # Pydantic models (~100 lines)
│
├── core/                            # Core functionality
│   ├── __init__.py
│   ├── llm.py                       # LLM chain management (~300 lines)
│   ├── sql_generator.py             # SQL generation + validation (~400 lines)
│   ├── query_executor.py            # Database execution (~200 lines)
│   └── cache.py                     # Caching logic (~150 lines)
│
├── analysis/                        # Data analysis
│   ├── __init__.py
│   ├── shares.py                    # Share calculations (~300 lines)
│   ├── seasonal.py                  # Seasonal decomposition (~200 lines)
│   ├── stats.py                     # Statistics & trends (~200 lines)
│   └── entity_contributions.py     # Entity price contributions (~150 lines)
│
├── visualization/                   # Chart generation
│   ├── __init__.py
│   ├── chart_selector.py            # Chart type selection (~200 lines)
│   ├── chart_builder.py             # Chart data formatting (~200 lines)
│   └── chart_validator.py           # Validate chart matches answer (~100 lines)
│
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── language.py                  # Language detection (~100 lines)
│   ├── metrics.py                   # Observability (~100 lines)
│   └── validators.py                # Input validation (~100 lines)
│
├── sql_helpers.py                   # SQL intent detection (existing)
├── context.py                       # Schema & labels (existing)
├── domain_knowledge.py              # Domain knowledge (existing)
│
└── tests/                           # Unit tests
    ├── test_llm.py
    ├── test_sql_generation.py
    ├── test_shares.py
    ├── test_chart_selection.py
    └── test_integration.py
```

---

## Step-by-Step Refactoring Plan

### Phase 1: Extract Configuration & Models (Low Risk)

**Estimated Time:** 1 hour

#### Step 1.1: Create `config.py`

Extract all configuration and constants from main.py:

```python
# config.py
"""
Application configuration and constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini")

# API
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
PORT = int(os.getenv("PORT", 8000))

# SQL
MAX_ROWS = 3750
SQL_TIMEOUT_SECONDS = 30

# Allowed tables for SQL validation
ALLOWED_TABLES = {
    "entities_mv",
    "price_with_usd",
    "tariff_with_usd",
    "tech_quantity_view",
    "trade_derived_entities",
    "monthly_cpi_mv",
    "energy_balance_long_mv"
}

# Table synonyms for auto-correction
TABLE_SYNONYMS = {
    "prices": "price_with_usd",
    "tariffs": "tariff_with_usd",
    "tech_quantity": "tech_quantity_view",
    "trade": "trade_derived_entities",
}

# Cache settings
CACHE_MAX_SIZE = 1000
CACHE_EVICTION_PERCENT = 0.1

# Analysis settings
SUMMER_MONTHS = [4, 5, 6, 7]
WINTER_MONTHS = [1, 2, 3, 8, 9, 10, 11, 12]
```

#### Step 1.2: Create `models.py`

Extract Pydantic models:

```python
# models.py
"""
Pydantic models for API requests and responses.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Question(BaseModel):
    """User question model."""
    query: str = Field(..., description="Natural language query")
    mode: str = Field(default="light", description="Analysis mode: 'light' or 'analyst'")

class APIResponse(BaseModel):
    """API response model."""
    answer: str
    chart_data: Optional[List[Dict[str, Any]]] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict[str, Any]] = None
    execution_time: float

class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    total_requests: int
    total_llm_calls: int
    total_sql_executions: int
    total_errors: int
    cache_hit_rate: float
    average_response_time: float
```

---

### Phase 2: Extract Core Functionality (Medium Risk)

**Estimated Time:** 4 hours

#### Step 2.1: Create `core/llm.py`

Extract all LLM-related functions:

```python
# core/llm.py
"""
LLM chain management and caching.
"""
import hashlib
import logging
from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import GEMINI_API_KEY, OPENAI_API_KEY, MODEL_TYPE, CACHE_MAX_SIZE
from domain_knowledge import DOMAIN_KNOWLEDGE
import json

log = logging.getLogger("Enai")

class LLMResponseCache:
    """In-memory cache for LLM responses."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        """Get cached response."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str):
        """Set cached response."""
        if len(self.cache) >= self.max_size:
            # Evict 10% oldest entries (simple FIFO)
            evict_count = int(self.max_size * 0.1)
            for _ in range(evict_count):
                self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# Global cache instance
llm_cache = LLMResponseCache()

def make_gemini():
    """Create Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )

def make_openai():
    """Create OpenAI LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_plan_and_sql(
    user_query: str,
    analysis_mode: str,
    lang_instruction: str,
    domain_json: str,
    schema_doc: str,
    few_shot_examples: str
) -> str:
    """
    Generate analysis plan and SQL query in one LLM call.

    Args:
        user_query: User's natural language query
        analysis_mode: 'light' or 'analyst'
        lang_instruction: Language instruction for response
        domain_json: Relevant domain knowledge JSON
        schema_doc: Database schema documentation
        few_shot_examples: Few-shot SQL examples

    Returns:
        String with format: "{JSON plan}\n---SQL---\n{SQL query}"
    """
    # Check cache first
    cache_key = hashlib.sha256(
        f"sql_v2|{user_query}|{analysis_mode}|{lang_instruction}".encode()
    ).hexdigest()

    cached = llm_cache.get(cache_key)
    if cached:
        log.info("📝 Plan/SQL: (cached)")
        return cached

    # Build prompt
    system = """You are an analytical PostgreSQL generator for Georgian energy market data.
    Generate both a JSON plan and SQL query in one response.

    Output format:
    {JSON plan}
    ---SQL---
    {SQL query}
    """

    prompt = f"""
User question: {user_query}

Domain knowledge: {domain_json}

Schema: {schema_doc}

Examples: {few_shot_examples}

{lang_instruction}
"""

    # Call LLM
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()

        # Cache response
        llm_cache.set(cache_key, response)

        return response
    except Exception as e:
        log.warning(f"Primary LLM failed: {e}, trying fallback")
        llm = make_openai()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        llm_cache.set(cache_key, response)
        return response

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_summarize(
    user_query: str,
    data_preview: str,
    stats_hint: str,
    lang_instruction: str,
    domain_json: str
) -> str:
    """
    Generate natural language answer from query results.

    Args:
        user_query: User's original query
        data_preview: Preview of query results
        stats_hint: Statistical summary
        lang_instruction: Language instruction
        domain_json: Relevant domain knowledge

    Returns:
        Natural language answer
    """
    # Check cache
    cache_key = hashlib.sha256(
        f"summary_v2|{user_query}|{data_preview[:200]}".encode()
    ).hexdigest()

    cached = llm_cache.get(cache_key)
    if cached:
        log.info("📝 Summary: (cached)")
        return cached

    system = """Generate a clear, concise answer based on the query results.
    Focus on answering the user's question directly.
    """

    prompt = f"""
User question: {user_query}

Query results: {data_preview}

Statistics: {stats_hint}

Domain knowledge: {domain_json}

{lang_instruction}
"""

    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        llm_cache.set(cache_key, response)
        return response
    except Exception as e:
        log.warning(f"Primary LLM failed: {e}, trying fallback")
        llm = make_openai()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        llm_cache.set(cache_key, response)
        return response

def get_cache_stats() -> Dict[str, float]:
    """Get cache statistics."""
    return {
        "hit_rate": llm_cache.hit_rate(),
        "hits": llm_cache.hits,
        "misses": llm_cache.misses,
        "size": len(llm_cache.cache)
    }
```

#### Step 2.2: Create `core/sql_generator.py`

Extract SQL generation logic:

```python
# core/sql_generator.py
"""
SQL generation, validation, and security.
"""
import re
import logging
from typing import Tuple
from sqlglot import parse_one, exp, ParseError
from fastapi import HTTPException

from config import ALLOWED_TABLES, TABLE_SYNONYMS, MAX_ROWS
from sql_helpers import detect_aggregation_intent, validate_aggregation_logic

log = logging.getLogger("Enai")

# Pre-compiled regex patterns for performance
SYNONYM_PATTERNS = [
    (re.compile(r'\bprices\b', re.IGNORECASE), 'price_with_usd'),
    (re.compile(r'\btariffs\b', re.IGNORECASE), 'tariff_with_usd'),
    (re.compile(r'\btech_quantity\b', re.IGNORECASE), 'tech_quantity_view'),
]
LIMIT_PATTERN = re.compile(r'\bLIMIT\b', re.IGNORECASE)

def validate_table_whitelist(sql: str) -> None:
    """
    Validate SQL only uses allowed tables.

    Uses SQLGlot AST parsing for robust validation.
    Supports CTEs and complex queries.

    Args:
        sql: SQL query to validate

    Raises:
        HTTPException: If SQL uses unauthorized tables
    """
    try:
        parsed = parse_one(sql, read='postgres')

        # Extract CTE names (allowed as temporary tables)
        cte_names = set()
        with_clause = parsed.find(exp.With)
        if with_clause:
            for cte in with_clause.expressions:
                if cte.alias is not None:
                    cte_names.add(cte.alias.lower())

        # Extract all table references
        for table_exp in parsed.find_all(exp.Table):
            table_name = table_exp.name.lower().split('.')[0]

            # Skip CTE names
            if table_name in cte_names:
                continue

            # Apply synonym mapping
            canonical_name = TABLE_SYNONYMS.get(table_name, table_name)

            # Check whitelist
            if canonical_name not in ALLOWED_TABLES:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Unauthorized table: `{table_name}`. Allowed: {sorted(ALLOWED_TABLES)}"
                )

        log.info(f"✅ Table whitelist validation passed")

    except ParseError as e:
        log.error(f"SQL parse error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL validation failed: Cannot parse query. {e}"
        )
    except Exception as e:
        log.error(f"Unexpected validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL validation error: {e}"
        )

def apply_synonym_corrections(sql: str) -> str:
    """Apply table synonym auto-corrections."""
    corrected = sql
    for pattern, replacement in SYNONYM_PATTERNS:
        corrected = pattern.sub(replacement, corrected)
    return corrected

def ensure_limit(sql: str) -> str:
    """Ensure SQL has LIMIT clause."""
    if " from " in sql.lower() and not LIMIT_PATTERN.search(sql):
        # Remove trailing semicolon if exists
        sql = sql.rstrip().rstrip(';')
        sql = f"{sql}\nLIMIT {MAX_ROWS}"
    return sql

def sanitize_sql(sql: str) -> str:
    """Basic SQL sanitization."""
    # Remove markdown fences
    sql = sql.strip().strip('`').strip()

    # Remove comments
    sql = re.sub(r"--.*", "", sql)

    # Ensure SELECT only
    if not sql.lower().startswith("select"):
        raise HTTPException(400, "Only SELECT statements allowed")

    return sql

def validate_aggregation_intent_match(sql: str, user_query: str) -> Tuple[bool, str]:
    """
    Validate SQL matches user's aggregation intent.

    Args:
        sql: Generated SQL query
        user_query: User's original query

    Returns:
        Tuple of (is_valid, reason)
    """
    intent = detect_aggregation_intent(user_query)
    is_valid, reason = validate_aggregation_logic(sql, intent)

    if not is_valid:
        log.warning(f"⚠️ SQL doesn't match intent: {reason}")
        log.warning(f"⚠️ User query: {user_query}")
        log.warning(f"⚠️ SQL: {sql[:200]}...")
    else:
        log.info(f"✅ Aggregation validation: {reason}")

    return is_valid, reason

def process_sql(raw_sql: str, user_query: str) -> str:
    """
    Process and validate SQL query.

    Steps:
    1. Sanitize
    2. Validate table whitelist
    3. Apply synonym corrections
    4. Ensure LIMIT clause
    5. Validate aggregation intent

    Args:
        raw_sql: Raw SQL from LLM
        user_query: User's original query

    Returns:
        Validated and corrected SQL

    Raises:
        HTTPException: If validation fails
    """
    # Sanitize
    sql = sanitize_sql(raw_sql)

    # Validate whitelist
    validate_table_whitelist(sql)

    # Apply corrections
    sql = apply_synonym_corrections(sql)
    sql = ensure_limit(sql)

    # Validate aggregation intent (non-blocking, just logs)
    validate_aggregation_intent_match(sql, user_query)

    return sql
```

#### Step 2.3: Create `core/query_executor.py`

Extract database execution logic:

```python
# core/query_executor.py
"""
Database query execution with security and monitoring.
"""
import time
import logging
from typing import Tuple, List
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import DatabaseError, OperationalError

from config import SUPABASE_URL, SUPABASE_KEY, SQL_TIMEOUT_SECONDS

log = logging.getLogger("Enai")

# Database engine
ENGINE = create_engine(
    f"postgresql://postgres:{SUPABASE_KEY}@{SUPABASE_URL.replace('https://', '')}/postgres",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
    connect_args={
        "options": f"-c statement_timeout={SQL_TIMEOUT_SECONDS * 1000}"
    }
)

def execute_sql(sql: str) -> Tuple[pd.DataFrame, List[str], List[tuple], float]:
    """
    Execute SQL query with read-only enforcement.

    Args:
        sql: Validated SQL query

    Returns:
        Tuple of (DataFrame, column_names, rows, execution_time)

    Raises:
        DatabaseError: If query fails or attempts modification
        OperationalError: If timeout or connection issues
    """
    start_time = time.time()

    try:
        with ENGINE.connect() as conn:
            # Enforce read-only mode
            conn.execute(text("SET TRANSACTION READ ONLY"))

            # Execute query
            result = conn.execute(text(sql))

            # Fetch results
            rows = result.fetchall()
            cols = list(result.keys())

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

            execution_time = time.time() - start_time

            log.info(f"✅ SQL executed: {len(rows)} rows in {execution_time:.2f}s")

            return df, cols, rows, execution_time

    except OperationalError as e:
        log.error(f"⚠️ Database operation error: {e}")
        raise DatabaseError(f"Query timeout or connection error: {e}")

    except DatabaseError as e:
        log.error(f"⚠️ Database error: {e}")
        raise

    except Exception as e:
        log.error(f"⚠️ Unexpected query execution error: {e}")
        raise DatabaseError(f"Query execution failed: {e}")

def test_connection() -> bool:
    """Test database connection."""
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error(f"Database connection test failed: {e}")
        return False
```

---

### Phase 3: Extract Analysis Functions (Low Risk)

**Estimated Time:** 3 hours

#### Step 3.1: Create `analysis/shares.py`

```python
# analysis/shares.py
"""
Share calculation functions for balancing electricity composition.
"""
import pandas as pd
import logging
from typing import Dict

log = logging.getLogger("Enai")

def calculate_balancing_shares(
    trade_data: pd.DataFrame,
    entities: list
) -> pd.DataFrame:
    """
    Calculate entity shares in balancing electricity.

    Args:
        trade_data: DataFrame from trade_derived_entities
        entities: List of entity names to calculate shares for

    Returns:
        DataFrame with date and share_* columns
    """
    # Filter to balancing segment only
    balancing = trade_data[
        trade_data['segment'].str.lower().str.replace(' ', '_') == 'balancing'
    ].copy()

    # Group by date and calculate totals
    shares = balancing.groupby('date').apply(
        lambda g: pd.Series({
            'total_qty': g['quantity'].sum(),
            **{
                f'share_{entity}': g[g['entity'] == entity]['quantity'].sum() / g['quantity'].sum()
                for entity in entities
            }
        })
    ).reset_index()

    return shares

# More share calculation functions...
```

#### Step 3.2: Create `analysis/seasonal.py`

```python
# analysis/seasonal.py
"""
Seasonal decomposition and analysis.
"""
import pandas as pd
import numpy as np
from config import SUMMER_MONTHS, WINTER_MONTHS

def add_season_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add season column to DataFrame.

    Args:
        df: DataFrame with date column
        date_col: Name of date column

    Returns:
        DataFrame with 'season' column added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['season'] = np.where(
        df[date_col].dt.month.isin(SUMMER_MONTHS),
        'summer',
        'winter'
    )
    return df

def compute_seasonal_average(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Compute seasonal averages and CAGR.

    Args:
        df: DataFrame with date and value columns
        value_col: Column to average
        date_col: Date column name

    Returns:
        DataFrame with seasonal statistics
    """
    df = add_season_column(df, date_col)

    # Calculate seasonal averages
    seasonal = df.groupby('season')[value_col].mean().reset_index()

    # TODO: Add CAGR calculation

    return seasonal

# More seasonal analysis functions...
```

#### Step 3.3: Create `analysis/stats.py`

```python
# analysis/stats.py
"""
Statistical analysis and quick stats generation.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

def quick_stats(df: pd.DataFrame) -> str:
    """
    Generate quick statistics summary.

    Args:
        df: Query results DataFrame

    Returns:
        String summary of statistics
    """
    if df.empty:
        return "No data returned"

    stats = [f"Rows: {len(df)}"]

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        stats.append(f"{col}: {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f})")

    # Detect trends
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
    if date_cols and numeric_cols:
        # TODO: Add trend detection
        pass

    return "\n".join(stats)

# More stats functions...
```

---

### Phase 4: Extract Visualization (Low Risk)

**Estimated Time:** 2 hours

#### Step 4.1: Create `visualization/chart_selector.py`

```python
# visualization/chart_selector.py
"""
Semantic chart type selection based on data structure.
"""
import pandas as pd
import logging
from typing import Optional, Dict

log = logging.getLogger("Enai")

def infer_dimension(col_name: str) -> str:
    """
    Infer semantic dimension from column name.

    Args:
        col_name: Column name

    Returns:
        Dimension type: 'xrate', 'share', 'price_tariff', 'energy_qty', 'index', 'other'
    """
    col_lower = col_name.lower()

    # Exchange rate (check FIRST before price)
    if any(x in col_lower for x in ["xrate", "exchange", "rate"]):
        return "xrate"

    # Shares
    if any(x in col_lower for x in ["share_", "proportion", "percent"]):
        return "share"

    # Index
    if any(x in col_lower for x in ["cpi", "index", "inflation"]):
        return "index"

    # Quantity
    if any(x in col_lower for x in ["quantity", "generation", "volume", "mw", "tj"]):
        return "energy_qty"

    # Price/Tariff (check AFTER xrate)
    if any(x in col_lower for x in ["price", "tariff", "_gel", "_usd"]):
        return "price_tariff"

    return "other"

def select_chart_type(
    df: pd.DataFrame,
    user_query: str
) -> tuple[Optional[str], Dict]:
    """
    Select appropriate chart type based on data structure.

    Args:
        df: Query results DataFrame
        user_query: User's original query

    Returns:
        Tuple of (chart_type, metadata)
    """
    if df.empty:
        return None, {}

    # Detect structural features
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "year", "month"])]
    category_cols = [c for c in df.columns if any(k in c.lower() for k in ["type", "sector", "entity"])]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Infer dimensions
    dimensions = {col: infer_dimension(col) for col in numeric_cols}
    dim_types = set(dimensions.values())

    has_time = len(time_cols) > 0
    has_categories = len(category_cols) > 0

    # Decision matrix
    if has_time and "share" in dim_types:
        return "stackedbar", {"reason": "Time series with shares"}

    elif has_time and not has_categories:
        return "line", {"reason": "Single time series"}

    elif has_time and has_categories:
        return "line", {"reason": "Multi-line trend"}

    elif not has_time and "share" in dim_types:
        n_cats = df[category_cols[0]].nunique() if category_cols else 0
        if n_cats <= 8:
            return "pie", {"reason": "Composition snapshot"}
        return "bar", {"reason": "Composition with many categories"}

    elif not has_time and has_categories:
        return "bar", {"reason": "Categorical comparison"}

    return "line", {"reason": "Default"}

# More chart selection functions...
```

---

### Phase 5: Update main.py (Medium Risk)

**Estimated Time:** 2 hours

#### New `main.py` (simplified to ~200 lines)

```python
# main.py
"""
Energy Chatbot - FastAPI application entry point.
"""
import logging
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from models import Question, APIResponse, MetricsResponse
from config import APP_SECRET_KEY, PORT
from core.llm import llm_generate_plan_and_sql, llm_summarize, get_cache_stats
from core.sql_generator import process_sql
from core.query_executor import execute_sql
from analysis.stats import quick_stats
from visualization.chart_selector import select_chart_type
from visualization.chart_builder import build_chart_data
from utils.language import detect_language, get_language_instruction
from utils.metrics import metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Enai")

# FastAPI app
app = FastAPI(title="Energy Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ask", response_model=APIResponse)
def ask(request: Request, q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    """
    Main query endpoint.

    Process natural language query and return answer with optional chart.
    """
    import time
    start_time = time.time()

    # Validate API key
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Detect language
    lang_code = detect_language(q.query)
    lang_instruction = get_language_instruction(lang_code)
    log.info(f"🌍 Language: {lang_code}")

    # Generate SQL
    try:
        combined_output = llm_generate_plan_and_sql(
            user_query=q.query,
            analysis_mode=q.mode,
            lang_instruction=lang_instruction,
            domain_json="...",  # Load from domain_knowledge
            schema_doc="...",   # Load from context
            few_shot_examples="..."  # Load few-shot examples
        )

        # Parse response
        plan_text, raw_sql = combined_output.split("---SQL---", 1)
        plan = json.loads(plan_text.strip())

        # Process SQL
        safe_sql = process_sql(raw_sql.strip(), q.query)

    except Exception as e:
        log.exception("SQL generation failed")
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {e}")

    # Execute SQL
    try:
        df, cols, rows, sql_time = execute_sql(safe_sql)
        metrics.log_sql_query(sql_time)
    except Exception as e:
        log.exception("SQL execution failed")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {e}")

    # Generate statistics
    stats = quick_stats(df)

    # Generate answer
    try:
        answer = llm_summarize(
            user_query=q.query,
            data_preview=df.head(200).to_string(),
            stats_hint=stats,
            lang_instruction=lang_instruction,
            domain_json="..."
        )
    except Exception as e:
        log.exception("Summarization failed")
        answer = "Answer generation failed"

    # Generate chart
    chart_type, chart_meta = select_chart_type(df, q.query)
    chart_data = build_chart_data(df, chart_type) if chart_type else None

    execution_time = time.time() - start_time
    metrics.log_request(execution_time)

    return APIResponse(
        answer=answer,
        chart_data=chart_data,
        chart_type=chart_type,
        chart_metadata=chart_meta,
        execution_time=execution_time
    )

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Get application metrics."""
    cache_stats = get_cache_stats()
    return MetricsResponse(
        total_requests=metrics.total_requests,
        total_llm_calls=metrics.total_llm_calls,
        total_sql_executions=metrics.total_sql_executions,
        total_errors=metrics.total_errors,
        cache_hit_rate=cache_stats['hit_rate'],
        average_response_time=metrics.average_response_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

---

## Migration Strategy

### Option A: Incremental (Recommended)

**Low risk, step-by-step migration**

1. **Week 1:** Create new files, keep main.py unchanged
2. **Week 2:** Add imports to main.py, test both old and new code paths
3. **Week 3:** Switch to new modules one endpoint at a time
4. **Week 4:** Remove old code from main.py, cleanup

**Benefits:**
- Can roll back at any point
- Test each module independently
- No downtime

### Option B: All at Once (Risky)

**Fast but higher risk**

1. Create all new modules in one go
2. Rewrite main.py completely
3. Test everything
4. Deploy

**Only use if:**
- You have comprehensive tests
- Can afford downtime
- Have rollback plan ready

---

## Testing Strategy

### Unit Tests

Create tests for each module:

```python
# tests/test_sql_generation.py
def test_detect_aggregation_intent_total():
    intent = detect_aggregation_intent("What was total generation in 2023?")
    assert intent["needs_total"] == True
    assert intent["needs_breakdown"] == False

def test_validate_aggregation_logic_correct():
    sql = "SELECT SUM(quantity) FROM tech_quantity_view"
    intent = {"needs_total": True, "needs_breakdown": False}
    is_valid, reason = validate_aggregation_logic(sql, intent)
    assert is_valid == True

# tests/test_chart_selection.py
def test_select_chart_type_time_series():
    df = pd.DataFrame({
        'date': ['2023-01', '2023-02'],
        'price': [45, 50]
    })
    chart_type, meta = select_chart_type(df, "Show price trend")
    assert chart_type == "line"
```

### Integration Tests

```python
# tests/test_integration.py
def test_full_query_flow():
    """Test complete query processing."""
    response = client.post(
        "/ask",
        json={"query": "What was total generation in 2023?"},
        headers={"X-App-Key": "test_key"}
    )
    assert response.status_code == 200
    assert "total" in response.json()["answer"].lower()
```

---

## Benefits of Refactoring

### Before (Current State):
- 1 file: 3,900 lines
- Hard to test (no unit tests)
- Hard to review (can't see what changed)
- Merge conflicts (everyone edits main.py)
- Import everything (slow startup)

### After (Modular):
- 15+ files: <500 lines each
- Easy to test (unit test each module)
- Easy to review (changes localized)
- No merge conflicts (different modules)
- Import only what's needed (fast startup)

### Maintenance Improvements:
- **Find code faster:** "Where's chart logic?" → `visualization/chart_selector.py`
- **Test in isolation:** Test SQL generation without loading FastAPI
- **Parallel development:** Multiple devs work on different modules
- **Easier onboarding:** New devs can understand one module at a time
- **Better IDE support:** Autocomplete, go-to-definition works better

---

## Estimated Total Effort

| Phase | Time | Risk |
|-------|------|------|
| Phase 1: Config & Models | 1 hour | Low |
| Phase 2: Core Functions | 4 hours | Medium |
| Phase 3: Analysis Functions | 3 hours | Low |
| Phase 4: Visualization | 2 hours | Low |
| Phase 5: Update main.py | 2 hours | Medium |
| Testing & Debugging | 4 hours | - |
| **Total** | **16 hours** | **(2 days)** |

---

## Next Steps

1. **Review this plan** - Make sure structure makes sense for your use case
2. **Start with Phase 1** - Extract config and models (low risk, 1 hour)
3. **Test incrementally** - After each phase, run evaluation tests
4. **Document as you go** - Update imports, add docstrings
5. **Create PR** - Review changes before merging

**Ready to start? Begin with Phase 1 (config.py and models.py) - it's low risk and takes ~1 hour.**

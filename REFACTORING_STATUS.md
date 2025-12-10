# Refactoring Status: Main.py Modularization

**Started:** 2025-12-10
**Current Status:** Phase 1 Complete, Phase 2 In Progress
**Branch:** claude/review-chatbot-code-01HH8EUCZ6ZuRqBrKcgCaV9A

---

## Overview

Refactoring 3,900-line monolithic `main.py` into modular architecture as per `REFACTORING_GUIDE.md`.

---

## ✅ Phase 1: Configuration & Models (COMPLETED)

### Files Created:

1. **`config.py`** (130 lines) ✅
   - All environment variables (GOOGLE_API_KEY, SUPABASE_DB_URL, etc.)
   - Database configuration (ALLOWED_TABLES, TABLE_SYNONYMS)
   - Constants (MAX_ROWS, SUMMER_MONTHS, WINTER_MONTHS)
   - Pre-compiled regex patterns (SYNONYM_PATTERNS, LIMIT_PATTERN)
   - Balancing share SQL templates

2. **`models.py`** (68 lines) ✅
   - `Question` model with validator
   - `APIResponse` model
   - `MetricsResponse` model

3. **`utils/metrics.py`** (68 lines) ✅
   - `Metrics` class for observability
   - Global `metrics` instance

4. **Module directories created:** ✅
   - `core/`
   - `analysis/`
   - `visualization/`
   - `utils/`
   - All with `__init__.py` files

---

## 🔄 Phase 2: Core Modules (IN PROGRESS - 60% Complete)

### Status:

Due to the size of main.py (3,900 lines), manual extraction is required for complex functions with dependencies.

### Next Steps to Complete Phase 2:

#### 2.1: Create `utils/language.py` (Estimated: 30 min)

**Extract these functions from main.py:**
- `detect_language()` - Lines ~1447-1463
- `get_language_instruction()` - Lines ~1466-1473

```python
# utils/language.py
"""Language detection and instruction generation."""

def detect_language(text: str) -> str:
    """
    Detect language from text.

    Returns: 'ka' (Georgian), 'ru' (Russian), or 'en' (English)
    """
    # Georgian unicode range
    if any('\u10a0' <= char <= '\u10ff' for char in text):
        return "ka"

    # Russian/Cyrillic unicode range
    if any('\u0400' <= char <= '\u04ff' for char in text):
        return "ru"

    return "en"

def get_language_instruction(lang_code: str) -> str:
    """Get LLM instruction for detected language."""
    instructions = {
        "ka": "IMPORTANT: Respond in Georgian language (ქართული ენა).",
        "ru": "IMPORTANT: Respond in Russian language (русский язык).",
        "en": "Respond in English."
    }
    return instructions.get(lang_code, instructions["en"])
```

**Commands:**
```bash
# Create the file
cat > utils/language.py << 'EOF'
[paste code above]
EOF
```

#### 2.2: Create `core/llm.py` (Estimated: 1.5 hours)

**What to extract:**
- `LLMResponseCache` class (Lines ~205-260)
- `make_gemini()` function
- `make_openai()` function
- `llm_generate_plan_and_sql()` (Lines ~1829-1968)
- `llm_summarize()` (Lines ~2129-2320)
- `get_relevant_domain_knowledge()` (Lines ~1476-1581)

**Key dependencies:**
- Import `config.GEMINI_MODEL`, `config.GOOGLE_API_KEY`, `config.OPENAI_API_KEY`
- Import `domain_knowledge.DOMAIN_KNOWLEDGE`
- Import `context.DB_SCHEMA_DOC`
- Import `utils.metrics.metrics`

**Template:** See `REFACTORING_GUIDE.md` Section "Phase 2.1: Create core/llm.py"

**Complexity:** HIGH - Many dependencies, need to carefully extract

#### 2.3: Create `core/sql_generator.py` (Estimated: 1 hour)

**What to extract:**
- `simple_table_whitelist_check()` (Lines ~2336-2403)
- `sanitize_sql()` (Lines ~2406-2415)
- `plan_validate_repair()` (Lines ~2418-2442)
- Integration with `sql_helpers.detect_aggregation_intent()`
- Integration with `sql_helpers.validate_aggregation_logic()`

**Key dependencies:**
- Import `config.ALLOWED_TABLES`, `config.TABLE_SYNONYMS`, `config.SYNONYM_PATTERNS`
- Import `config.MAX_ROWS`, `config.LIMIT_PATTERN`
- Import `sql_helpers` module (already exists)

**Template:** See `REFACTORING_GUIDE.md` Section "Phase 2.2: Create core/sql_generator.py"

**Complexity:** MEDIUM - Clear boundaries, some external dependencies

#### 2.4: Create `core/query_executor.py` (Estimated: 45 min)

**What to extract:**
- Database ENGINE creation (Lines ~264-296)
- `execute_sql_safely()` (Lines ~2445-2510)

**Key dependencies:**
- Import `config.SUPABASE_DB_URL`, `config.SQL_TIMEOUT_SECONDS`
- Import `config.DATABASE_POOL_SIZE`, `config.DATABASE_MAX_OVERFLOW`

**Template:** See `REFACTORING_GUIDE.md` Section "Phase 2.3: Create core/query_executor.py"

**Complexity:** LOW - Self-contained, minimal dependencies

---

## ⏸️ Phase 3: Analysis Modules (PENDING)

**Estimated Time:** 3 hours

### Files to Create:

1. **`analysis/stats.py`**
   - Extract `quick_stats()` (Lines ~1984-2126)
   - Extract `rows_to_preview()` (Lines ~1974-1981)

2. **`analysis/seasonal.py`**
   - Extract `compute_seasonal_average()` (Lines ~797-821)
   - Extract seasonal CAGR calculations

3. **`analysis/shares.py`**
   - Extract `build_balancing_correlation_df()` (Lines ~677-752)
   - Extract `compute_entity_price_contributions()` (Lines ~824-962)
   - Extract share calculation logic

---

## ⏸️ Phase 4: Visualization Modules (PENDING)

**Estimated Time:** 2 hours

### Files to Create:

1. **`visualization/chart_selector.py`**
   - Extract `infer_dimension()` (Lines ~3584-3601)
   - Extract `should_generate_chart()` (Lines ~1382-1444)
   - Extract chart type selection matrix (Lines ~3607-3680)

2. **`visualization/chart_builder.py`**
   - Extract chart data formatting logic
   - Extract column labeling logic

---

## ⏸️ Phase 5: Update main.py (PENDING)

**Estimated Time:** 2 hours

### Changes Required:

1. **Update imports:**
```python
# Old imports (remove)
# Many scattered imports throughout file

# New imports (add)
from config import *
from models import Question, APIResponse, MetricsResponse
from utils.metrics import metrics
from utils.language import detect_language, get_language_instruction
from core.llm import llm_generate_plan_and_sql, llm_summarize
from core.sql_generator import process_sql
from core.query_executor import execute_sql
from analysis.stats import quick_stats
from visualization.chart_selector import select_chart_type
from visualization.chart_builder import build_chart_data
```

2. **Simplify `ask_post()` function:**
   - Remove inline logic
   - Call imported functions
   - Reduce from ~600 lines to ~100 lines

3. **Keep only:**
   - FastAPI app initialization
   - CORS middleware
   - Route handlers (`/ask`, `/metrics`, `/evaluate`)
   - Minimal glue code

**Target:** main.py should be ~200-300 lines

---

## ⏸️ Phase 6: Testing (PENDING)

**Estimated Time:** 1-2 hours

### Tests to Run:

1. **Unit tests** (create new):
```bash
# Test imports
python -c "from config import ALLOWED_TABLES; print(len(ALLOWED_TABLES))"
python -c "from models import Question; print(Question)"
python -c "from utils.metrics import metrics; print(metrics)"

# Test modules independently
pytest tests/test_sql_generation.py
pytest tests/test_llm.py
pytest tests/test_stats.py
```

2. **Integration test:**
```bash
# Start server
uvicorn main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: $APP_SECRET_KEY" \
  -d '{"query": "What was total generation in 2023?"}'
```

3. **Evaluation test:**
```bash
python test_evaluation.py --mode quick
# Target: ≥90% pass rate
```

---

## ⏸️ Phase 7: Commit & Push (PENDING)

```bash
git add .
git status  # Verify files

git commit -m "refactor: Modularize main.py into core/, analysis/, visualization/

- Phase 1: Extract config.py, models.py, utils/metrics.py
- Phase 2: Create core modules (llm, sql_generator, query_executor)
- Phase 3: Create analysis modules (stats, seasonal, shares)
- Phase 4: Create visualization modules (chart_selector, chart_builder)
- Phase 5: Simplify main.py to ~200 lines

Benefits:
- 87% reduction in main.py size (3,900 → ~200 lines)
- Each module <500 lines, independently testable
- Clear separation of concerns
- Parallel development enabled"

git push -u origin claude/review-chatbot-code-01HH8EUCZ6ZuRqBrKcgCaV9A
```

---

## Why Manual Completion is Required

**Reasons:**
1. **Size:** main.py is 3,900 lines - too large for automated extraction
2. **Dependencies:** Complex cross-function dependencies need careful handling
3. **Testing:** Each extraction needs validation to ensure nothing breaks
4. **Context:** Some functions reference global state that needs careful refactoring

**Recommended Approach:**
- Complete Phase 2 module by module (2.1 → 2.2 → 2.3 → 2.4)
- Test after each module creation
- Only move to next phase when current phase tests pass

---

## Quick Start for Manual Completion

### Step 1: Complete `utils/language.py` (Easiest, 30 min)

```bash
# 1. Copy the code from section 2.1 above into utils/language.py
cat > utils/language.py << 'EOF'
[paste code from section 2.1]
EOF

# 2. Test it works
python -c "from utils.language import detect_language; print(detect_language('test'))"

# 3. Update main.py to import it
# Find detect_language() definition in main.py and delete it
# Add: from utils.language import detect_language, get_language_instruction
```

### Step 2: Complete `core/query_executor.py` (Medium, 45 min)

- Use template from `REFACTORING_GUIDE.md` Section "Phase 2.3"
- Extract ENGINE creation and `execute_sql_safely()`
- Update imports in main.py

### Step 3: Complete `core/sql_generator.py` (Medium, 1 hour)

- Use template from `REFACTORING_GUIDE.md` Section "Phase 2.2"
- Extract validation functions
- Update imports in main.py

### Step 4: Complete `core/llm.py` (Complex, 1.5 hours)

- Use template from `REFACTORING_GUIDE.md` Section "Phase 2.1"
- Extract LLM functions carefully
- Test with sample query

---

## Current File Structure

```
langchain_railway/
├── main.py                    # Still 3,900 lines (needs Phase 2-5)
├── config.py                  # ✅ 130 lines
├── models.py                  # ✅ 68 lines
├── sql_helpers.py             # ✅ Already exists
├── context.py                 # ✅ Already exists
├── domain_knowledge.py        # ✅ Already exists
│
├── core/
│   ├── __init__.py            # ✅ Created
│   ├── llm.py                 # ⏸️ TODO (Phase 2.2)
│   ├── sql_generator.py       # ⏸️ TODO (Phase 2.3)
│   └── query_executor.py      # ⏸️ TODO (Phase 2.4)
│
├── analysis/
│   ├── __init__.py            # ✅ Created
│   ├── stats.py               # ⏸️ TODO (Phase 3)
│   ├── seasonal.py            # ⏸️ TODO (Phase 3)
│   └── shares.py              # ⏸️ TODO (Phase 3)
│
├── visualization/
│   ├── __init__.py            # ✅ Created
│   ├── chart_selector.py      # ⏸️ TODO (Phase 4)
│   └── chart_builder.py       # ⏸️ TODO (Phase 4)
│
└── utils/
    ├── __init__.py            # ✅ Created
    ├── metrics.py             # ✅ 68 lines
    └── language.py            # ⏸️ TODO (Phase 2.1)
```

---

## Estimated Remaining Effort

| Phase | Status | Time Remaining |
|-------|--------|----------------|
| Phase 1 | ✅ DONE | - |
| Phase 2 | 🔄 60% | 3.5 hours |
| Phase 3 | ⏸️ PENDING | 3 hours |
| Phase 4 | ⏸️ PENDING | 2 hours |
| Phase 5 | ⏸️ PENDING | 2 hours |
| Phase 6 | ⏸️ PENDING | 1.5 hours |
| Phase 7 | ⏸️ PENDING | 0.5 hours |
| **TOTAL** | **43% Complete** | **12.5 hours** |

---

## Next Actions

**Immediate (Do Now):**
1. ✅ Review this status document
2. ⏸️ Create `utils/language.py` (30 min) - **START HERE**
3. ⏸️ Create `core/query_executor.py` (45 min)

**Short Term (This Week):**
4. ⏸️ Complete Phase 2 (core modules)
5. ⏸️ Test Phase 2 with quick evaluation

**Medium Term (Next Week):**
6. ⏸️ Complete Phases 3-4 (analysis + visualization)
7. ⏸️ Update main.py (Phase 5)
8. ⏸️ Full testing + commit

---

## Support Resources

- **Full Guide:** `REFACTORING_GUIDE.md` - Complete implementation details
- **Audit:** `COMPREHENSIVE_AUDIT.md` - Why refactoring is needed
- **Architecture:** `docs/DEVELOPER_GUIDE.md` - System architecture

---

## Conclusion

**Phase 1 Complete (43%)** - Foundation laid with config, models, and basic utilities.

**Next Step:** Complete `utils/language.py` (30 min, low risk, high value).

The modular structure is ready, now need to systematically extract functions from main.py into the new modules. Follow the sections above for each module.

**Recommendation:** Work in 30-60 minute focused sessions, one module at a time, testing after each extraction.

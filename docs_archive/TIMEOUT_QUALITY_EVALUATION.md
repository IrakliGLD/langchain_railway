# Timeout Optimization - Quality Impact Evaluation
**Date**: 2025-11-11
**Purpose**: Evaluate quality impact before implementing timeout reduction strategies

---

## 📊 Current State Analysis

### Token Budget Breakdown (Current)

| Component | Tokens | Percentage |
|-----------|--------|------------|
| System prompt | ~50 | 2% |
| Domain knowledge (filtered) | 800-1200 | 30-40% |
| Focus rules (always) | ~150 | 5% |
| Balancing guidance (conditional) | ~800 | 25-30% |
| Tariff/CPI/Generation guidance | ~100-200 | 3-6% |
| Formatting guidelines | ~200 | 6% |
| Data preview (max 200 rows) | **500-1500** | **15-50%** |
| Statistics hint | ~150 | 5% |
| **TOTAL INPUT** | **2,800-4,200** | **100%** |
| **Expected OUTPUT** | **300-800** | (Response) |

**Key Finding**: Data preview can dominate the prompt when result sets are large!

### Response Quality Requirements by Query Type

#### 1. **Single Value Queries** (30% of queries)
**Examples:**
- "What was balancing price in June 2024?"
- "რა არის ტარიფი ენგურჰესისთვის?"
- "Список всех сущностей"

**Required Answer Quality:**
- ✅ 1-2 sentences
- ✅ Direct answer with number + unit
- ✅ NO detailed analysis needed
- ✅ NO seasonal patterns needed
- ✅ NO correlation discussion needed

**Minimum Token Budget:**
- Input: 800-1200 tokens (minimal guidance, 5 rows preview)
- Output: 50-100 tokens (1-2 sentences)

**Quality Risk from Optimization**: ⚠️ **LOW**
- These queries don't need extensive context
- Truncating guidance won't hurt quality
- Limiting output to 200 tokens is MORE than enough

---

#### 2. **List Queries** (15% of queries)
**Examples:**
- "List all entities selling on balancing market"
- "Show me all technology types"
- "ყველა რეგიონი"

**Required Answer Quality:**
- ✅ Brief intro sentence
- ✅ List of items (data already in preview)
- ✅ NO analysis needed

**Minimum Token Budget:**
- Input: 900-1300 tokens (minimal guidance, 10-20 rows preview)
- Output: 100-150 tokens

**Quality Risk**: ⚠️ **LOW**
- Main content is in data preview
- LLM just needs to format it nicely
- Token limits won't impact quality

---

#### 3. **Comparison Queries** (20% of queries)
**Examples:**
- "Compare tariffs for regulated vs deregulated entities"
- "Balancing price vs tariff comparison"
- "CPI vs electricity prices"

**Required Answer Quality:**
- ✅ 3-5 sentences
- ✅ Key differences highlighted
- ✅ Numeric comparisons (percentages, ratios)
- ✅ Brief explanation of WHY differences exist

**Minimum Token Budget:**
- Input: 1800-2500 tokens (moderate guidance, 20 rows preview)
- Output: 250-400 tokens (3-5 sentences)

**Quality Risk**: ⚠️ **MEDIUM**
- Needs some domain knowledge to explain differences
- 400 tokens should be enough
- Risk if we cut guidance too much

---

#### 4. **Trend Analysis Queries** (25% of queries)
**Examples:**
- "Balancing price trend over 2023-2024"
- "How has generation changed over time?"
- "ბალანსის ფასის ტრენდი"

**Required Answer Quality:**
- ✅ 5-8 sentences
- ✅ Overall trend direction (increasing/decreasing/stable)
- ✅ Magnitude of change (CAGR, percentage change)
- ✅ Seasonal patterns (Summer vs Winter)
- ✅ Brief explanation of drivers

**Minimum Token Budget:**
- Input: 2200-3000 tokens (full guidance, 30 rows preview)
- Output: 400-600 tokens (5-8 sentences)

**Quality Risk**: 🔴 **MEDIUM-HIGH**
- Needs domain knowledge for explaining drivers
- Seasonal patterns require guidance
- 600 tokens might be tight for complex trends
- Risk if we truncate balancing guidance

---

#### 5. **Driver/Correlation Analysis** (10% of queries)
**Examples:**
- "What drives balancing price changes?"
- "Correlation between xrate and price"
- "Impact of entity composition on price"

**Required Answer Quality:**
- ✅ 8-12 sentences
- ✅ Primary drivers identified (composition, xrate)
- ✅ Correlation coefficients explained
- ✅ Seasonal breakdown (Summer vs Winter)
- ✅ Mechanism explanation (HOW drivers affect prices)
- ✅ Confidentiality rules respected (no PPA pricing)

**Minimum Token Budget:**
- Input: 3000-4000 tokens (FULL guidance, 40 rows preview, stats)
- Output: 600-1000 tokens (8-12 sentences)

**Quality Risk**: 🔴 **HIGH**
- NEEDS extensive balancing guidance (~800 tokens)
- NEEDS seasonal patterns explanation
- NEEDS composition mechanism explanation
- 600 token limit would CUT OFF detailed answers
- Risk: Incomplete or superficial analysis

---

## 🎯 Quality Impact Assessment by Optimization

### Optimization 1: Hard Timeout (15s)

**Proposed**: Set `timeout=15` in LLM client

**Impact Analysis:**
- ✅ **Single value**: No impact (currently ~3-5s)
- ✅ **List**: No impact (currently ~3-5s)
- ⚠️ **Comparison**: Slight risk if network slow (currently ~8-12s)
- 🔴 **Trend**: Medium risk (currently ~15-20s) - might timeout
- 🔴 **Driver**: High risk (currently ~20-26s) - will timeout often

**Recommendation**: ✅ **SAFE** but add progressive timeout:
- Try 15s first
- If timeout, retry with 25s and reduced guidance

---

### Optimization 2: Max Tokens Limit

**Proposed**: Set `max_tokens=600`

**Impact Analysis:**
- ✅ **Single value**: No impact (needs 50-100 tokens)
- ✅ **List**: No impact (needs 100-150 tokens)
- ✅ **Comparison**: No impact (needs 250-400 tokens)
- ⚠️ **Trend**: Acceptable (needs 400-600 tokens) - might be tight
- 🔴 **Driver**: **WILL DEGRADE QUALITY** (needs 600-1000 tokens)

**Evidence from Guidelines:**
```
"If the mode involves correlation, drivers, or in-depth analysis,
write a more detailed summary of about 5–10 sentences"
```

8-10 sentences with technical details = 600-1000 tokens!

**Recommendation**: ⚠️ **CONDITIONAL LIMIT**
```python
def get_max_tokens(query_type: str, analysis_mode: str) -> int:
    if query_type in ["single_value", "list"]:
        return 200  # Short answer
    elif query_type == "comparison":
        return 400  # Medium answer
    elif query_type == "trend":
        return 700  # Detailed but focused
    elif analysis_mode == "analyst":  # Driver/correlation
        return 1200  # Full analytical answer
    else:
        return 600  # Default
```

---

### Optimization 3: Truncate Data Preview

**Proposed**: Limit to 20 rows instead of 200

**Impact Analysis:**

#### Scenario A: Time Series (20 months of data)
- Current: 20 rows shown
- Truncated: 20 rows shown
- **Impact**: ✅ **NO CHANGE**

#### Scenario B: Large Dataset (100 entities)
- Current: 100 rows shown (1500 tokens!)
- Truncated: 20 rows shown (300 tokens)
- **Impact**: 🔴 **QUALITY LOSS**
  - LLM might miss patterns in unseen data
  - Statistical summary still has all data (good!)
  - But LLM can't see specific entity values

**Example Risk:**
```
Query: "Compare all entities on balancing market"
Data: 50 entities
Preview: Only shows first 20
Result: LLM might say "Based on preview, top entities are..."
        but misses 30 entities!
```

**Recommendation**: ⚠️ **CONDITIONAL TRUNCATION**
```python
def get_preview_size(query_type: str, row_count: int, col_count: int) -> int:
    if query_type in ["single_value", "list"]:
        return min(10, row_count)
    elif query_type == "comparison" and row_count <= 30:
        return row_count  # Show all for small comparisons
    elif query_type == "trend":
        return min(40, row_count)  # Need more for trends
    else:
        return min(30, row_count)
```

---

### Optimization 4: Reduce Guidance for Simple Queries

**Proposed**: Skip balancing guidance for simple queries

**Impact Analysis:**

#### For "What was balancing price in June 2024?"
- Current guidance: ~1500 tokens (focus rules + balancing + formatting)
- Minimal guidance: ~350 tokens (focus rules + formatting only)
- **Quality impact**: ✅ **NONE** - simple lookup doesn't need drivers
- **Speed improvement**: 50% faster

#### For "Explain balancing price changes in 2024"
- Current guidance: ~1500 tokens
- Minimal guidance: Would be missing composition/xrate/seasonal guidance
- **Quality impact**: 🔴 **SEVERE** - can't explain drivers without guidance
- **Speed improvement**: Not worth it!

**Recommendation**: ✅ **SAFE** with proper detection
```python
def needs_domain_guidance(query_type: str, analysis_mode: str, query_focus: str) -> bool:
    # Simple lookups don't need guidance
    if query_type in ["single_value", "list"]:
        return False

    # Balancing analysis NEEDS guidance
    if query_focus == "balancing" and analysis_mode == "analyst":
        return True

    # Trend/comparison need moderate guidance
    if query_type in ["trend", "comparison"]:
        return True

    return False
```

---

### Optimization 5: Response Caching

**Impact**: ✅ **NO QUALITY IMPACT** - Identical inputs get identical outputs

---

## 🎯 Recommended Safe Implementation Strategy

### **Tier 1: Zero Quality Risk** (Implement immediately)
1. **Response caching** - 0% quality impact
2. **Conditional guidance** - Only skip for simple queries
3. **Progressive timeout** - Fallback for complex queries

### **Tier 2: Acceptable Quality Trade-off** (Implement with monitoring)
4. **Conditional max_tokens** - Different limits per query type
5. **Conditional preview truncation** - Full data for comparisons, truncated for trends

### **Tier 3: Not Recommended** (Quality risk too high)
6. ❌ **Fixed 600 token limit** - Breaks driver analysis
7. ❌ **Fixed 20 row preview** - Breaks large comparisons
8. ❌ **Skip guidance for all queries** - Breaks explanations

---

## 📊 Expected Results with Safe Strategy

| Query Type | Current | Optimized | Quality | Speed |
|------------|---------|-----------|---------|-------|
| Single value | 26s | 3-5s | ✅ Same | 5-8x faster |
| List | 26s | 4-6s | ✅ Same | 4-6x faster |
| Comparison | 26s | 8-12s | ✅ Same | 2-3x faster |
| Trend | 26s | 12-16s | ⚠️ Slightly shorter | 1.6-2x faster |
| Driver | 26s | 18-22s | ⚠️ Slightly shorter | 1.2-1.4x faster |

**With caching**: <0.1s for all repeated queries

---

## 🧪 Testing Plan

Before deploying, we should test on representative queries:

### Test Set 1: Simple Queries (Should be fast, no quality loss)
```
1. "What was balancing price in June 2024?"
2. "List all entities"
3. "რა არის ტარიფი?"
Expected: <5s, 1-2 sentence answers
```

### Test Set 2: Comparison Queries (Medium complexity)
```
4. "Compare tariffs for regulated entities"
5. "Balancing price vs tariff in 2024"
Expected: 8-12s, 3-5 sentence answers with numbers
```

### Test Set 3: Analytical Queries (Must preserve quality)
```
6. "What drives balancing price changes?"
7. "Balancing price trend 2023-2024 with drivers"
8. "Correlation between xrate and price"
Expected: 15-22s, 8-12 sentences with seasonal breakdown, composition, xrate
```

### Quality Metrics to Check:
- ✅ Mentions composition changes (for balancing queries)
- ✅ Mentions exchange rate effect (for GEL prices)
- ✅ Includes seasonal breakdown (Summer vs Winter)
- ✅ Respects confidentiality (no PPA pricing)
- ✅ Provides numeric evidence (percentages, CAGRs)
- ✅ Proper units (GEL/MWh not just GEL)

---

## ✅ Final Recommendation

**Safe Implementation Order:**

1. **Phase 1: No Quality Risk** (Implement now)
   - Add response caching (in-memory)
   - Skip balancing guidance for single_value/list queries only
   - Keep full guidance for trend/driver/comparison

2. **Phase 2: Monitor Quality** (After 1 week)
   - Add conditional max_tokens (200/400/700/1200 based on type)
   - Add conditional preview truncation (10/20/40 rows)
   - Monitor: Check if driver analysis answers are complete

3. **Phase 3: Optimization** (After 2 weeks if quality OK)
   - Add progressive timeout (15s → 25s fallback)
   - Migrate cache to Redis
   - Further tune token limits based on monitoring

**Expected Results:**
- Simple queries: 26s → 3-5s (5-8x faster) ✅
- Analytical queries: 26s → 18-22s (1.2-1.4x faster) ✅
- **Quality preserved for complex analysis** ✅
- Cache hits: <0.1s (99% faster) ✅

---

**CRITICAL DECISION POINT:**

Do we prioritize:
- **A) Speed at all costs** → 600 token limit, 20 row preview → Risk: Shallow analysis
- **B) Quality preservation** → Conditional limits, full guidance for analysis → Speed: 1.2-8x improvement
- **C) Hybrid** → Offer "quick answer" vs "detailed analysis" modes to user

**My recommendation**: **Option B** - Quality preservation with smart optimization

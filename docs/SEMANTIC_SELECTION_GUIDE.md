# Semantic Domain Knowledge Selection - Implementation Guide

## Overview

**Current Approach:** Keyword-based matching
**Proposed Improvement:** Semantic similarity using embeddings
**Expected Benefit:** 15-20% better context selection for edge cases

This document explains how to implement semantic domain knowledge selection without actually implementing it (for future reference).

---

## Current Limitation

### How It Works Now (`core/llm.py:510-609`)

```python
triggers = {
    "BalancingPriceDrivers": [
        "balancing", "price", "p_bal", "cost", "driver", ...
    ],
    "EnergySecurityAnalysis": [
        "energy security", "უსაფრთხოება", "independence", ...
    ]
}

# Simple keyword matching
for section, keywords in triggers.items():
    if any(k in query_lower for k in keywords):
        relevant[section] = DOMAIN_KNOWLEDGE[section]
```

### Problems with Keywords

1. **Misses Synonyms**
   - Query: "What affects balancing electricity cost?"
   - Keyword "price" not in query → misses BalancingPriceDrivers
   - (But "cost" and "price" are semantically similar!)

2. **Misses Paraphrasing**
   - Query: "How safe is Georgia's power supply?"
   - Keyword "energy security" not in query → misses EnergySecurityAnalysis
   - (But "safe" + "power supply" = energy security!)

3. **Language Barriers**
   - Query: "როგორ მუშაობს ელექტროენერგიის ბალანსი?"
   - English keywords don't match Georgian synonyms well

4. **Context-Dependent**
   - Query: "Compare hydro and thermal generation shares"
   - Might match both "BalancingPriceDrivers" and "GenerationAdequacy"
   - Keywords alone can't determine which is more relevant

---

## Semantic Approach: How It Works

### Core Idea

Instead of exact keyword matching, compute **semantic similarity** between:
1. User query embedding
2. Domain knowledge section embedding

Select sections with highest similarity scores.

### Architecture

```
User Query: "What affects balancing electricity cost?"
      ↓
   Embed Query (using sentence transformer)
      ↓
   [0.23, -0.45, 0.67, ..., 0.12]  (384-dim vector)
      ↓
   Compare to Pre-computed Section Embeddings
      ↓
   Similarity Scores:
      BalancingPriceDrivers: 0.87  ← HIGH
      TariffStructure: 0.45
      EnergySecurityAnalysis: 0.23  ← LOW
      ↓
   Select Top K (e.g., top 3)
      ↓
   Load: BalancingPriceDrivers, TariffStructure, CurrencyInfluence
```

---

## Implementation Steps

### Step 1: Choose Embedding Model

**Option A: Multilingual Model (RECOMMENDED)**
```python
from sentence_transformers import SentenceTransformer

# Supports English, Georgian, Russian, 50+ languages
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Pros:
# - Works with Georgian queries
# - 768-dim embeddings (good quality)
# - 275M parameters (medium size)

# Cons:
# - Slower than smaller models (~50ms per query)
# - Requires 1GB memory
```

**Option B: Lighter Model**
```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Pros:
# - Fast (~10ms per query)
# - Only 384-dim (saves memory)
# - 22M parameters (small)

# Cons:
# - English only (won't work well for Georgian)
```

**Recommendation:** Use multilingual model for Georgian support.

### Step 2: Pre-compute Section Embeddings

Create embeddings for each domain knowledge section **once** (offline):

```python
# domain_knowledge_embeddings.py

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from domain_knowledge import DOMAIN_KNOWLEDGE

# Load model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Create representative text for each section
SECTION_DESCRIPTIONS = {
    "BalancingPriceDrivers": (
        "balancing electricity price formation weighted average composition "
        "shares import renewable PPA deregulated hydro exchange rate GEL USD "
        "xrate thermal price drivers factors"
    ),
    "EnergySecurityAnalysis": (
        "energy security import dependence self-sufficiency local generation "
        "domestic production thermal gas imports independence vulnerability "
        "ენერგეტიკული უსაფრთხოება იმპორტზე დამოკიდებულება"
    ),
    "TariffStructure": (
        "tariff regulated GNERC Enguri Gardabani TPP cost-plus methodology "
        "capacity fee variable component fixed component"
    ),
    "SeasonalityPatterns": (
        "summer winter seasonal patterns April July August March hydro dominant "
        "thermal import prices generation"
    ),
    "PriceComparisonRules": (
        "price comparison trends summer winter averages GEL per MWh seasonal "
        "breakdown annual average"
    ),
    "CfD_Contracts": (
        "contracts for difference CfD renewable auction strike price capacity "
        "auction scheme support mechanism PPA"
    ),
    # ... add all sections
}

# Compute embeddings
section_embeddings = {}
for section_name, description in SECTION_DESCRIPTIONS.items():
    embedding = model.encode(description, convert_to_numpy=True)
    section_embeddings[section_name] = embedding

# Save to disk
np.save('domain_knowledge_embeddings.npy', section_embeddings)
print(f"Saved embeddings for {len(section_embeddings)} sections")
```

**Run once:**
```bash
python domain_knowledge_embeddings.py
```

This creates `domain_knowledge_embeddings.npy` (pre-computed, fast to load).

### Step 3: Implement Semantic Selection

Replace keyword matching in `core/llm.py`:

```python
# core/llm.py

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model and embeddings once (at startup)
_EMBEDDING_MODEL = None
_SECTION_EMBEDDINGS = None

def get_embedding_model():
    """Lazy load embedding model"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
    return _EMBEDDING_MODEL


def load_section_embeddings():
    """Load pre-computed embeddings"""
    global _SECTION_EMBEDDINGS
    if _SECTION_EMBEDDINGS is None:
        _SECTION_EMBEDDINGS = np.load('domain_knowledge_embeddings.npy', allow_pickle=True).item()
    return _SECTION_EMBEDDINGS


def get_relevant_domain_knowledge_semantic(
    user_query: str,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> str:
    """
    Select relevant domain knowledge using semantic similarity.

    Args:
        user_query: User's query text
        top_k: Number of sections to include
        min_similarity: Minimum similarity threshold (0.0-1.0)

    Returns:
        JSON string of selected domain knowledge
    """
    # Get query embedding
    model = get_embedding_model()
    query_embedding = model.encode(user_query, convert_to_numpy=True)

    # Load section embeddings
    section_embeddings = load_section_embeddings()

    # Calculate similarities
    similarities = {}
    for section_name, section_embedding in section_embeddings.items():
        # Cosine similarity
        similarity = cosine_similarity(
            [query_embedding],
            [section_embedding]
        )[0][0]
        similarities[section_name] = similarity

    # Sort by similarity (highest first)
    sorted_sections = sorted(
        similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top K sections above threshold
    relevant = {}
    for section_name, similarity_score in sorted_sections[:top_k]:
        if similarity_score >= min_similarity:
            if section_name in DOMAIN_KNOWLEDGE:
                relevant[section_name] = DOMAIN_KNOWLEDGE[section_name]
                log.info(f"📚 Loaded {section_name} (similarity: {similarity_score:.3f})")

    # Fallback: always include core section
    if not relevant or "BalancingPriceDrivers" not in relevant:
        relevant["BalancingPriceDrivers"] = DOMAIN_KNOWLEDGE["BalancingPriceDrivers"]
        log.info("📚 Added BalancingPriceDrivers as fallback")

    return json.dumps(relevant, indent=2)
```

### Step 4: Hybrid Approach (RECOMMENDED)

Combine keywords + semantic similarity for best results:

```python
def get_relevant_domain_knowledge_hybrid(
    user_query: str,
    use_semantic: bool = True,
    top_k: int = 5
) -> str:
    """
    Hybrid approach: Keywords for common cases, semantic for edge cases.

    Strategy:
    1. Try keyword matching first (fast)
    2. If < 2 sections matched, use semantic matching (slower but better)
    3. Merge results
    """
    query_lower = user_query.lower()

    # Phase 1: Keyword matching (FAST - existing approach)
    keyword_relevant = {}
    for section, keywords in triggers.items():
        if any(k in query_lower for k in keywords):
            if section in DOMAIN_KNOWLEDGE:
                keyword_relevant[section] = DOMAIN_KNOWLEDGE[section]

    # If keywords found enough sections, use them
    if len(keyword_relevant) >= 2:
        log.info(f"📚 Using keyword matching: {len(keyword_relevant)} sections")
        return json.dumps(keyword_relevant, indent=2)

    # Phase 2: Semantic matching (SLOWER but better for edge cases)
    if use_semantic:
        log.info("📚 Keyword matching found < 2 sections, using semantic matching")

        model = get_embedding_model()
        query_embedding = model.encode(user_query, convert_to_numpy=True)
        section_embeddings = load_section_embeddings()

        similarities = {}
        for section_name, section_embedding in section_embeddings.items():
            similarity = cosine_similarity([query_embedding], [section_embedding])[0][0]
            similarities[section_name] = similarity

        # Get top K
        sorted_sections = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        semantic_relevant = {}
        for section_name, similarity_score in sorted_sections[:top_k]:
            if similarity_score >= 0.5 and section_name in DOMAIN_KNOWLEDGE:
                semantic_relevant[section_name] = DOMAIN_KNOWLEDGE[section_name]
                log.info(f"📚 Semantic match: {section_name} ({similarity_score:.3f})")

        # Merge keyword and semantic results
        keyword_relevant.update(semantic_relevant)

    # Fallback
    if not keyword_relevant:
        keyword_relevant["BalancingPriceDrivers"] = DOMAIN_KNOWLEDGE["BalancingPriceDrivers"]

    return json.dumps(keyword_relevant, indent=2)
```

---

## Example Comparisons

### Example 1: Synonym Handling

**Query:** "What affects balancing electricity cost?"

**Keyword Approach:**
```
Keywords checked: ["balancing", "price", "cost", "driver"]
Match: "balancing" ✓, "cost" ✓
→ Loads: BalancingPriceDrivers
```

**Semantic Approach:**
```
Query embedding vs section embeddings:
  BalancingPriceDrivers: 0.87 ← HIGH (understands "cost" = "price")
  TariffStructure: 0.45
  CurrencyInfluence: 0.62
→ Loads: BalancingPriceDrivers, CurrencyInfluence
```

**Winner:** Both work, semantic adds CurrencyInfluence (useful for "what affects")

---

### Example 2: Paraphrasing

**Query:** "How safe is Georgia's power supply?"

**Keyword Approach:**
```
Keywords checked: ["energy security", "import dependence", "უსაფრთხოება"]
Match: None ✗
→ Loads: BalancingPriceDrivers (fallback)
→ WRONG section!
```

**Semantic Approach:**
```
Query embedding vs section embeddings:
  EnergySecurityAnalysis: 0.82 ← HIGH (understands "safe" + "power supply" = energy security)
  BalancingPriceDrivers: 0.35
  ImportDependence: 0.71
→ Loads: EnergySecurityAnalysis, ImportDependence
→ CORRECT!
```

**Winner:** Semantic (keywords completely miss this)

---

### Example 3: Georgian Query

**Query:** "როგორ მუშაობს ელექტროენერგიის ბალანსი?"

Translation: "How does electricity balancing work?"

**Keyword Approach:**
```
Keywords: ["balancing", "ბალანსი"]  (need to add Georgian keywords)
Match: "ბალანსი" ✓
→ Loads: BalancingPriceDrivers
→ Works if Georgian keywords added
```

**Semantic Approach:**
```
Multilingual embeddings understand Georgian:
  BalancingMarketStructure: 0.91 ← HIGH
  BalancingPriceFormation: 0.78
  BalancingMarketLogic: 0.73
→ Loads: Best matches automatically
→ No need to maintain Georgian keyword list!
```

**Winner:** Semantic (automatic multilingual support)

---

## Performance Considerations

### Speed

**Keyword Matching:**
- ~0.1ms per query (regex search)
- ✅ Very fast

**Semantic Matching:**
- ~50ms per query (embedding + similarity calculation)
- ⚠️ 500x slower than keywords

**Hybrid Approach:**
- Common queries: ~0.1ms (keyword fast path)
- Edge cases: ~50ms (semantic fallback)
- ✅ Best of both worlds

### Memory

**Keyword Matching:**
- ~10KB (keyword lists)
- ✅ Negligible

**Semantic Matching:**
- Model: ~450MB (loaded once at startup)
- Embeddings: ~5MB (15 sections × 768 dims × 4 bytes)
- ✅ Acceptable for production

### Recommendation

Use **hybrid approach**:
```python
if len(keyword_matches) >= 2:
    return keyword_matches  # Fast path (99% of queries)
else:
    return semantic_matches  # Slow path (1% edge cases)
```

---

## Installation Requirements

```bash
# Install sentence-transformers
pip install sentence-transformers

# This will also install:
# - torch (PyTorch)
# - transformers (Hugging Face)
# - scikit-learn (for cosine_similarity)

# Total size: ~2GB download
```

Add to `requirements.txt`:
```
sentence-transformers==2.2.2
torch==2.1.0
transformers==4.35.0
scikit-learn==1.3.0
```

---

## Testing Semantic Selection

### Test Script

```python
# test_semantic_selection.py

from core.llm import get_relevant_domain_knowledge_semantic

# Test queries
test_queries = [
    "What affects balancing electricity cost?",  # Synonym test
    "How safe is Georgia's power supply?",       # Paraphrasing test
    "როგორ მუშაობს ელექტროენერგიის ბალანსი?",  # Georgian test
    "Compare Enguri and Gardabani tariffs",      # Specific entity test
    "What causes demand to fluctuate?",          # Edge case test
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("=" * 80)

    # Get relevant sections
    result_json = get_relevant_domain_knowledge_semantic(query, top_k=3)
    sections = list(json.loads(result_json).keys())

    print(f"Selected sections: {sections}")
    print()
```

Run:
```bash
python test_semantic_selection.py
```

Expected output:
```
Query: What affects balancing electricity cost?
================================================================================
Selected sections: ['BalancingPriceDrivers', 'CurrencyInfluence', 'TariffStructure']

Query: How safe is Georgia's power supply?
================================================================================
Selected sections: ['EnergySecurityAnalysis', 'ImportDependence', 'GenerationAdequacy']

Query: როგორ მუშაობს ელექტროენერგიის ბალანსი?
================================================================================
Selected sections: ['BalancingMarketStructure', 'BalancingPriceFormation', 'BalancingMarketLogic']
```

---

## When to Implement This

### Implement NOW if:
- ❌ Users frequently ask edge case queries that keywords miss
- ❌ Georgian queries not working well
- ❌ You have capacity to add ~2GB dependency

### Implement LATER if:
- ✅ Keywords work well for 95%+ queries
- ✅ You just added better keywords (recent fixes)
- ✅ Other improvements (few-shot examples) are higher priority

### Recommendation for YOU:

**Wait 2-4 weeks**

Reasons:
1. You just fixed keyword triggers (energy security, price comparison, etc.)
2. You're about to add few-shot examples (Phase 1 - higher ROI)
3. Let the system stabilize with current changes
4. Gather more user queries to identify true edge cases
5. Then implement semantic selection for remaining 5-10% edge cases

**Timeline:**
- Week 1-2: Implement Phase 1 (few-shot examples, format rules, test suite)
- Week 3-4: Gather data, identify edge cases where keywords fail
- Week 5+: Implement semantic selection if needed

---

## Summary

**Current:** Keyword matching (fast, works for 90-95% of queries)
**Proposed:** Semantic similarity (slower, better for edge cases)
**Best:** Hybrid approach (keywords first, semantic fallback)

**Expected Improvement:** 15-20% better for edge cases, but keywords + few-shot examples should get you 90-95% of the way there first.

**Priority:** MEDIUM (implement after Phase 1 quick wins)

---

## References

- **Sentence Transformers:** https://www.sbert.net/
- **Multilingual Model:** https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Cosine Similarity:** https://en.wikipedia.org/wiki/Cosine_similarity
- **Semantic Search Tutorial:** https://www.sbert.net/examples/applications/semantic-search/README.html

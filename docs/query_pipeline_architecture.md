# Query Pipeline Architecture: End-to-End Review

This document provides a detailed technical review of the query processing pipeline, from the moment a user submits a question to the final validated answer.

## Overview
The pipeline is a multi-stage orchestrator designed to balance **speed** (via deterministic tools), **flexibility** (via SQL fallback), and **accuracy** (via truth-clamping and data enrichment).

---

## Stage-by-Stage Breakdown

### Phase 1: Ingestion & Analysis (The "Brain")

#### Stage 0: Context Preparation (`planner.prepare_context`)
*   **Purpose:** Fast, heuristic-based classification.
*   **Sub-steps:**
    1.  **Language Detection:** Determines code (`en`, `ka`, `ru`) and corresponding instructions.
    2.  **Mode Selection:** Heuristically chooses `light` (fact-seeking) vs `analyst` (causal-seeking).
    3.  **Conceptual Detection:** Determines if the query is a request for a definition (short-circuiting data paths).
*   **Decision Driver:** Keyword-based matching in `planner.py`.

#### Stage 0.2: Question Analysis (`planner.analyze_question`)
*   **Purpose:** Deep LLM-based reasoning about intent BEFORE writing code/SQL.
*   **Sub-steps:**
    1.  **Canonicalization:** Rewrites the query into a standard English analytical query.
    2.  **Routing Preference:** LLM suggests whether the query should follow a `tool`, `sql`, or `conceptual` path.
    3.  **Knowledge Retrieval:** Pulls domain segments (e.g. "Balancing Price") to ground the LLM's understanding.
*   **Decision Driver:** Structured JSON output from Gemini-1.5-Pro.

---

### Phase 2: Routing & Execution (The "Hands")

#### Stage 0.5: Tool Selection (`router.match_tool`)
*   **Purpose:** Bypassing slow SQL generation for common typed retrieval.
*   **Logic:**
    - Checks semantic similarity against a registry of tools (`get_prices`, `get_balancing_composition`, etc.).
    - If a high-confidence match is found, the system enters the **Typed Tool Path**.
*   **Decision Driver:** Vector-based similarity or deterministic keyword triggers.

#### Stage 0.6: Tool Execution & Enrichment
*   **Standard Path:** Executes the primary tool and stamps provenance.
*   **Enrichment Sub-step (New):** If the query is a "Why" question about Balancing Prices:
    1.  Calls `get_prices`.
    2.  Automatically triggers a secondary call to `get_balancing_composition`.
    3.  Merges share-composition columns into the price dataset.
*   **Decision Driver:** Intent detection (is it an explanation query?) + Metric detection (is it about balancing?).

#### Fallback Path: SQL Generation (`sql_executor.validate_and_execute`)
*   **Purpose:** Handling long-tail queries that tools can't satisfy.
*   **Sub-steps:**
    1.  **SQL Generation:** Gemini writes raw SQL for DuckDB/Supabase.
    2.  **Sanitization:** Strict whitelist check for tables.
    3.  **Auto-Repair:** If a query fails with `UndefinedColumn`, the system checks for synonyms or attempts to inject a trade-share pivot CTE.

---

### Phase 3: Analysis & Summarization (The "Voice")

#### Stage 3: Data Enrichment (`analyzer.enrich`)
*   **Purpose:** Generating analytical "finds" that simple SQL cannot express.
*   **Sub-steps:**
    1.  **Correlation Logic:** Computes Pearson coefficients (e.g. xrate vs price).
    2.  **Causal Overrides (New):** For balancing prices, it checks for contradictions. 
        - *Example:* If xrate rose but price fell, it injects a "Contradiction Guard" sentence.
    3.  **Observational Wording:** Separates factual findings ("Shares shifted") from causal claims ("Consistent with").
*   **Decision Driver:** Statistical thresholds (e.g. shifts > 0.5% are significant).

#### Stage 4: Summarization (`summarizer.summarize_data`)
*   **Purpose:** Final natural language response generation.
*   **Sub-steps:**
    1.  **Deterministic Payloads:** Merges the "Finds" from Stage 3 with raw data.
    2.  **Provenance Gate (Security):** Verifies that every number mentioned by the LLM exists in the source SQL/Tool output.
*   **Decision Driver:** Coverage score (must be > 0.9 for the gate to pass).

---

### Phase 4: Visualization

#### Stage 5: Chart Building (`chart_pipeline.build_chart`)
*   **Purpose:** Deciding the best way to visualize the result.
*   **Decision Drivers:**
    - Number of rows (2 rows = no chart).
    - Data types (Time-series = Line chart; Categories = Bar chart).
    - Analyzer hints (e.g. `add_trendlines`).

---

## Summary of Key Decision Drivers

| Decision | Driver | Mechanism |
| :--- | :--- | :--- |
| **Tool vs SQL** | Precision/Speed | Router semantic match score |
| **Conceptual vs Data** | Task Complexity | Keyword heuristics + LLM Classifier |
| **Causal Enrichment** | Intent Type | "Why/Explain" query detection in Pipeline |
| **Truth-Clamping** | Reliability | Provenance Gate coverage score |

---

## Implementation Roadmap (Causal Analysis Project)

The system is currently undergoing a phased rollout of the enhanced Causal Analysis features for balancing electricity prices.

### ✅ Phase A: Contract & Skill Files
*   **Status:** Complete
*   **Work:** Defined the `QuestionAnalysis` contract and updated developer skills for phased audits and LLM-data grounding.

### ✅ Phase B: Analyzer & Shadow Mode
*   **Status:** Complete
*   **Work:** Implemented `_generate_why_summary_payload` in `analyzer.py` with prioritized contradiction guards. Successfully tested in "Shadow Mode" to verify logic without disrupting production traffic.

### ✅ Phase C: Routing Integration
*   **Status:** Complete
*   **Work:** Integrated the `data_explanation` query type into the `router.match_tool` logic and enabled automatic composition enrichment in `pipeline.py`.

### 🚀 Phase D: Planner & Chart Integration
*   **Status:** Upcoming/In-Progress
*   **Work:** 
    *   **Planner:** Refine `planner.py` prompt instructions to better utilize enriched findings in the final narrative plan.
    *   **Charts:** Update `chart_pipeline.py` to automatically trigger specific visualizations (e.g., stacked bar charts or waterfall charts of contributions) when a "Why-Context" is detected, visually communicating the "drivers" described in the text.

### 🧪 Phase E: Evaluation & Threshold Tuning
*   **Status:** Upcoming
*   **Work:**
    *   **Accuracy Baseline:** Run the system against the `evaluation/` gold dataset to quantify improvements in causal narrative accuracy.
    *   **Threshold Finetuning:** Adjust the significance thresholds (e.g., the 0.5% share shift filter in `analyzer.py`) based on real-world sensitivity analysis.
    *   **Performance:** Audit the end-to-end latency of the dual-fetch pipeline and optimize SQL joins if necessary.

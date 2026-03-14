# Query Pipeline Architecture: End-to-End Review

This document provides a detailed technical review of the query processing pipeline, from the moment a user submits a question to the final validated answer.

## Overview
The pipeline is a multi-stage orchestrator designed to balance **speed** (via deterministic tools), **flexibility** (via SQL fallback), and **accuracy** (via truth-clamping and data enrichment).

---

## Stage-by-Stage Breakdown

## Stage-by-Stage Deep Dive: Decision Drivers & Logic

### Phase 1: Ingestion & Intent Analysis
The goal of this phase is to classify the query with minimum latency to determine the most efficient execution path.

#### Stage 0: Heuristic Context Preparation (`planner.prepare_context`)
*   **Why:** To avoid expensive LLM calls for simple language detection or conceptual short-circuits.
*   **Decisions & Drivers:**
    1.  **Language Detection:** Uses `langdetect` on the raw query string. 
        - *Direct Result:* Selects the `lang_instruction` (en/ka/ru) used for all subsequent LLM prompts.
    2.  **Mode Selection:** Scans for keywords in `ANALYTICAL_KEYWORDS` (e.g., "trend", "impact", "why").
        - *Decision:* If query contains "what is" or "list" → `light` mode. If contains "analyze" or "correlation" → `analyst` mode.
        - *Driver:* Higher priority given to "simple fact" patterns to minimize complexity for direct status checks.
    3.  **Conceptual Detection:** Scans for definition-seeking patterns (e.g., "რა არის...", "что такое...").
        - *Decision:* If positive, the pipeline **exits data paths immediately** and jumps to `summarizer.answer_conceptual`.

#### Stage 0.2: LLM Structured Analysis (`planner.analyze_question`)
*   **Why:** Heuristics can miss nuance (e.g., "Why is the list so long?" is not a data explanation).
*   **Decisions & Drivers:**
    1.  **Routing Preference:** The LLM evaluates the query against the available tool schemas and domain knowledge.
        - *Driver:* If the query targets a specific metric (e.g., "balancing price") and a known tool exists (`get_prices`) → `tool` preference.
        - *Driver:* If it requires complex joins or non-standard aggregation → `sql` preference.
    2.  **Confidence Score:** The LLM provides a confidence float (0.0 to 1.0).
        - *Logic:* If confidence < 0.7, the system treats the analysis as a "hint" rather than a command.

---

### Phase 2: Execution Path Selection
This is the most critical branching point: **Deterministic (Fast) vs. Generative (Flexible)**.

#### Stage 0.5: Deterministic Routing (`router.match_tool`)
*   **Why:** SQL generation is slow (3-5s) and prone to hallucinations. Tools are fast (<500ms) and type-safe.
*   **Decisions & Drivers:**
    1.  **Keyword Hits:** Scans for metric triggers (e.g., `has_price`, `has_tariff`).
    2.  **Semantic Similarity:** If keyword hits are inconclusive, it computes a "Semantic Score" based on term hits (e.g., "cost" counts for `get_prices`).
        - *Threshold:* `ROUTER_SEMANTIC_MIN_SCORE` (default 0.62).
        - *Decision Driver:* The "Score Gap." If the top tool's score isn't at least 0.08 higher than the second-best, the match is rejected as ambiguous to prevent misrouting.
    3.  **Date Extraction:** Uses Regex to find years (2020-2024) or months (june 2023).
        - *Special Logic:* For "Why" queries, the router **automatically expands the range** to include the previous month to allow for MoM delta calculations.

#### Stage 0.6: Execution & Enrichment Policy
*   **Why:** Raw data is often insufficient for causal analysis.
*   **Enrichment Decision:** 
    - *Condition:* If `is_explanation` is True AND tool is `get_prices` AND metric is `balancing`.
    - *Action:* Automatically trigger `get_balancing_composition`.
    - *Rationale:* Users asking "Why" about prices always need the "What" (supplier mix) to form a valid answer.

#### Stage 1/2: Generative Fallback (`sql_executor`)
*   **Why:** Handle queries like "Compare total imports of the top 3 buyers in 2022."
*   **Decision Driver:** If Stage 0.5 returns `None`, the system falls back to LLM SQL generation.
*   **Auto-Repair Logic:** If SQL execution fails with `UndefinedColumn`:
    - *Decision:* Search `context.py` for column synonyms.
    - *Action:* If "time" was used instead of "date," the system intercepts the error, replaces the code, and re-executes.

---

### Phase 3: Qualitative Analysis (`analyzer.enrich`)
*   **Why:** A table of numbers is not an "explanation." The system must compute derived metrics (correlations, shifts, trends) to provide an analytical payload to the LLM.
*   **Decisions & Drivers:**
    1.  **Semantic Intent Detection:** Scans the combined intent string and user query for causal keywords (e.g., `driver`, `cause`, `why`, `impact`) or Regex patterns (e.g., `what.*affect`). 
        - *Driver:* If a match is found, the system dynamically changes the execution plan `intent` to `correlation`.
    2.  **Statistically Significant Shifts:**
        - *Threshold:* The system filters out noise by only evaluating month-over-month balancing share shifts where the absolute delta magnitude is `>= 0.005` (0.5 percentage points).
    3.  **Contradiction Logic (Priority 1 for 'Why' queries):** 
        - computes the `price_direction` vs the `mix_pressure` (did expensive sources go up?) and `xrate_direction`.
        - *Condition:* If `price_direction` contradicts both `mix_pressure` and `xrate_direction`, or contradicts one while the other is neutral.
        - *Impact:* The analyzer **overrides** generic LLM summarization with a deterministic "Contradiction Guard" string (e.g., "The observed composition shift points in the opposite direction...") ensuring the LLM cannot hallucinate an aligned cause.
        - *Fallback:* If no share data is available, it explicitly outputs "No balancing composition data was available" rather than a false negative "No shift observed."

---

### Phase 4: Summarization & Verification (`summarizer.summarize_data`)
*   **Why:** Prevent LLM "hallucination of numbers" and ensure analytical narratives are strictly grounded in retrieved data.
*   **Decisions & Drivers:**
    1.  **Deterministic Payloads vs Generative:** If `share_summary_override` or `why_summary_override` is populated (e.g., by the Contradiction Guard), the LLM path is completely bypassed.
    2.  **Structured Generation & Strict Grounding Retry:**
        - The system prompts the LLM to generate an answer and extract claims arrays.
        - It compares every numeric token in the claims to the raw `ctx.rows`.
        - *Decision:* If validation fails the first time, the system **automatically retries** the LLM generation with `strict_grounding=True`.
    3.  **Provenance Gate (Security):** 
        - After generation (and optional retry), it calculates the `summary_provenance_coverage` (number of grounded tokens / total numeric tokens).
        - *Threshold:* If there is ANY ungrounded claim (`has_ungrounded_claim == True`), or if coverage falls below `PROVENANCE_MIN_COVERAGE`.
        - *Action:* The response is completely redacted and replaced with a conservative fallback: "I could not produce citation-grade grounding for all numeric claims..." and the confidence is pegged to 0.2.

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

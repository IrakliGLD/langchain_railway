"""
Evaluation Engine - Core logic for running quality validation tests

Can be used both as standalone script and imported by FastAPI endpoints.
"""

import json
import time
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


def load_evaluation_dataset(dataset_path: str = "evaluation_dataset.json") -> Dict[str, Any]:
    """Load the evaluation dataset from JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def validate_sql_patterns(sql: str, expected_patterns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that SQL contains expected patterns.

    Returns:
        tuple: (all_found, missing_patterns)
    """
    sql_lower = sql.lower()
    missing = []

    for pattern in expected_patterns:
        pattern_lower = pattern.lower()
        # Handle regex-like patterns - just check key terms
        if "(" in pattern and ")" in pattern:
            key_terms = re.findall(r'\w+', pattern_lower)
            if not all(term in sql_lower for term in key_terms if len(term) > 2):
                missing.append(pattern)
        else:
            if pattern_lower not in sql_lower:
                missing.append(pattern)

    return len(missing) == 0, missing


def validate_quality_criteria(
    answer: str,
    criteria: Dict[str, Any],
    sql: str
) -> Tuple[bool, List[str]]:
    """
    Validate that answer meets quality criteria.

    Returns:
        tuple: (passes, failed_checks)
    """
    failed = []
    answer_lower = answer.lower()

    # Check must_include items
    if "must_include" in criteria:
        for item in criteria["must_include"]:
            item_lower = item.lower()
            if item_lower not in answer_lower:
                failed.append(f"Missing: {item}")

    # Check must_not_include items
    if "must_not_include" in criteria:
        for item in criteria["must_not_include"]:
            item_lower = item.lower()
            if item_lower in answer_lower:
                failed.append(f"Should not include: {item}")

    # Check sentence count
    if "max_sentences" in criteria:
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) > criteria["max_sentences"]:
            failed.append(f"Too many sentences: {len(sentences)} > {criteria['max_sentences']}")

    if "min_sentences" in criteria:
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) < criteria["min_sentences"]:
            failed.append(f"Too few sentences: {len(sentences)} < {criteria['min_sentences']}")

    # Check language (basic detection)
    if "response_language" in criteria:
        expected_lang = criteria["response_language"]
        if expected_lang == "ka":
            # Georgian uses Unicode range 0x10A0-0x10FF
            has_georgian = any('\u10A0' <= c <= '\u10FF' for c in answer)
            if not has_georgian:
                failed.append("Missing Georgian script")
        elif expected_lang == "ru":
            # Russian uses Cyrillic Unicode range 0x0400-0x04FF
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in answer)
            if not has_cyrillic:
                failed.append("Missing Cyrillic script")

    return len(failed) == 0, failed


def validate_performance(
    elapsed_ms: float,
    expected_perf: Dict[str, Any],
    query_type: str
) -> Tuple[bool, str]:
    """
    Validate that performance meets expectations.

    Returns:
        tuple: (passes, message)
    """
    # Use query-specific expectations if provided
    if "total_time_ms" in expected_perf:
        threshold_str = expected_perf["total_time_ms"].replace("<", "").strip()
        threshold = float(threshold_str)
        if elapsed_ms <= threshold:
            return True, f"✓ {elapsed_ms:.0f}ms < {threshold:.0f}ms"
        else:
            return False, f"✗ {elapsed_ms:.0f}ms > {threshold:.0f}ms"

    # Use general thresholds by query type
    if query_type in ["single_value", "list"]:
        threshold = 8000
    elif query_type in ["comparison"]:
        threshold = 15000
    elif query_type in ["trend"]:
        threshold = 25000
    else:  # analyst
        threshold = 45000

    if elapsed_ms <= threshold:
        return True, f"✓ {elapsed_ms:.0f}ms < {threshold:.0f}ms"
    else:
        return False, f"✗ {elapsed_ms:.0f}ms > {threshold:.0f}ms"


def run_single_evaluation(
    query_data: Dict[str, Any],
    api_func,  # Function that takes (query_text) and returns (response_dict, elapsed_ms, error)
) -> Dict[str, Any]:
    """
    Run a single evaluation test.

    Args:
        query_data: Query definition from dataset
        api_func: Function to call API - signature: (query_text) -> (response, elapsed_ms, error)

    Returns:
        dict: Test results with all validation details
    """
    query_id = query_data["id"]
    query_type = query_data["type"]
    query_text = query_data["query"]

    # Execute query
    response, elapsed_ms, error = api_func(query_text)

    if error:
        return {
            "id": query_id,
            "type": query_type,
            "query": query_text,
            "status": "error",
            "error": error,
            "elapsed_ms": elapsed_ms
        }

    # Extract results
    sql = response.get("sql", "")
    answer = response.get("answer", "")

    # Validate SQL patterns
    sql_valid = True
    sql_missing = []
    if "expected_sql_patterns" in query_data:
        sql_valid, sql_missing = validate_sql_patterns(sql, query_data["expected_sql_patterns"])

    # Validate quality criteria
    quality_valid = True
    quality_failed = []
    if "quality_criteria" in query_data:
        quality_valid, quality_failed = validate_quality_criteria(
            answer,
            query_data["quality_criteria"],
            sql
        )

    # Validate performance
    perf_valid = True
    perf_msg = ""
    if "expected_performance" in query_data:
        perf_valid, perf_msg = validate_performance(
            elapsed_ms,
            query_data["expected_performance"],
            query_type
        )
    else:
        perf_valid, perf_msg = validate_performance(elapsed_ms, {}, query_type)

    # Determine overall status
    all_passed = sql_valid and quality_valid and perf_valid
    status = "pass" if all_passed else "fail"

    return {
        "id": query_id,
        "type": query_type,
        "query": query_text,
        "status": status,
        "sql_valid": sql_valid,
        "sql_missing": sql_missing,
        "quality_valid": quality_valid,
        "quality_failed": quality_failed,
        "performance_valid": perf_valid,
        "performance_msg": perf_msg,
        "elapsed_ms": elapsed_ms,
        "sql": sql,
        "answer": answer
    }


def filter_queries(
    queries: List[Dict[str, Any]],
    mode: str = "quick",
    query_type: Optional[str] = None,
    query_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter queries based on mode and filters.

    Args:
        queries: Full list of queries
        mode: 'full' or 'quick'
        query_type: Filter by type (single_value, list, comparison, trend, analyst)
        query_id: Filter by specific ID

    Returns:
        Filtered list of queries
    """
    if query_id:
        return [q for q in queries if q["id"] == query_id]

    if query_type:
        queries = [q for q in queries if q["type"] == query_type]

    if mode == "quick" and not query_type and not query_id:
        # Representative sample: 2 of each type
        sample_queries = []
        for qtype in ["single_value", "list", "comparison", "trend", "analyst"]:
            type_queries = [q for q in queries if q["type"] == qtype]
            sample_queries.extend(type_queries[:2])
        return sample_queries

    return queries


def generate_summary(results: List[Dict[str, Any]], dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.

    Returns:
        dict: Summary statistics
    """
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] == "error")

    # Calculate metrics by type
    by_type = {}
    for result in results:
        qtype = result["type"]
        if qtype not in by_type:
            by_type[qtype] = {"total": 0, "passed": 0, "failed": 0}
        by_type[qtype]["total"] += 1
        if result["status"] == "pass":
            by_type[qtype]["passed"] += 1
        elif result["status"] == "fail":
            by_type[qtype]["failed"] += 1

    # Performance stats
    avg_time = sum(r["elapsed_ms"] for r in results) / len(results) if results else 0
    simple_queries = [r for r in results if r["type"] in ["single_value", "list"]]
    complex_queries = [r for r in results if r["type"] in ["trend", "analyst"]]

    avg_simple = sum(r["elapsed_ms"] for r in simple_queries) / len(simple_queries) if simple_queries else 0
    avg_complex = sum(r["elapsed_ms"] for r in complex_queries) / len(complex_queries) if complex_queries else 0

    # Quality issues
    sql_issues = sum(1 for r in results if not r.get("sql_valid", True))
    quality_issues = sum(1 for r in results if not r.get("quality_valid", True))
    perf_issues = sum(1 for r in results if not r.get("performance_valid", True))

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset_version": dataset["metadata"]["version"],
        "total_queries": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / total if total > 0 else 0,
        "by_type": by_type,
        "performance": {
            "avg_time_ms": avg_time,
            "avg_simple_ms": avg_simple,
            "avg_complex_ms": avg_complex
        },
        "issues": {
            "sql_pattern_issues": sql_issues,
            "quality_issues": quality_issues,
            "performance_issues": perf_issues
        }
    }

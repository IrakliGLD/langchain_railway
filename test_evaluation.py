"""
Automated Evaluation Test Runner for Text-to-SQL Quality Testing

Purpose: Systematically test query generation and answer quality across
         all query types to validate Phase 1 optimizations don't degrade quality.

Usage:
    python test_evaluation.py --mode full              # Run all 75 queries
    python test_evaluation.py --mode quick             # Run 10 representative queries
    python test_evaluation.py --type single_value      # Run only single_value queries
    python test_evaluation.py --query sv_001           # Run specific query by ID
"""

import json
import time
import re
import argparse
from typing import Dict, List, Any, Tuple
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
APP_KEY = os.getenv("APP_SECRET_KEY", "")

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def load_evaluation_dataset() -> Dict[str, Any]:
    """Load the evaluation dataset from JSON file."""
    with open("evaluation_dataset.json", "r") as f:
        return json.load(f)


def run_query(query_text: str, timeout: int = 60) -> Tuple[Dict[str, Any], float, str]:
    """
    Execute a query against the API and return results.

    Returns:
        tuple: (response_json, execution_time_ms, error_message)
    """
    try:
        start = time.time()
        response = requests.post(
            API_URL,
            json={"query": query_text},
            headers={"X-App-Key": APP_KEY},
            timeout=timeout
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            return response.json(), elapsed_ms, ""
        else:
            return {}, elapsed_ms, f"HTTP {response.status_code}: {response.text}"

    except requests.Timeout:
        return {}, timeout * 1000, "Request timeout"
    except Exception as e:
        return {}, 0, f"Error: {str(e)}"


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
        # Handle regex-like patterns
        if "(" in pattern and ")" in pattern:
            # Simplified pattern matching - just check key terms
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


def run_single_test(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single test query and validate results.

    Returns:
        dict: Test results with all validation details
    """
    query_id = query_data["id"]
    query_type = query_data["type"]
    query_text = query_data["query"]

    print(f"\n{Colors.BOLD}[{query_id}] {query_type.upper()}{Colors.RESET}")
    print(f"Query: {query_text}")

    # Execute query
    response, elapsed_ms, error = run_query(query_text)

    if error:
        print(f"{Colors.RED}✗ FAILED: {error}{Colors.RESET}")
        return {
            "id": query_id,
            "type": query_type,
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

    # Print results
    if sql_valid:
        print(f"{Colors.GREEN}✓ SQL patterns valid{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ SQL missing patterns: {sql_missing}{Colors.RESET}")

    if quality_valid:
        print(f"{Colors.GREEN}✓ Quality criteria met{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ Quality issues: {quality_failed}{Colors.RESET}")

    if perf_valid:
        print(f"{Colors.GREEN}{perf_msg}{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{perf_msg}{Colors.RESET}")

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ PASSED{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ FAILED{Colors.RESET}")

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


def generate_report(results: List[Dict[str, Any]], dataset: Dict[str, Any]) -> None:
    """Generate a comprehensive test report."""

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

    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}EVALUATION REPORT{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Dataset Version: {dataset['metadata']['version']}")

    print(f"\n{Colors.BOLD}Overall Results:{Colors.RESET}")
    print(f"  Total queries: {total}")
    print(f"  {Colors.GREEN}Passed: {passed} ({passed/total*100:.1f}%){Colors.RESET}")
    print(f"  {Colors.RED}Failed: {failed} ({failed/total*100:.1f}%){Colors.RESET}")
    if errors > 0:
        print(f"  {Colors.YELLOW}Errors: {errors} ({errors/total*100:.1f}%){Colors.RESET}")

    print(f"\n{Colors.BOLD}Results by Query Type:{Colors.RESET}")
    for qtype, stats in sorted(by_type.items()):
        pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qtype:15s}: {stats['passed']:2d}/{stats['total']:2d} passed ({pass_rate:5.1f}%)")

    print(f"\n{Colors.BOLD}Performance:{Colors.RESET}")
    print(f"  Average response time: {avg_time:.0f}ms")
    print(f"  Simple queries avg:    {avg_simple:.0f}ms (target: <8s)")
    print(f"  Complex queries avg:   {avg_complex:.0f}ms (target: <45s)")

    # Quality issues summary
    sql_issues = sum(1 for r in results if not r.get("sql_valid", True))
    quality_issues = sum(1 for r in results if not r.get("quality_valid", True))
    perf_issues = sum(1 for r in results if not r.get("performance_valid", True))

    print(f"\n{Colors.BOLD}Issue Breakdown:{Colors.RESET}")
    print(f"  SQL pattern issues:    {sql_issues}")
    print(f"  Quality issues:        {quality_issues}")
    print(f"  Performance issues:    {perf_issues}")

    # Failed queries detail
    if failed > 0 or errors > 0:
        print(f"\n{Colors.BOLD}Failed Queries:{Colors.RESET}")
        for result in results:
            if result["status"] != "pass":
                print(f"\n  [{result['id']}] {result['type']}")
                print(f"  Query: {result['query']}")
                if result["status"] == "error":
                    print(f"  {Colors.RED}Error: {result.get('error', 'Unknown')}{Colors.RESET}")
                else:
                    if not result.get("sql_valid", True):
                        print(f"  {Colors.RED}SQL issues: {result.get('sql_missing', [])}{Colors.RESET}")
                    if not result.get("quality_valid", True):
                        print(f"  {Colors.RED}Quality issues: {result.get('quality_failed', [])}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation tests")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                        help="Test mode: full (all queries) or quick (representative sample)")
    parser.add_argument("--type", choices=["single_value", "list", "comparison", "trend", "analyst"],
                        help="Run only queries of specific type")
    parser.add_argument("--query", help="Run specific query by ID (e.g., sv_001)")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Load dataset
    dataset = load_evaluation_dataset()
    queries = dataset["queries"]

    # Filter queries based on arguments
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
        if not queries:
            print(f"{Colors.RED}Query ID '{args.query}' not found{Colors.RESET}")
            return
    elif args.type:
        queries = [q for q in queries if q["type"] == args.type]
    elif args.mode == "quick":
        # Representative sample: 2 of each type
        sample_queries = []
        for qtype in ["single_value", "list", "comparison", "trend", "analyst"]:
            type_queries = [q for q in queries if q["type"] == qtype]
            sample_queries.extend(type_queries[:2])
        queries = sample_queries

    print(f"{Colors.BOLD}Starting evaluation with {len(queries)} queries...{Colors.RESET}")

    # Run tests
    results = []
    for i, query_data in enumerate(queries, 1):
        print(f"\n{Colors.BLUE}[{i}/{len(queries)}]{Colors.RESET}", end=" ")
        result = run_single_test(query_data)
        results.append(result)
        time.sleep(0.5)  # Rate limiting

    # Generate report
    generate_report(results, dataset)

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset_version": dataset["metadata"]["version"],
            "total_queries": len(results),
            "results": results
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n{Colors.GREEN}Results saved to {args.output}{Colors.RESET}")


if __name__ == "__main__":
    main()

"""
Metrics tracking for observability.

Simple in-memory metrics for monitoring application performance.
"""
import logging

log = logging.getLogger("Enai")


class Metrics:
    """Simple metrics tracker for observability."""

    def __init__(self):
        self.request_count = 0
        self.llm_call_count = 0
        self.sql_query_count = 0
        self.error_count = 0
        self.total_llm_time = 0.0
        self.total_sql_time = 0.0
        self.total_request_time = 0.0

    def log_request(self, duration: float):
        """Log a request with its duration."""
        self.request_count += 1
        self.total_request_time += duration
        avg_time = self.total_request_time / self.request_count
        log.info(f"📊 Metrics: requests={self.request_count}, avg_time={avg_time:.2f}s")

    def log_llm_call(self, duration: float):
        """Log an LLM API call with its duration."""
        self.llm_call_count += 1
        self.total_llm_time += duration

    def log_sql_query(self, duration: float):
        """Log a SQL query with its duration."""
        self.sql_query_count += 1
        self.total_sql_time += duration

    def log_error(self):
        """Log an error occurrence."""
        self.error_count += 1

    def get_stats(self) -> dict:
        """
        Get current metrics statistics.

        Returns:
            Dictionary with all metrics
        """
        return {
            "requests": self.request_count,
            "llm_calls": self.llm_call_count,
            "sql_queries": self.sql_query_count,
            "errors": self.error_count,
            "avg_request_time": self.total_request_time / max(1, self.request_count),
            "avg_llm_time": self.total_llm_time / max(1, self.llm_call_count),
            "avg_sql_time": self.total_sql_time / max(1, self.sql_query_count),
        }


# Global metrics instance
metrics = Metrics()

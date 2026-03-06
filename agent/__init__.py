"""
Agent pipeline package.

Decomposes the monolithic ask_post handler into testable stages:
  planner -> sql_executor -> analyzer -> summarizer -> chart_pipeline

Orchestrated by pipeline.process_query().
"""

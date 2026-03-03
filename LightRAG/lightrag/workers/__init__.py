"""
LightRAG Workers Package

This package contains Celery-based background workers for processing
GitHub pull request webhooks and other async tasks.

Components:
- celery_app: Celery application configuration
- pr_worker: Pull request processing tasks
- run_worker: Worker entry point script
"""

from .celery_app import celery_app

__all__ = ["celery_app"]
